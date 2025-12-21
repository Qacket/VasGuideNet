import time

from monai.data import load_decathlon_datalist
import torch
import math
import numpy as np
import torch.distributed as dist
import warnings

from utils.utils_new import LogPadedd

warnings.filterwarnings("ignore")


class Sampler(torch.utils.data.Sampler):
    # dataset: 数据集对象。
    # num_replicas: 总共有几个进程/worker（通常等于 GPU 数目）。
    # rank: 当前进程的编号。
    # shuffle: 是否在每轮打乱数据。
    # make_even: 是否保证每个进程拿到一样多的数据（重要！防止数据不均衡）。
    def __init__(
            self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True
    ):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        print(rank)
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # num_samples：每个进程理论上应该采样的样本数（向上取整，确保每个进程都有样本）；
        # total_size：表示总样本数 = 每个进程的样本数 × 进程数；
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # 记录当前 rank 的真实数据长度（即采样后实际返回的样本数量），用于后续统计或评估。
        indices = list(range(len(self.dataset)))
        self.valid_length = len(
            indices[self.rank: self.total_size: self.num_replicas]
        )

    def __iter__(self):
        # 是否打乱
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # 是否补齐样本
        # 当数据量不是总进程数的整数倍时：
        # 如果差距小，直接前面的补一下；
        # 如果差距大，随机补齐一些样本；
        # 目的是 确保每个 rank 的 batch size 相同，避免训练中断。
        print("len(indices)", len(indices))
        print("total_size", self.total_size)
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(
                        low=0, high=len(indices), size=self.total_size - len(indices)
                    )
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        print("len(indices)", len(indices))
        print("total_size", self.total_size)
        # 划分当前 rank 的索引
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        print("len(indices)", len(indices))
        print("self.num_samples", self.num_samples)
        print("end ... ")
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# 旧的方法读loader
def test():
    from monai import data, transforms
    from monai.transforms.transform import MapTransform
    from monai.config import IndexSelection, KeysCollection, SequenceStr
    class LogPadedd(MapTransform):
        def __init__(
                self,
                keys: KeysCollection,
        ) -> None:
            super().__init__(keys)

        def __call__(self, data):
            d = dict(data)
            for key in self.key_iterator(data):
                trans = d[key].applied_operations
                d[key].meta["padded"] = trans[-1]["extra_info"]["padded"]
                # print(d[key].meta['padded'])
            return d

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=4,
        rank=4,
    )
    # 图像重采样的目标空间间距（单位 mm），即图像标准化到的 spacing（体素尺寸）
    space_x = 1.5
    space_y = 1.5
    space_z = 2
    # 对图像进行强度归一化（intensity normalization），将[-10, 225] 线性映射到 [0, 1]
    a_min = -10.0
    a_max = 225.0
    b_min = 0.0
    b_max = 1.0
    # 用于裁剪patch的大小（3D块）
    roi_x = 96
    roi_y = 96
    roi_z = 32
    RandFlipd_prob = 0.2
    RandRotate90d_prob = 0.2
    RandScaleIntensityd_prob = 0.1
    RandShiftIntensityd_prob = 0.1

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label", "label_vessel"]
            ),  # 根据数据类型选择对应的读取器读取数据
            transforms.AddChanneld(
                keys=["image", "label", "label_vessel"]
            ),  # 在最前添加通道维度
            transforms.Spacingd(  # 按照pixdim对图像进行重采样
                keys=["image", "label", "label_vessel"],
                pixdim=(space_x, space_y, space_z),
                mode=("bilinear", "nearest", "nearest"),
            ),
            transforms.ScaleIntensityRanged(  # 图像值变化a->b（类似clip）
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label", "label_vessel"],
                spatial_size=(-1, -1, roi_z),
                method="end",
            ),
            transforms.RandCropByPosNegLabeld(  # 按照特定阴性阳性比例裁剪子图
                keys=["image", "label", "label_vessel"],
                label_key="label",
                spatial_size=(roi_x, roi_y, roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=RandFlipd_prob,
                spatial_axis=0,
            ),  # 随机水平翻转
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=RandFlipd_prob,
                spatial_axis=1,
            ),
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=RandFlipd_prob,
                spatial_axis=2,
            ),
            transforms.RandRotate90d(
                keys=["image", "label", "label_vessel"],
                prob=RandRotate90d_prob,
                max_k=3,
            ),  # 随机旋转
            transforms.RandScaleIntensityd(
                keys="image", factors=0.1, prob=RandScaleIntensityd_prob
            ),  # 随机放大图像值
            transforms.RandShiftIntensityd(
                keys="image", offsets=0.1, prob=RandShiftIntensityd_prob
            ),  # 随机偏移图像值
            transforms.ToTensord(keys=["image", "label", "label_vessel"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "label_vessel"]),
            transforms.AddChanneld(keys=["image", "label", "label_vessel"]),
            transforms.Spacingd(
                keys=["image", "label_vessel"],
                pixdim=(space_x, space_y, space_z),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label_vessel"], spatial_size=(-1, -1, roi_z)
            ),
            LogPadedd(keys="image"),
            transforms.ToTensord(keys=["image", "label", "label_vessel"]),
        ]
    )

    # datalist_json = "/data0/scj/code_final/lib/datasets/data_json/PuJian_Vessel_dataset_new.json"
    datalist_json = "/data0/scj/code_final/lib/datasets/data_json/couinaud_dataset_private.json"
    train_files, validation_files = (
        load_decathlon_datalist(datalist_json, True, "training"),
        load_decathlon_datalist(datalist_json, True, "validation"),
    )
    # print(train_files)
    # print(len(train_files))
    # print(train_files[0])

    train_ds = data.Dataset(
        data=train_files,
        transform=train_transform
    )
    train_sampler = Sampler(train_ds)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=8,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        sampler=val_sampler,
        pin_memory=True
    )

    # start_time = time.time()
    # for idx, batch_data in enumerate(train_loader):
    #     if isinstance(batch_data, list):
    #         data, target = batch_data
    #     else:
    #         data, target = batch_data["image"], batch_data["label"]
    #         print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
    #         print("22222222222222222", data.shape, target.shape)
    #     break

    start_time = time.time()
    for idx, batch_data in enumerate(val_loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            label_vessel = batch_data["label_vessel"]
            print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
            print("22222222222222222", data.shape, target.shape)
            print("33333333333333333", label_vessel.shape)
        break


# 测试lmdb存储数据
def test2():
    from lib.lmdb_dataset import lmdb_io, transform
    from monai import data
    from einops import rearrange
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=4,
        rank=4,
    )

    space_x = 1.5
    space_y = 1.5
    space_z = 2
    # 对图像进行强度归一化（intensity normalization），将[-10, 225] 线性映射到 [0, 1]
    a_min = -10.0
    a_max = 225.0
    b_min = 0.0
    b_max = 1.0
    # 用于裁剪patch的大小（3D块）
    roi_x = 96
    roi_y = 96
    roi_z = 32
    num_sample = 4
    pos_ratio = 1
    neg_ratio = 1
    train_transforms_my = transform.Compose_my([
        transform.Normalize(min_value=a_min, max_value=a_max, clip_min=b_min, clip_max=b_max),
        transform.LabelMapping(),
        transform.RandomCrop_xy_ratio(output_size=[roi_x, roi_y, roi_z], num_sample=num_sample,
                                      pos_ratio=pos_ratio, neg_ratio=neg_ratio,
                                      random_state=np.random.RandomState(47)),
        transform.RandomFlip(axis_prob=0.2, spatial_axis=2),
        transform.RandomFlip(axis_prob=0.2, spatial_axis=3),
        transform.RandomFlip(axis_prob=0.2, spatial_axis=4),
        transform.RandScaleIntensity(factors=0.1, prob=0.1),
        transform.RandShiftIntensity(offsets=0.1, prob=0.1)
    ])

    val_transforms_my = transform.Compose_my([
        transform.Normalize(min_value=a_min, max_value=a_max, clip_min=b_min, clip_max=b_max),
        transform.LabelMapping(),
    ])

    train_dir = "/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private/train/"
    val_dir = "/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private/val/"

    num_multiphase = 1
    phase_id = [0]
    train_dataset = lmdb_io.Dataset_3D(input_dir=train_dir, num_multiphase=num_multiphase,
                                       out_phase_id=phase_id,
                                       dataset_type='train',
                                       transforms=train_transforms_my)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D(input_dir=val_dir, num_multiphase=num_multiphase,
                                     out_phase_id=phase_id,
                                     dataset_type='val',
                                     transforms=val_transforms_my)
    val_sampler = Sampler(val_dataset)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=False
    )

    start_time = time.time()
    device = "cpu"
    for i, (inputs, targets) in enumerate(train_loader):
        print("111111111111111111", inputs.shape, targets.shape)
        inputs = rearrange(inputs, "b n c w h d -> (b n) c w h d").to(device, non_blocking=True).contiguous().float()
        targets = rearrange(targets, "b n c w h d -> (b n) c w h d").to(device, non_blocking=True).contiguous().long()
        print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
        print("22222222222222222", inputs.shape, targets.shape)
        break

    for i, (inputs, targets, _, _) in enumerate(val_loader):
        inputs = inputs.to(device, non_blocking=True).contiguous().float()
        targets = targets.to(device, non_blocking=True).contiguous().long()
        print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
        print("22222222222222222", inputs.shape, targets.shape)
        break

# 测试lmdb存储数据 transform 使用的是monai
def test3():

    from monai.transforms import (
        Compose, ScaleIntensityRanged, RandCropByPosNegLabeld,
        RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
        ToTensord, EnsureTyped, ResizeWithPadOrCropd
    )
    from monai.data import DataLoader
    from lib.lmdb_dataset import lmdb_io
    from einops import rearrange

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=4,
        rank=4,
    )

    space_x = 1.5
    space_y = 1.5
    space_z = 2
    # 对图像进行强度归一化（intensity normalization），将[-10, 225] 线性映射到 [0, 1]
    a_min = -10.0
    a_max = 225.0
    b_min = 0.0
    b_max = 1.0
    # 用于裁剪patch的大小（3D块）
    roi_x = 96
    roi_y = 96
    roi_z = 32
    num_sample = 4
    pos_ratio = 1
    neg_ratio = 1
    train_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=( roi_x, roi_y, roi_z),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True
        ),
        ToTensord(keys=["image", "label"]),
    ])

    num_multiphase = 1
    phase_id = [0]

    train_dir = "/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private/train/"
    val_dir = "/data0/scj/datasets/肝八段数据集/Vessel_lmdb/Vessel_private/val/"

    train_dataset = lmdb_io.Dataset_3D_my(input_dir=train_dir, num_multiphase=num_multiphase,
                                          out_phase_id=phase_id,
                                          dataset_type='train',
                                          transforms=train_transform)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D_my(input_dir=val_dir, num_multiphase=num_multiphase,
                                        out_phase_id=phase_id,
                                        dataset_type='val',
                                        transforms=val_transform)
    val_sampler = Sampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        pin_memory=False
    )

    start_time = time.time()
    device = "cpu"
    for i, batch_data in enumerate(train_loader):
        inputs, targets = batch_data["image"], batch_data["label"]
        print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
        print("22222222222222222", inputs.shape, targets.shape)
        break

    for i, batch_data in enumerate(val_loader):
        inputs, targets = batch_data["image"], batch_data["label"]
        print("22222222222222222", inputs.shape, targets.shape)
        break

# 测试lmdb存储数据 transform 使用的是monai 针对的是couinaud
def test4():

    from monai.transforms import (
        Compose, ScaleIntensityRanged, RandCropByPosNegLabeld,
        RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
        ToTensord, EnsureTyped, ResizeWithPadOrCropd, SpatialPadd,RandRotate90d
    )
    from monai.data import DataLoader
    from lib.lmdb_dataset import lmdb_io
    from einops import rearrange

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=4,
        rank=4,
    )

    space_x = 1.5
    space_y = 1.5
    space_z = 2
    # 对图像进行强度归一化（intensity normalization），将[-10, 225] 线性映射到 [0, 1]
    a_min = -10.0
    a_max = 225.0
    b_min = 0.0
    b_max = 1.0
    # 用于裁剪patch的大小（3D块）
    roi_x = 96
    roi_y = 96
    roi_z = 32
    num_sample = 4
    pos_ratio = 1
    neg_ratio = 1

    RandFlipd_prob = 0.2
    RandRotate90d_prob = 0.2
    RandScaleIntensityd_prob = 0.1
    RandShiftIntensityd_prob = 0.1


    train_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label", "label_vessel"],
            spatial_size=(-1, -1, roi_z),
            method="end",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label", "label_vessel"],
            label_key="label",
            spatial_size=( roi_x, roi_y, roi_z),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        RandRotate90d(
            keys=["image", "label", "label_vessel"],
            prob=RandRotate90d_prob,
            max_k=3,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label", "label_vessel"], spatial_size=(-1, -1, roi_z)
        ),
        LogPadedd(keys="image"),
        ToTensord(keys=["image", "label", "label_vessel"]),
    ])


    num_multiphase = 1
    phase_id = [0]
    train_dir = "/data0/scj/datasets/肝八段数据集/Couinaud_lmdb/Couinaud_private/fold1/train/"
    val_dir = "/data0/scj/datasets/肝八段数据集/Couinaud_lmdb/Couinaud_private/fold1/val/"

    train_dataset = lmdb_io.Dataset_3D_my_couinaud(input_dir=train_dir, num_multiphase=num_multiphase,
                                          out_phase_id=phase_id,
                                          dataset_type='train',
                                          transforms=train_transform)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D_my_couinaud(input_dir=val_dir, num_multiphase=num_multiphase,
                                        out_phase_id=phase_id,
                                        dataset_type='val',
                                        transforms=val_transform)
    val_sampler = Sampler(val_dataset)



    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=6,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=val_sampler,
        pin_memory=False
    )

    start_time = time.time()
    device = "cpu"
    for i, batch_data in enumerate(train_loader):
        inputs, targets = batch_data["image"], batch_data["label"]
        label_vessel = batch_data["label_vessel"]
        print("Batch loaded in {:.2f} seconds".format(time.time() - start_time))
        print("22222222222222222", inputs.shape, targets.shape)
        print("33333333333333333", label_vessel.shape)
        break

    for i, batch_data in enumerate(val_loader):
        inputs, targets = batch_data["image"], batch_data["label"]
        label_vessel = batch_data["label_vessel"]
        affine_matrix = batch_data["affine_matrix"]
        name = batch_data["name"]
        print("22222222222222222", inputs.shape, targets.shape)
        print("33333333333333333", label_vessel.shape)
        print("444444444444444444", affine_matrix.numpy()[0], affine_matrix.numpy()[0].shape)
        print("5555555555555555", name[0])

        # 打印 targets（主标签）中的类别分布
        unique_targets, counts_targets = torch.unique(targets, return_counts=True)
        print("🎯 targets 标签分布:")
        for val, count in zip(unique_targets.cpu().numpy(), counts_targets.cpu().numpy()):
            print(f"  类别 {val}: {count} 个像素")

        # 打印 label_vessel（血管标签）中的类别分布
        unique_vessel, counts_vessel = torch.unique(label_vessel, return_counts=True)
        print("🩸 label_vessel 标签分布:")
        for val, count in zip(unique_vessel.cpu().numpy(), counts_vessel.cpu().numpy()):
            print(f"  值 {val}: {count} 个像素（1表示血管）")
        break

if __name__ == "__main__":
    # test()
    # test2()
    # test3()
    test4()