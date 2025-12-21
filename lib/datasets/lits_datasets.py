import math
import os

import numpy as np
import torch

from monai import data, transforms
from utils.utils_new import LogPadedd
import json
from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
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
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(
            indices[self.rank: self.total_size: self.num_replicas]
        )

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
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
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_files, validation_files = (
        load_decathlon_datalist(datalist_json, True, "training"),
        load_decathlon_datalist(datalist_json, True, "validation"),
    )

    # train_files, validation_files = datafold_read(
    #     datalist=datalist_json, basedir=data_dir, fold=args.fold
    # )

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
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest", "nearest"),
            ),
            transforms.ScaleIntensityRanged(  # 图像值变化a->b（类似clip）
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label", "label_vessel"],
                spatial_size=(-1, -1, args.roi_z),
                method="end",
            ),
            transforms.RandCropByPosNegLabeld(  # 按照特定阴性阳性比例裁剪子图
                keys=["image", "label", "label_vessel"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=args.RandFlipd_prob,
                spatial_axis=0,
            ),  # 随机水平翻转
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=args.RandFlipd_prob,
                spatial_axis=1,
            ),
            transforms.RandFlipd(
                keys=["image", "label", "label_vessel"],
                prob=args.RandFlipd_prob,
                spatial_axis=2,
            ),
            transforms.RandRotate90d(
                keys=["image", "label", "label_vessel"],
                prob=args.RandRotate90d_prob,
                max_k=3,
            ),  # 随机旋转
            transforms.RandScaleIntensityd(
                keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob
            ),  # 随机放大图像值
            transforms.RandShiftIntensityd(
                keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob
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
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label_vessel"], spatial_size=(-1, -1, args.roi_z)
            ),
            LogPadedd(keys="image"),
            transforms.ToTensord(keys=["image", "label", "label_vessel"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "label_vessel"]),
            transforms.AddChanneld(keys=["image", "label", "label_vessel"]),
            transforms.Spacingd(
                keys=["image", "label_vessel"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys=["image", "label_vessel"], spatial_size=(-1, -1, args.roi_z)
            ),
            LogPadedd(keys="image"),
            transforms.ToTensord(keys=["image", "label", "label_vessel"]),
        ]
    )

    if args.test_mode:
        # test_json = load_decathlon_datalist(datalist_json, True, "training")  # 做测试
        test_json = load_decathlon_datalist(datalist_json, True, "test")

        test_ds = data.Dataset(data=test_json, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        if args.debug:
            train_ds = data.Dataset(data=train_files[:10], transform=train_transform)
            validation_files = validation_files[:1]
        elif args.use_normal_dataset:
            train_ds = data.Dataset(data=train_files, transform=train_transform)
        else:
            # train_ds = data.CacheDataset(
            #     data=train_files,
            #     transform=train_transform,
            #     cache_num=24,
            #     cache_rate=1.0,
            #     num_workers=args.workers,
            # )
            train_ds = data.Dataset(
                data=train_files,
                transform=train_transform
            )

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=False,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=False,
        )
        loader = [train_loader, val_loader]

    return loader

def get_loader_lmdb(args):
    from lib.lmdb_dataset import lmdb_io, transform
    from torch.utils.data import DataLoader

    num_multiphase = 1
    phase_id = [0]

    train_transforms_my = transform.Compose_my([
        transform.Normalize(min_value=args.a_min, max_value=args.a_max, clip_min=args.b_min, clip_max=args.b_max),
        transform.LabelMapping(),
        transform.RandomCrop_xy_ratio(output_size=[args.roi_x, args.roi_y, args.roi_z], num_sample=4,
                                      pos_ratio=1, neg_ratio=1,
                                      random_state=np.random.RandomState(47)),
        transform.RandomFlip(axis_prob=args.RandFlipd_prob, spatial_axis=2),
        transform.RandomFlip(axis_prob=args.RandFlipd_prob, spatial_axis=3),
        transform.RandomFlip(axis_prob=args.RandFlipd_prob, spatial_axis=4),
        transform.RandScaleIntensity(factors=0.1, prob=args.RandScaleIntensityd_prob),
        transform.RandShiftIntensity(offsets=0.1, prob=args.RandShiftIntensityd_prob)
    ])

    val_transforms_my = transform.Compose_my([
        transform.Normalize(min_value=args.a_min, max_value=args.a_max, clip_min=args.b_min, clip_max=args.b_max),
        transform.LabelMapping(),
    ])

    train_dataset = lmdb_io.Dataset_3D(input_dir=args.train_dir, num_multiphase=num_multiphase,
                                       out_phase_id=phase_id,
                                       dataset_type='train',
                                       transforms=train_transforms_my)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D(input_dir=args.val_dir, num_multiphase=num_multiphase,
                                     out_phase_id=phase_id,
                                     dataset_type='val',
                                     transforms=val_transforms_my)
    val_sampler = Sampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False
    )
    return [train_loader, val_loader]

def get_loader_lmdb_monai(args):
    from monai.transforms import (
        Compose, ScaleIntensityRanged, RandCropByPosNegLabeld,
        RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
        ToTensord, EnsureTyped, ResizeWithPadOrCropd, SpatialPadd,RandRotate90d
    )
    from monai.data import DataLoader
    from lib.lmdb_dataset import lmdb_io
    train_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=(-1, -1, args.roi_z),
            method="end",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(
            keys=["image", "label"],
            prob=args.RandRotate90d_prob,
            max_k=3,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ToTensord(keys=["image", "label"]),
    ])

    val_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label"], spatial_size=(-1, -1, args.roi_z)
        ),
        LogPadedd(keys="image"),
        ToTensord(keys=["image", "label"]),
    ])

    num_multiphase = 1
    phase_id = [0]

    train_dataset = lmdb_io.Dataset_3D_my(input_dir=args.train_dir, num_multiphase=num_multiphase,
                                          out_phase_id=phase_id,
                                          dataset_type='train',
                                          transforms=train_transform)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D_my(input_dir=args.val_dir, num_multiphase=num_multiphase,
                                        out_phase_id=phase_id,
                                        dataset_type='val',
                                        transforms=val_transform)
    val_sampler = Sampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False
    )
    return [train_loader, val_loader]

def get_loader_couinaud_lmdb_monai(args):
    from monai.transforms import (
        Compose, ScaleIntensityRanged, RandCropByPosNegLabeld,
        RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
        ToTensord, EnsureTyped, ResizeWithPadOrCropd, SpatialPadd,RandRotate90d
    )
    from monai.data import DataLoader
    from lib.lmdb_dataset import lmdb_io


    # if args.test_mode:
    #     test_transform = Compose([
    #         ScaleIntensityRanged(
    #             keys=["image"],
    #             a_min=args.a_min,
    #             a_max=args.a_max,
    #             b_min=args.b_min,
    #             b_max=args.b_max,
    #             clip=True
    #         ),
    #         SpatialPadd(
    #             keys=["image", "label_vessel"], spatial_size=(-1, -1, args.roi_z)
    #         ),
    #         LogPadedd(keys="image"),
    #         ToTensord(keys=["image", "label", "label_vessel"]),
    #     ])
    #     num_multiphase = 1
    #     phase_id = [0]
    #     test_dataset = lmdb_io.Dataset_3D_my_couinaud(input_dir=args.test_dir, num_multiphase=num_multiphase,
    #                                                  out_phase_id=phase_id,
    #                                                  dataset_type='val',
    #                                                  transforms=test_transform)
    #     test_sampler = Sampler(test_dataset)
    #
    #     test_loader = DataLoader(
    #         dataset=test_dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         sampler=test_sampler,
    #         pin_memory=False
    #     )
    #     return test_loader
    # else:
    train_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label", "label_vessel"],
            spatial_size=(-1, -1, args.roi_z),
            method="end",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label", "label_vessel"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label", "label_vessel"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label", "label_vessel"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label", "label_vessel"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(
            keys=["image", "label", "label_vessel"],
            prob=args.RandRotate90d_prob,
            max_k=3,
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ToTensord(keys=["image", "label", "label_vessel"]),
    ])

    val_transform = Compose([
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True
        ),
        SpatialPadd(
            keys=["image", "label_vessel"], spatial_size=(-1, -1, args.roi_z)
        ),
        LogPadedd(keys="image"),
        ToTensord(keys=["image", "label", "label_vessel"]),
    ])

    num_multiphase = 1
    phase_id = [0]

    train_dataset = lmdb_io.Dataset_3D_my_couinaud(input_dir=args.train_dir, num_multiphase=num_multiphase,
                                          out_phase_id=phase_id,
                                          dataset_type='train',
                                          transforms=train_transform)
    train_sampler = Sampler(train_dataset)

    val_dataset = lmdb_io.Dataset_3D_my_couinaud(input_dir=args.val_dir, num_multiphase=num_multiphase,
                                        out_phase_id=phase_id,
                                        dataset_type='val',
                                        transforms=val_transform)
    val_sampler = Sampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=False
    )
    return [train_loader, val_loader]

def get_loader_2D(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_files, validation_files = datafold_read(
        datalist=datalist_json, basedir=data_dir, fold=args.fold
    )
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image", "label"]
            ),  # 根据数据类型选择对应的读取器读取数据
            transforms.EnsureChannelFirstd(
                keys=["image", "label"]
            ),  # 把series维度变成channel
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"), # 统一图像方向
            # z轴不需要改变
            transforms.Spacingd(  # 按照pixdim对图像进行重采样
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, -1),
                mode=("bilinear", "nearest"),
            ),
            transforms.ScaleIntensityRanged(  # 图像值变化a->b（类似clip）
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            # transforms.CropForegroundd(keys=["image", "label"], source_key="image"), # 矩形裁剪，按照值>0
            # z轴不需要改变
            # transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, -1)),
            transforms.RandCropByPosNegLabeld(  # 按照特定阴性阳性比例裁剪子图
                keys=["image", "label"],
                label_key="label",
                # spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                # z轴是1，代表是一层CT
                spatial_size=(args.roi_x, args.roi_y, 1),
                pos=1,
                neg=1,
                num_samples=16,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(
                keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0
            ),  # 随机水平翻转
            transforms.RandFlipd(
                keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1
            ),
            transforms.RandFlipd(
                keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2
            ),
            # transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),  # 随机旋转
            transforms.RandScaleIntensityd(
                keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob
            ),  # 随机放大图像值
            transforms.RandShiftIntensityd(
                keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob
            ),  # 随机偏移图像值
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"]
            ),  # 把series维度变成channel
            transforms.Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, -1),
                mode=("bilinear"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"]
            ),  # 把series维度变成channel
            transforms.Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, -1),
                mode=("bilinear"),
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_ds = data.Dataset(data=validation_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=train_files, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=train_files,
                transform=train_transform,
                cache_num=24,
                cache_rate=1.0,
                num_workers=args.workers,
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
        )
        loader = [train_loader, val_loader]

    return loader
