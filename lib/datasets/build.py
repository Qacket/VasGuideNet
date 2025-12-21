import sys
import os
import traceback
import torch

sys.path.append(os.path.dirname(sys.path[0]))
from lib.datasets.data_load_vessel import PNG_DATA_LOADER_VESSEL

from monai import transforms, data
from monai.data import load_decathlon_datalist
from lib.utils.common_function import LogPadedd
from utils.FrangiFilter import FrangiFilter3D
import numpy as np
import math
from monai.transforms import Lambdad, Transform


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
            indices[self.rank : self.total_size : self.num_replicas]
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
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_transforms(cfg, is_train=True):
    if is_train:
        train_transforms = transforms.Compose(
            [
                # 根据数据类型选择对应的读取器读取数据
                transforms.LoadImaged(keys=cfg.MONAI.TRANSFORMS.KEYS, image_only=False),
                # 在最前添加通道维度
                transforms.EnsureChannelFirstd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    channel_dim="no_channel",  # 添加通道维度，适应新的API
                ),
                # 统一图像方向
                transforms.Orientationd(keys=cfg.MONAI.TRANSFORMS.KEYS, axcodes="RAS"),
                # 按照pixdim对图像进行重采样
                transforms.Spacingd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    pixdim=cfg.MONAI.TRANSFORMS.SPACING,
                    mode=("trilinear", "nearest"),  # bicubic
                ),
                # 图像值变化a->b（类似clip）
                transforms.ScaleIntensityRanged(
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    a_min=cfg.MONAI.TRANSFORMS.WINDOW[0],
                    a_max=cfg.MONAI.TRANSFORMS.WINDOW[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=cfg.MONAI.TRANSFORMS.CLIP,
                ),
                # transforms.CropForegroundd(keys=cfg.MONAI.TRANSFORMS.KEYS, source_key="image"), # 矩形裁剪，按照值>0
                transforms.SpatialPadd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    spatial_size=cfg.MONAI.TRANSFORMS.ROI,
                    method="end",
                ),
                transforms.RandCropByPosNegLabeld(  # 按照特定阴性阳性比例裁剪子图
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    image_key=cfg.MONAI.TRANSFORMS.KEYS[0],
                    label_key=cfg.MONAI.TRANSFORMS.KEYS[1],
                    spatial_size=cfg.MONAI.TRANSFORMS.ROI,
                    pos=1,
                    neg=0,
                    num_samples=4,  # 这里的数值会影响BatchSize
                    allow_smaller=False,
                ),
                # transforms.RandFlipd(
                #     keys=cfg.MONAI.TRANSFORMS.KEYS,
                #     prob=cfg.MONAI.TRANSFORMS.RAND.FILP_PROB,
                #     spatial_axis=0,
                # ),
                # transforms.RandFlipd(
                #     keys=cfg.MONAI.TRANSFORMS.KEYS,
                #     prob=cfg.MONAI.TRANSFORMS.RAND.FILP_PROB,
                #     spatial_axis=1,
                # ),
                # transforms.RandFlipd(
                #     keys=cfg.MONAI.TRANSFORMS.KEYS,
                #     prob=cfg.MONAI.TRANSFORMS.RAND.FILP_PROB,
                #     spatial_axis=2,
                # ),
                # transforms.RandScaleIntensityd(
                #     keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                #     factors=0.1,
                #     prob=cfg.MONAI.TRANSFORMS.RAND.SCALE_PROB,
                # ),  # 随机放大图像值
                # transforms.RandShiftIntensityd(
                #     keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                #     offsets=0.1,
                #     prob=cfg.MONAI.TRANSFORMS.RAND.SHIFT_PROB,
                # ),  # 随机偏移图像值
                transforms.ToTensord(keys=cfg.MONAI.TRANSFORMS.KEYS),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=cfg.MONAI.TRANSFORMS.KEYS, image_only=False),
                transforms.EnsureChannelFirstd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    channel_dim="no_channel",  # 添加通道维度，适应新的API
                ),  # 在最前添加通道维度
                # transforms.AddChanneld(keys=cfg.MONAI.TRANSFORMS.KEYS),
                transforms.Orientationd(keys=cfg.MONAI.TRANSFORMS.KEYS, axcodes="RAS"),
                transforms.Spacingd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    pixdim=cfg.MONAI.TRANSFORMS.SPACING,
                    mode=("trilinear"),
                ),
                transforms.ScaleIntensityRanged(  # 图像值变化a->b（类似clip）
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    a_min=cfg.MONAI.TRANSFORMS.WINDOW[0],
                    a_max=cfg.MONAI.TRANSFORMS.WINDOW[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=cfg.MONAI.TRANSFORMS.CLIP,
                ),
                # transforms.CropForegroundd(keys=cfg.MONAI.TRANSFORMS.KEYS, source_key="image"),
                transforms.SpatialPadd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    spatial_size=cfg.MONAI.TRANSFORMS.ROI,
                    method="end",
                ),
                LogPadedd(keys=cfg.MONAI.TRANSFORMS.KEYS[0]),
                transforms.ToTensord(keys=cfg.MONAI.TRANSFORMS.KEYS),
            ]
        )

        return [train_transforms, val_transforms]

    else:
        test_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=cfg.MONAI.TRANSFORMS.KEYS, image_only=False),
                transforms.EnsureChannelFirstd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS,
                    channel_dim="no_channel",  # 添加通道维度，适应新的API
                ),  # 在最前添加通道维度
                # transforms.AddChanneld(keys=cfg.MONAI.TRANSFORMS.KEYS),
                transforms.Orientationd(keys=cfg.MONAI.TRANSFORMS.KEYS, axcodes="RAS"),
                transforms.Spacingd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    pixdim=cfg.MONAI.TRANSFORMS.SPACING,
                    mode=("trilinear"),
                ),
                transforms.ScaleIntensityRanged(  # 图像值变化a->b（类似clip）
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    a_min=cfg.MONAI.TRANSFORMS.WINDOW[0],
                    a_max=cfg.MONAI.TRANSFORMS.WINDOW[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=cfg.MONAI.TRANSFORMS.CLIP,
                ),
                # transforms.CropForegroundd(keys=cfg.MONAI.TRANSFORMS.KEYS, source_key="image"),
                transforms.SpatialPadd(
                    keys=cfg.MONAI.TRANSFORMS.KEYS[0],
                    spatial_size=cfg.MONAI.TRANSFORMS.ROI,
                ),
                LogPadedd(keys=cfg.MONAI.TRANSFORMS.KEYS[0]),
                transforms.ToTensord(keys=cfg.MONAI.TRANSFORMS.KEYS),
            ]
        )
        return test_transforms


def build_dataloader_monai(cfg, is_train=False):
    datalist_json = os.path.join(cfg.MONAI.JSON_PATH, cfg.MONAI.NAME + ".json")
    if is_train:

        def custom_collate_fn(batch):
            # 过滤掉无效样本（例如，尺寸为0的样本）
            valid_samples = []
            for pairs in batch:
                for item in pairs:
                    if (
                        item[cfg.MONAI.TRANSFORMS.KEYS[0]].shape[1:]
                        != cfg.MONAI.TRANSFORMS.ROI
                    ):
                        print(
                            item["image_meta_dict"]["filename_or_obj"], " is  ERROR !"
                        )
                    else:
                        # valid_samples.append(FrangiFilter3D(item))
                        valid_samples.append(item)
            if len(valid_samples) == 0:
                return None  # 或者其他处理方式
            return torch.utils.data.dataloader.default_collate(valid_samples)

        # datalist = load_decathlon_datalist(datalist_json, True, "training")[:40]
        datalist = load_decathlon_datalist(datalist_json, True, "training")
        train_transforms, val_transforms = get_transforms(cfg, is_train=True)
        train_ds = data.CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=cfg.DATA.CACHE_NUM,
            runtime_cache=True,
            cache_rate=cfg.DATA.CACHE_RATE,
            num_workers=cfg.DATA.NUM_WORKERS,
        )
        train_sampler = Sampler(train_ds) if not cfg.local_test else None

        train_loader = data.DataLoader(
            train_ds,
            batch_size=cfg.DATA.TRAIN_BATCHSIZE,
            shuffle=False if train_sampler is not None else True,
            sampler=train_sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation")

        val_ds = data.Dataset(data=val_files, transform=val_transforms)
        val_sampler = Sampler(val_ds, shuffle=False) if not cfg.local_test else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=cfg.DATA.VAL_BATCHSIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            sampler=val_sampler,
            pin_memory=True,
        )
        loader = [train_loader, val_loader]
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "test")
        test_transforms = get_transforms(cfg, is_train=False)
        test_ds = data.Dataset(data=datalist, transform=test_transforms)
        test_loader = data.DataLoader(
            test_ds,
            batch_size=cfg.DATA.VAL_BATCHSIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    return loader


def build_dataset(cfg, is_train):
    name = cfg.DATA.NAME
    if name == "VESSEL":
        if is_train:
            dataset = PNG_DATA_LOADER_VESSEL(cfg, True)
        else:
            dataset = PNG_DATA_LOADER_VESSEL(cfg, False)
    else:
        raise NotImplementedError
    return dataset


def build_dataloader(cfg, dataset, sampler=None):
    if cfg.DATA.SAMPLERS == "distribute" and not cfg.local_test:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.DATA.TRAIN_BATCHSIZE,
            sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
            # persistent_workers=cfg.DATA.PERSISTENT_WORKERS
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.DATA.TRAIN_BATCHSIZE,
            shuffle=True,
            sampler=None,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=False,
            drop_last=False,
        )
    return train_dataloader, sampler
