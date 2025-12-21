# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

from lib.solvers.lr_scheduler_monai import (
    LinearWarmupCosineAnnealingLR,
    WarmupExponentialLR,
)
from engine.trainer_lits import run_training
from lib.datasets import get_loader, get_loader_lmdb, get_loader_lmdb_monai, get_loader_couinaud_lmdb_monai

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from lib.models.swin_unetr_resswin_skipadd import SwinUNETR
from lib.models.UNet3D import UNet3D
from lib.models.MISSFormer3D import MISSFormer3D_origin
from lib.models.MISSFormer3D_disEmbed import MISSFormer3D_disEmbed
from lib.models.MISSFormer3D_tokenDecoder import MISSFormer3D_tokenDecoder
from lib.models.LiverFormer import LiverFormer
from lib.models.generic_UNet import Generic_UNet
from lib.models.VesselEnhancedNet import VesselEnhancedNet
from lib.models.SegMamba import SegMamba
from lib.models.segformer import SegFormer3D
from lib.solvers.MultiModalContrastLoss import MultiModalContrastLoss
from utils.utils_new import info_if_main, load_model, logger_info
from loguru import logger
from lib.solvers.boundary_loss import DiceBDLoss
import torch.nn as nn

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--checkpoint", default=None, help="start training from saved checkpoint"
)
parser.add_argument(
    "--logdir",
    default="/data7/zzh/new_couinaud",
    type=str,
    help="directory to save the tensorboard logs",
)
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument(
    "--data_dir",
    default="/home/zhaozihao/Vessel_Segmentation/lib/datasets/data_json",
    type=str,
    help="dataset directory",
)
parser.add_argument(
    "--json_list", default="couinaud_dataset.json", type=str, help="dataset json file"
)
parser.add_argument(
    "--model_name",
    default="HCFormer",
    type=str,
    help="model name",
)
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument(
    "--save_checkpoint",
    action="store_true",
    default=True,
    help="save checkpoint during training",
)
parser.add_argument(
    "--max_epochs", default=1000, type=int, help="max number of training epochs"
)
parser.add_argument("--batch_size", default=12, type=int, help="number of batch size")
parser.add_argument(
    "--sw_batch_size",
    default=4,
    type=int,
    help="number of sliding window batch size(window size)",
)
parser.add_argument(
    "--optim_lr", default=1e-4, type=float, help="optimization learning rate"
)
parser.add_argument(
    "--optim_name", default="adamw", type=str, help="optimization algorithm"
)
parser.add_argument(
    "--reg_weight", default=1e-5, type=float, help="regularization weight"
)
parser.add_argument("--save_intervals", default=50, type=int, help="")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=True,
    help="start distributed training",
)
parser.add_argument(
    "--world_size", default=6, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url"
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--norm_name", default="instance", type=str, help="normalization name"
)
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument(
    "--in_channels", default=1, type=int, help="number of input channels"
)
parser.add_argument(
    "--out_channels", default=9, type=int, help="number of output channels"
)
parser.add_argument(
    "--use_normal_dataset", action="store_true", help="use monai Dataset class"
)
parser.add_argument(
    "--a_min", default=-10.0, type=float, help="a_min in ScaleIntensityRanged"
)
parser.add_argument(
    "--a_max", default=225.0, type=float, help="a_max in ScaleIntensityRanged"
)
parser.add_argument(
    "--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged"
)
parser.add_argument(
    "--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged"
)
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=32, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument(
    "--dropout_path_rate", default=0.0, type=float, help="drop path rate"
)
parser.add_argument(
    "--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability"
)
parser.add_argument(
    "--RandRotate90d_prob",
    default=0.2,
    type=float,
    help="RandRotate90d aug probability",
)
parser.add_argument(
    "--RandScaleIntensityd_prob",
    default=0.1,
    type=float,
    help="RandScaleIntensityd aug probability",
)
parser.add_argument(
    "--RandShiftIntensityd_prob",
    default=0.1,
    type=float,
    help="RandShiftIntensityd aug probability",
)
parser.add_argument(
    "--infer_overlap", default=0.7, type=float, help="sliding window inference overlap"
)
parser.add_argument(
    "--lrschedule",
    default="warmup_cosine",
    type=str,
    help="type of learning rate scheduler",
)
parser.add_argument(
    "--warmup_epochs", default=50, type=int, help="number of warmup epochs"
)
parser.add_argument(
    "--resume_ckpt",
    action="store_true",
    help="resume training from pretrained checkpoint",
)
parser.add_argument(
    "--smooth_dr",
    default=1e-6,
    type=float,
    help="constant added to dice denominator to avoid nan",
)
parser.add_argument(
    "--smooth_nr",
    default=0.0,
    type=float,
    help="constant added to dice numerator to avoid zero",
)
parser.add_argument(
    "--use_checkpoint",
    action="store_true",
    help="use gradient checkpointing to save memory",
)
parser.add_argument(
    "--use_ssl_pretrained",
    action="store_true",
    help="use self-supervised pretrained weights",
)
parser.add_argument(
    "--spatial_dims", default=3, type=int, help="spatial dimension of input data"
)
parser.add_argument("--loss", default="DiceCE", help="loss function to use")
parser.add_argument(
    "--debug",
    action="store_true",
    help="use gradient checkpointing to save memory",
)
parser.add_argument(
    "--with_contrast_loss",
    action="store_true",
    help="use contrastive loss for training",
)
parser.add_argument(
    "--is_vessel",
    action="store_true",
)
parser.add_argument(
    "--memobank_path",
    default="memobank.pth",
    type=str,
)
parser.add_argument(
    "--loss_warmup_epochs",
    default=5,
    type=int,
)
parser.add_argument(
    "--num_class",
    default=-1,
    type=int,
    help="exclude background; if num_class=-1, calcuate binary dice",
)
parser.add_argument(
    "--train_dir",
    default="",
    type=str,
)

parser.add_argument(
    "--val_dir",
    default="",
    type=str,
)
parser.add_argument(
    "--save_output",
    action="store_true",
    default=True,
    help="whether to save output as nifty file",
)

parser.add_argument(
    "--replace_strategy",
    default="cats",
    type=str,
)
parser.add_argument("--output_name", default="outputs", type=str, help="output name")

import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai.utils import set_determinism
from functools import partial
import warnings
import logging
import re


# 假设已有这些自定义导入
# from your_module import logger, info_if_main, logger_info, DiceCELoss, SwinUNETR, get_loader, run_training


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    args.amp = not args.noamp

    # 日志目录创建（仅主进程操作）
    if not args.distributed or args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)

    logger.add(os.path.join(args.logdir, "train_log.txt"))
    info_if_main("args:", args)

    if args.distributed:
        # 自动获取物理GPU数量
        args.ngpus_per_node = torch.cuda.device_count()
        info_if_main(f"Found total gpus {args.ngpus_per_node}")

        # 计算全局进程数（单机情况下 world_size = ngpus_per_node）
        args.world_size = args.ngpus_per_node

        # 启动多进程训练
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,), join=True)
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    # 初始化日志（每个进程独立记录）
    logger.add(os.path.join(args.logdir, f"train_log_rank{gpu}.txt"))

    # 设置进程绑定
    args.local_rank = gpu
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    # 初始化分布式训练
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )
        info_if_main(f"Initialized distributed training on rank {args.rank}")

    # 确定性配置
    torch.backends.cudnn.benchmark = True
    set_determinism(seed=42)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    # 数据加载
    args.test_mode = False
    # loader = get_loader(args)
    # loader = get_loader_lmdb(args)
    # loader = get_loader_lmdb_monai(args)
    loader = get_loader_couinaud_lmdb_monai(args)

    # 模型初始化

    if args.model_name == "UNet3d":
        model = UNet3D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            final_sigmoid=False,  # 不在模型里做 Sigmoid/Softmax
            is_segmentation=False,  # 关闭内部激活，直接返回 logits
        ).to(device)
    elif args.model_name == "HCFormer":
        model = MISSFormer3D_origin(
            input_dims=args.in_channels,
            num_classes=args.out_channels,
        ).to(device)
    elif args.model_name == "HCFormer_2channels":
        model = MISSFormer3D_origin(
            input_dims=args.in_channels * 2,
            num_classes=args.out_channels,
        ).to(device)
    elif args.model_name == "HCFormer_disEmbed":
        model = MISSFormer3D_disEmbed(
            input_dims=args.in_channels,
            num_classes=args.out_channels,
        ).to(device)
    elif args.model_name == "HCFormer_tokenDecoder":
        model = MISSFormer3D_tokenDecoder(
            input_dims=args.in_channels,
            num_classes=args.out_channels,
        ).to(device)
    elif args.model_name == "Unetr":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
            use_checkpoint=args.use_checkpoint,
        ).to(device)

    elif args.model_name == "LiverFormer":
        model = LiverFormer(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            base_num_features=args.feature_size,  # reuse CLI feature_size
            num_pool=4,
        ).to(device)
    elif args.model_name == "GUNet":
        model = Generic_UNet(
            input_channels=args.in_channels,
            base_num_features=args.feature_size,
            num_classes=args.out_channels,
            num_pool=4,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            dropout_op=nn.Dropout3d,
            nonlin=nn.LeakyReLU,
            deep_supervision=False,
            max_num_features=320,
            convolutional_pooling=True,
            convolutional_upsampling=True
        ).to(device)
    elif args.model_name == "SegMamba":
        model = SegMamba(
            in_chans=args.in_channels,
            out_chans=args.out_channels,
        ).to(device)
    elif args.model_name == "VesselEnhancedNet":
        model = VesselEnhancedNet(
            in_channels=args.in_channels,
            num_classes=args.out_channels,
        ).to(device)
    elif args.model_name == "SegFormer":
        model = SegFormer3D(
            model_name='B0_3D',  # 可改为 B1~B5
            num_classes=args.out_channels,
            in_channels=args.in_channels,
        ).to(device)
    else:
        raise ValueError("Unsupported model name")
    # 加载预训练权重
    if args.resume_ckpt:
        model_dict = torch.load(
            os.path.join(args.pretrained_dir, args.pretrained_model_name),
            map_location=device,
        )["state_dict"]
        model.load_state_dict(model_dict, strict=False)
        info_if_main(f"Loaded pretrained weights from {args.pretrained_model_name}")

    # 分布式数据并行
    if args.distributed:
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 允许某些参数在部分 forward 中不被使用
        if args.model_name in ["GUNet", "LiverFormer"]:
            find_unused_parameters = True
        else:
            find_unused_parameters = False

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

        info_if_main(f"Initialized DDP on rank {args.rank}")

    # 损失函数
    if args.loss == "DiceCE":
        dice_loss = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr,
        )
    else:
        raise ValueError("Unsupported loss function")

    loss_2 = None
    if args.with_contrast_loss:
        loss_2 = MultiModalContrastLoss(
            num_classes=args.out_channels,
            vessel_idx=0,
            # warmup_epochs=args.loss_warmup_epochs,
            warmup_epochs=10,
            replace_strategy = args.replace_strategy,
        )

    # 推理函数
    model_inferer = partial(
        sliding_window_inference,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
        mode="gaussian",
    )

    # 优化器
    if args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported optimizer")

    # 学习率调度
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    else:
        scheduler = None

    # 混合精度
    scaler = torch.amp.GradScaler()

    # 恢复检查点
    best_acc = 0
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if loss_2 is not None and os.path.exists(
                os.path.join(args.logdir, args.memobank_path)
        ):
            loss_2.load(os.path.join(args.logdir, args.memobank_path), device=device)
        load_checkpoint(
            model, checkpoint["state_dict"]
        )  # model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0)
        info_if_main(f"Resumed from checkpoint at epoch {start_epoch}")

    # 训练执行
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        val_acc_max=best_acc,
        scaler=scaler,
        loss_func2=loss_2,
    )
    return accuracy


def load_checkpoint(model, state_dict):
    # 自动处理有无 'module.' 前缀的情况
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            new_key = key  # 已有前缀则保留
        else:
            new_key = f"module.{key}"  # 无前缀则添加
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    main()
