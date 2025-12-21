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


import warnings

warnings.filterwarnings("ignore")

import argparse
import os
from utils.lw_measure import get_class, cal_iou, cal_dsc, cal_nsd, get_RVD, safe_divide
from tqdm import tqdm
from joblib import Parallel, delayed

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.datasets import get_loader, get_loader_couinaud_lmdb_monai
from utils.utils_new import (
    cal_dice,
    cal_hd95,
    resample_3d,
    unSpatialPad_v2,
    info_if_main,
    reduce_by_weight,
)

from monai.inferers import sliding_window_inference

# from algorithms.SwinUNETR.network.swin_unetr import SwinUNETR

# from lib.models.swin_unetr_resswin_skipadd import SwinUNETR
# from lib.models.UNet3D import UNet3D
from lib.models.MISSFormer3D import MISSFormer3D_origin
from lib.models.VesselEnhancedNet import VesselEnhancedNet
from lib.models.MISSFormer3D_disEmbed import MISSFormer3D_disEmbed
from lib.models.MISSFormer3D_tokenDecoder import MISSFormer3D_tokenDecoder

from loguru import logger

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--data_dir",
    default="/home/zhaozihao/Vessel_Segmentation/lib/datasets/data_json",
    type=str,
    help="dataset directory",
)
parser.add_argument(
    "--json_list",
    default="couinaud_dataset_private.json",
    type=str,
    help="dataset json file",
)
parser.add_argument(
    "--logdir",
    default="/data7/zzh/private_train/new_couinaud_HCFormer/",
    type=str,
    help="experiment name",
)
parser.add_argument(
    "--model_name",
    default="HCFormer",
    type=str,
    help="model name",
)
parser.add_argument(
    "--pretrained_model_name",
    default="model.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument(
    "--infer_overlap", default=0.7, type=float, help="sliding window inference overlap"
)
parser.add_argument(
    "--in_channels", default=1, type=int, help="number of input channels"
)
parser.add_argument(
    "--out_channels", default=9, type=int, help="number of output channels"
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
    "--distributed",
    action="store_true",
    default=False,
    help="start distributed training",
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
parser.add_argument("--workers", default=1, type=int, help="number of workers")
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
    "--spatial_dims", default=3, type=int, help="spatial dimension of input data"
)
parser.add_argument(
    "--use_checkpoint",
    action="store_true",
    help="use gradient checkpointing to save memory",
)
parser.add_argument(
    "--num_class",
    default=-1,
    type=int,
    help="exclude background; if num_class=-1, calcuate binary dice",
)
parser.add_argument(
    "--save_output",
    action="store_true",
    default=True,
    help="whether to save output as nifty file",
)
parser.add_argument(
    "--world_size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument("--output_name", default="outputs", type=str, help="output name")

parser.add_argument(
    "--test_dir",
    default="",
    type=str,
)


def main():
    args = parser.parse_args()
    logger.add(os.path.join(args.logdir, f"test_{args.output_name}.txt"))
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        info_if_main(f"Found total gpus {args.ngpus_per_node}")
        args.world_size = args.ngpus_per_node
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    logger.add(os.path.join(args.logdir, f"test_{args.output_name}.txt"))

    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    args.test_mode = True
    output_directory = os.path.join(args.logdir, args.output_name)
    os.makedirs(output_directory, exist_ok=True)
    val_loader = get_loader_couinaud_lmdb_monai(args)
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(args.logdir, model_name)

    if args.model_name == "unet3d":
        model = UNet3D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
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
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    elif args.model_name == "VesselEnhancedNet":
        model = VesselEnhancedNet(
            in_channels=args.in_channels,
            num_classes=args.out_channels,
            is_test=True,
        ).to(device)
    else:
        raise Exception("model name not found")
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    dice_sum, iou_sum, rvd_sum, data_num, dice_sum_old = (0, 0, 0, 0, 0)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            # img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            original_affine = batch["affine_matrix"][0].numpy()
            img_name = batch["name"][0]
            input_shape = val_inputs.shape[2:]
            if (
                    input_shape[0] < args.roi_x
                    or input_shape[1] < args.roi_y
                    or input_shape[2] < args.roi_z
            ):
                raise Exception("image size small than trainning roi")
            if (
                    args.model_name == "VesselEnhancedNet"
                    or args.model_name == "HCFormer_2channels"
                    or args.model_name == "HCFormer_disEmbed"
                    or args.model_name == "HCFormer_tokenDecoder"
            ):
                val_inputs = torch.cat(
                    [val_inputs, batch["label_vessel"].cuda()], dim=1
                )
            val_outputs = sliding_window_inference(
                val_inputs,
                (args.roi_x, args.roi_y, args.roi_z),
                4,
                model,
                overlap=args.infer_overlap,
                mode="gaussian",
            )
            val_outputs = torch.argmax(val_outputs, axis=1)[
                0
            ]  # b, c, h, w, d -> h, w, d

            val_outputs = unSpatialPad_v2(val_outputs, val_inputs).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :].astype(np.uint8)
            target_shape = val_labels.shape
            val_outputs = resample_3d(val_outputs, target_shape)

            intersect_list = []
            union1_list = []
            union2_list = []
            pre_area_list = []
            gt_area_list = []
            for j in range(1, args.num_class):
                pre_binary = get_class(val_outputs, j)
                gt_binary = get_class(val_labels, j)

                intesect = np.count_nonzero(pre_binary * gt_binary)
                union1 = np.count_nonzero(pre_binary) + np.count_nonzero(gt_binary)
                union2 = np.count_nonzero((pre_binary + gt_binary) > 0)
                pre_area_list.append(pre_binary.sum())
                gt_area_list.append(gt_binary.sum())
                intersect_list.append(intesect)
                union1_list.append(union1)
                union2_list.append(union2)

            dice_old = cal_dsc(val_outputs > 0, val_labels > 0)
            dice = safe_divide(2 * np.sum(intersect_list), np.sum(union1_list))
            iou = safe_divide(np.sum(intersect_list), np.sum(union2_list))
            rvd = safe_divide(
                abs(np.sum(pre_area_list) - np.sum(gt_area_list)),
                np.sum(gt_area_list),
            )

            # dice_list_sub = []
            # if args.num_class == -1:
            #     # organ_Dice = cal_dice(val_outputs > 0, val_labels > 0)
            #     organ_Dice = cal_dice(val_outputs, val_labels)
            #     dice_list_sub.append(organ_Dice)
            # else:
            #     for j in range(args.num_class + 1):
            #         organ_Dice = cal_dice(val_outputs == j, val_labels == j)
            #         dice_list_sub.append(organ_Dice)
            # mean_dice = np.mean(dice_list_sub)
            # dice_sum += mean_dice

            dice_sum_old += dice_old
            dice_sum += dice
            iou_sum += iou
            rvd_sum += rvd
            data_num += 1
            logger.info(f"Start validation: {i}/{len(val_loader)}, Image: {img_name}")
            logger.info(
                f"DSC: {dice:.4f}, IOU: {iou:.4f}, RVD: {rvd:.4f}, DSC_old: {dice_old:.4f}"
            )
            if args.save_output:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                    os.path.join(output_directory, img_name),
                )
    avg_dice = reduce_by_weight(dice_sum / data_num, data_num)
    avg_iou = reduce_by_weight(iou_sum / data_num, data_num)
    avg_rvd = reduce_by_weight(rvd_sum / data_num, data_num)
    avg_dice_old = reduce_by_weight(dice_sum_old / data_num, data_num)
    info_if_main("Overall Mean Dice_old: {}".format(np.mean(avg_dice_old)))
    info_if_main("Overall Mean Dice: {}".format(np.mean(avg_dice)))
    info_if_main("Overall Mean iou: {}".format(np.mean(avg_iou)))
    info_if_main("Overall Mean rvd: {}".format(np.mean(avg_rvd)))


if __name__ == "__main__":
    main()
