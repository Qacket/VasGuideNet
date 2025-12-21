import os
import shutil
import sys

sys.path.append(os.path.dirname(sys.path[0]))

from scipy import ndimage
import torch
import torch.nn as nn
import glob
import re
import numpy as np
import torch.distributed as dist
from monai.transforms.transform import MapTransform
from monai.config import IndexSelection, KeysCollection, SequenceStr
from loguru import logger
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.morphology import distance_transform_edt as edt
import nibabel as nib


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def logger_info(*args):
    msg = ""
    for arg in args:
        msg += f"{arg} "
    logger.info(msg)


def info_if_main(*args):
    if is_main_process():
        logger_info(*args)


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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


import torch


def distributed_all_gather(
    tensor_list,
    valid_batch_size=None,
    out_numpy=False,
    world_size=None,
    no_barrier=False,
    is_valid=None,
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()

    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(
            bool(is_valid), dtype=torch.bool, device=tensor_list[0].device
        )

    if not no_barrier:
        torch.distributed.barrier()

    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [
                torch.zeros_like(is_valid, dtype=torch.bool) for _ in range(world_size)
            ]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid_list = [x.item() for x in is_valid_list]

        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)

            # 🚨 检查是否有 NaN 或 负值
            for i, g in enumerate(gather_list):
                if torch.isnan(g).any():
                    print(
                        f"Rank {torch.distributed.get_rank()} Warning: NaN detected in gather_list[{i}]"
                    )
                if (g < 0).any():
                    print(
                        f"Rank {torch.distributed.get_rank()} Warning: Negative value detected in gather_list[{i}]"
                    )

            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]

            # 🚨 避免 `out_numpy=True` 时 gather_list 为空
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list if t.numel() > 0]

            tensor_list_out.append(gather_list)

    return tensor_list_out


def unSpatialPad_v2(pred, input):
    """If the input is padding along the z-axis, remove the corresponding part of the output.
       (Must use LogPadded in data transfrom after SpatialPadd)

    Args:
        pred (array): 3D array (x, y, z)
        input (MetaTensor): origin input (from dataloader)
    """
    padded = input.meta["padded"]
    front, back = padded[3]
    front = int(front)
    back = pred.shape[2] - int(back)
    return pred[:, :, front:back]


def inverse_resample_with_sitk(output, original_spacing, target_spacing, output_size):
    """
    使用 SimpleITK 的 ResampleImageFilter 逆向重采样推理结果到原始 spacing。

    Parameters:
        output (ndarray): 推理结果。
        original_spacing (tuple): 原始 spacing。
        target_spacing (tuple): 预处理时使用的目标 spacing。

    Returns:
        ndarray: 恢复到原始 spacing 的推理结果。
    """

    original_spacing, target_spacing = [abs(sp) for sp in original_spacing.tolist()], [
        abs(sp) for sp in target_spacing
    ]
    # 将 NumPy 数组转换为 SimpleITK 图像
    sitk_output = sitk.GetImageFromArray(output)
    sitk_output.SetSpacing(target_spacing)

    # 设置目标 spacing 为原始 spacing
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(original_spacing)

    # 计算新的尺寸
    if output_size == None:
        # 计算缩放因子
        scale_factors = [
            target / orig for orig, target in zip(original_spacing, target_spacing)
        ][::-1]

        output_size = [
            int(sz * scale) for sz, scale in zip(sitk_output.GetSize(), scale_factors)
        ]

    resample.SetSize(output_size)

    resample.SetInterpolator(
        sitk.sitkLinear
    )  # 或者使用 sitkNearestNeighbor 根据需要调整插值
    output_resampled = resample.Execute(sitk_output)

    # 转换回 NumPy 数组
    return sitk.GetArrayFromImage(output_resampled)


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (
        float(tx) / float(imx),
        float(ty) / float(imy),
        float(tz) / float(imz),
    )
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def intersectionAndUnion(output, target, K):
    """
    计算每个类别的 IoU 和 Dice
    output: (H, W, D)，预测类别索引
    target: (H, W, D)，真实类别索引
    K: 类别数
    ignore_index: 忽略的索引（默认 255）
    """
    assert (
        output.shape == target.shape
    ), f"Shape mismatch: {output.shape} vs {target.shape}"

    output = output.flatten()  # 变为 1D
    target = target.flatten()

    # 计算交集（intersection）和区域面积
    intersection = output[output == target]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target, area_output


def cal_dice(x, y):
    """
    Args:
        x (numpy.ndarry|torch.Tensor): predict result
        y (numpy.ndarry|torch.Tensor): label

    Returns:
        float: dice score
    """
    if isinstance(x, np.ndarray):
        intersect = np.count_nonzero(x * y)
        x_sum = np.count_nonzero(x)
        y_sum = np.count_nonzero(y)
    else:
        intersect = torch.count_nonzero(x * y).item()
        x_sum = torch.count_nonzero(x).item()
        y_sum = torch.count_nonzero(y).item()
    if x_sum == y_sum == 0:
        return 1.0
    return 2 * intersect / (x_sum + y_sum)


def reduce_by_weight(value, weight):
    mult = torch.Tensor([value * weight]).cuda()
    weight = torch.Tensor([weight]).cuda()
    dist.all_reduce(mult)
    dist.all_reduce(weight)
    avg = mult / weight
    return avg.item()


def resume_model(resume_dict, cfg, resume_step):
    for key in resume_dict.keys():
        load_model(
            resume_dict[key],
            os.path.join(cfg.SAVE_DIR.MODEL, "{}_{}.pth".format(key, resume_step)),
            cfg,
            pretrain=False,
        )
        print(
            "RESUME {} from {}".format(
                key,
                os.path.join(cfg.SAVE_DIR.MODEL, "{}_{}.pth".format(key, resume_step)),
            )
        )


def load_model(model_def, model_filename, config, strict=True, pretrain=True):
    print(f"======> load weight for finetuning.....")

    checkpoint = torch.load(model_filename, map_location="cuda", weights_only=True)

    if pretrain:
        if config.MODEL.NAME == "swin_transform":
            strict = False
            state_dict = checkpoint["model"]

            relative_position_index_keys = [
                k for k in state_dict.keys() if "relative_position_index" in k
            ]
            for k in relative_position_index_keys:
                del state_dict[k]

            relative_position_index_keys = [
                k for k in state_dict.keys() if "relative_corrds_table" in k
            ]
            for k in relative_position_index_keys:
                del state_dict[k]

            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "relative_position_bias_table" in k
            ]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model_def.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    print(f"Error in loading {k}, passing...")
                else:
                    if L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        relative_position_bias_table_pretrained_resized = (
                            torch.nn.functional.interpolate(
                                relative_position_bias_table_pretrained.permute(
                                    1, 0
                                ).view(1, nH1, S1, S1),
                                size=(S2, S2),
                                mode="bicubic",
                            )
                        )
                        state_dict[k] = (
                            relative_position_bias_table_pretrained_resized.view(
                                nH2, L2
                            ).permute(1, 0)
                        )

            absolute_pos_embed_keys = [
                k for k in state_dict.keys() if "absolute_pos_embed" in k
            ]
            for k in absolute_pos_embed_keys:
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model_def.state_dict()[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C2:
                    print(f"Error in loading {k}, passing...")
                else:
                    if L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        absolute_pos_embed_pretrained = (
                            absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        )
                        absolute_pos_embed_pretrained = (
                            absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                        )

                        absolute_pos_embed_pretrained_resized = (
                            torch.nn.functional.interpolate(
                                absolute_pos_embed_pretrained,
                                size=(S2, S2),
                                mode="bicubic",
                            )
                        )
                        absolute_pos_embed_pretrained_resized = (
                            absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                        )
                        absolute_pos_embed_pretrained_resized = (
                            absolute_pos_embed_pretrained_resized.flatten(1, 2)
                        )
                        state_dict[k] = absolute_pos_embed_pretrained_resized

            del state_dict["head.weight"]
            del state_dict["head.bias"]
        elif config.MODEL.NAME == "UNet3D" or config.MODEL.NAME == "ResidualUNet3D":
            strict = False
            state_dict = checkpoint["model_state_dict"]
            del state_dict["final_conv.weight"]
            del state_dict["final_conv.bias"]
            if hasattr(model_def, "final_conv"):
                nn.init.kaiming_normal_(
                    model_def.final_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(model_def.final_conv.bias, 0.0)
            else:
                raise KeyError(
                    "Model does not have 'final_conv' layer. Check model definition."
                )
        elif config.MODEL.NAME == "unetr":
            strict = False
            state_dict = checkpoint
            del state_dict["out.conv.conv.weight"]
            del state_dict["out.conv.conv.bias"]
            nn.init.kaiming_normal_(
                model_def.out.conv.conv.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(model_def.out.conv.conv.bias)
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if isinstance(model_def, nn.DataParallel) or isinstance(
        model_def, nn.parallel.DistributedDataParallel
    ):
        if not strict:
            msg = model_def.module.load_state_dict(state_dict, strict)
        else:
            msg = model_def.module.load_state_dict(state_dict)
    else:
        if not strict:
            msg = model_def.load_state_dict(state_dict, strict)
        else:
            model_def_dicts = model_def.state_dict()
            new_dicts = dict()
            uncover = []
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if k not in model_def_dicts.keys():
                    uncover.append(k)
                    continue
                new_dicts[k] = v
            print("uncover keys: ", uncover)
            msg = model_def.load_state_dict(new_dicts)
    # print(msg)
    # print(f"======> load weight done! '{model_filename}'")
    print(f"======> load weight done! ")

    del checkpoint
    torch.cuda.empty_cache()


def check(cfg, batch_data):
    img_name = batch_data["image_meta_dict"]["filename_or_obj"][0]
    label_name = batch_data["label_meta_dict"]["filename_or_obj"][0]
    if "PuJian" in img_name:
        path1 = os.path.join(
            "/data1/zzh/VesselSegModel/old/private", img_name.split("/")[-1]
        )
    elif "hepaticvessel" in img_name:
        path1 = os.path.join(
            "/data1/zzh/VesselSegModel/old/hepaticvessel", img_name.split("/")[-1]
        )
    else:
        path1 = os.path.join(
            "/data1/zzh/VesselSegModel/old/IRCADb", img_name.split("/")[-1]
        )
    shutil.copy(
        path1,
        os.path.join(
            cfg.OUTPUT.SAVEFILEDIR,
            "predict_" + img_name.split("/")[-1],
        ),
    )
    shutil.copy(
        img_name,
        os.path.join(cfg.OUTPUT.SAVEFILEDIR, img_name.split("/")[-1]),
    )
    shutil.copy(
        label_name,
        os.path.join(
            cfg.OUTPUT.SAVEFILEDIR,
            "label_" + label_name.split("/")[-1],
        ),
    )
    return False


def save_dict_of_models(input_dict, suffix, folder, config, **kwargs):
    def operation(k, v, model_path):
        save_model(v, k, model_path, config)

    operate_on_dict_of_models(input_dict, suffix, folder, operation, "SAVE", **kwargs)


def load_dict_of_models(input_dict, suffix, folder, config, **kwargs):
    def operation(k, v, model_path):
        load_model(v, model_path, config, pretrain=False)

    operate_on_dict_of_models(input_dict, suffix, folder, operation, "LOAD", **kwargs)


def operate_on_dict_of_models(input_dict, suffix, folder, operation, logging_string=""):
    for k, v in input_dict.items():
        model_path = modelpath_creator(folder, k, suffix)
        try:
            operation(k, v, model_path)
            print("%s %s" % (logging_string, model_path))
        except IOError:
            print("Could not %s %s" % (logging_string, model_path))
            raise IOError


def modelpath_creator(folder, basename, identifier, extension=".pth"):
    if identifier is None:
        return os.path.join(folder, basename + extension)
    else:
        return os.path.join(folder, "%s_%s%s" % (basename, str(identifier), extension))


def save_model(model, model_name, file_path, config):
    if any(
        [
            isinstance(model, x)
            for x in [torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel]
        ]
    ):
        torch.save(model.module.state_dict(), file_path)
    else:
        torch.save(model.state_dict(), file_path)


def save_nifti(array, spacing, save_path, is_affine=False):
    """
    保存 NumPy 数组为 NIFTI 文件，并设置 spacing。

    参数:
    array (numpy.ndarray): 要保存的 3D NumPy 数组。
    spacing (tuple): 体素的空间分辨率（spacing），如 (x, y, z)。
    save_path (str): 保存 NIFTI 文件的路径。
    """
    # 检查输入数组的维度
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    if len(array.shape) == 4:
        array = array[0, :, :, :]
    elif len(array.shape) == 5:
        array = array[0, 0, :, :, :]
    if array.ndim != 3:
        raise ValueError("输入数组必须是一个三维 NumPy 数组。")

    if array.dtype == np.int64:
        array = array.astype(np.uint8)
    # 创建 NIFTI 图像

    # 设置空间分辨率
    if is_affine:
        nifti_img = nib.Nifti1Image(array, affine=spacing[0])
    else:
        nifti_img = nib.Nifti1Image(array, affine=np.eye(4))
        nifti_img.header["pixdim"][1:4] = spacing  # pixdim[0] 是时间维度

    # 保存 NIFTI 文件
    nib.save(nifti_img, save_path)
    print(f"NIFTI 文件已保存到: {save_path}")
