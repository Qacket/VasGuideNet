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

import numpy as np
import scipy.ndimage as ndimage
import torch
from loguru import logger
import torch.distributed as dist
from collections.abc import Callable, Hashable, Mapping, Sequence
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms.transform import MapTransform
from monai.data.meta_tensor import MetaTensor
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree


def unSpatialPad(pred, input):
    """If the input is padding along the z-axis, remove the corresponding part of the output

    Args:
        pred (array): 3D array (x, y, z)
        input (array): 3D array (x, y, z)
    """
    start = 0
    end = input.shape[2]
    while np.max(input[:, :, start]) == 0 and np.min(input[:, :, start]) == 0:
        start += 1
    while np.max(input[:, :, end - 1]) == 0 and np.min(input[:, :, end - 1]) == 0:
        end -= 1
    return pred[:, :, start:end]


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


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (
        float(tx) / float(imx),
        float(ty) / float(imy),
        float(tz) / float(imz),
    )
    img_resampled = ndimage.zoom(img, zoom_ratio, order=3, prefilter=True)
    return img_resampled

def resample_3d_order_0(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (
        float(tx) / float(imx),
        float(ty) / float(imy),
        float(tz) / float(imz),
    )
    # 改为最近邻插值，避免类别污染
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0)
    return img_resampled

def resample_3d_torch(img: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    import torch.nn.functional as F
    """使用 PyTorch 实现高效重采样，替代 ndimage.zoom"""
    if img.ndim == 3:
        img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    img = img.float()
    resampled = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=False)
    return resampled[0, 0]

def cal_hd95(pred, target):
    """
    计算单个类别的 HD95（95%分位数双向Hausdorff距离）

    Args:
        pred (numpy.ndarray|torch.Tensor):  预测的二值化掩膜（0或1）
        target (numpy.ndarray|torch.Tensor): 真实的二值化掩膜（0或1）

    Returns:
        float: HD95值（空掩膜时返回0或图像最大距离）
    """
    # 统一输入为NumPy数组
    if not isinstance(pred, np.ndarray):
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.array(pred)
    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy() if hasattr(target, "cpu") else np.array(target)

    # 提取非零坐标点（体素位置）
    pred_pts = np.argwhere(pred > 0)
    target_pts = np.argwhere(target > 0)

    # 处理空掩膜逻辑
    if len(pred_pts) == 0 and len(target_pts) == 0:
        return 0.0  # 两者均为空，HD95=0（完美匹配）
    elif len(pred_pts) == 0 or len(target_pts) == 0:
        # 计算图像最大可能距离（对角线长度）作为替代值
        shape = pred.shape if len(pred_pts) == 0 else target.shape
        return np.sqrt(sum(dim**2 for dim in shape))

    # 使用KD-Tree加速距离计算
    try:
        tree_pred = KDTree(pred_pts)
        tree_target = KDTree(target_pts)

        # 计算双向Hausdorff距离
        d1 = tree_pred.query(target_pts, k=1)[0].max()
        d2 = tree_target.query(pred_pts, k=1)[0].max()
    except ValueError:
        # 处理无效输入（如重复点导致无法计算凸包）
        return np.sqrt(sum(dim**2 for dim in pred.shape))

    # 取95%分位数
    hd95 = np.percentile([d1, d2], 95)
    return hd95


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


def cal_brats_dsc(x, y):
    """_summary_

    Args:
        x (numpy.ndarry|torch.Tensor): shape (h, w, d)
        y (numpy.ndarry|torch.Tensor): shape (3, h, w, d)
    """
    dscs = []
    x_TC = (x == 1) | (x == 4)
    x_WT = (x == 1) | (x == 4) | (x == 2)
    x_ET = x == 4
    dscs.append(cal_dice(x_TC, y[0]))
    dscs.append(cal_dice(x_WT, y[1]))
    dscs.append(cal_dice(x_ET, y[2]))
    return dscs


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
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


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


def reduce_by_weight(value, weight):
    if not dist.is_initialized():
        return value
    mult = torch.Tensor([value * weight]).cuda()
    weight = torch.Tensor([weight]).cuda()
    dist.all_reduce(mult)
    dist.all_reduce(weight)
    avg = mult / weight
    return avg.item()

def load_model(model, resume, strict=False):
    checkpoint = torch.load(resume, map_location=lambda storge, loc: storge)
    if "epoch" in checkpoint:
        logger.info(
            "loaded model from {}, epoch {}".format(resume, checkpoint["epoch"])
        )
        state_dict_ = checkpoint["state_dict"]
    else:
        state_dict_ = checkpoint
    state_dict = {}

    for k_ in state_dict_:
        k = k_
        if k_.startswith("module"):
            k = k.replace("module", "swinViT")
        if k.count(".fc"):
            k = k.replace(".fc", ".linear")
        state_dict[k] = state_dict_[k_]
    model_state_dict = model.state_dict()

    unloaded_num = 0
    if not strict:
        del_list = []
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    info_if_main(
                        "Skip loading parameter {}, required shape{}, "
                        "loaded shape{}.".format(
                            k, model_state_dict[k].shape, state_dict[k].shape
                        )
                    )
                    state_dict[k] = model_state_dict[k]
                    unloaded_num += 1
            else:
                del_list.append(k)
                info_if_main("Drop parameter {}.".format(k))
        for k in del_list:
            del state_dict[k]
        for k in model_state_dict:
            if not (k in state_dict):
                info_if_main("No param {}.".format(k))
                state_dict[k] = model_state_dict[k]
                unloaded_num += 1
    loaded_num = len(model_state_dict) - unloaded_num
    info_if_main(f"model param {loaded_num} loaded, {unloaded_num} unloaded")
    model.load_state_dict(state_dict, strict=True)
    return model


def layer_infer(input, model=None, output_channel=4):
    """2D模型分层预测整例数据

    Args:
        model (torch.nn.Module):
        input (tensor): (b, s, d, h, w)

    Returns:
        output (tensor): (b, c, d, h, w)
    """
    if len(input.shape) == 5:
        input = input[0]
    s, d, h, w = input.shape
    output = torch.zeros((1, output_channel, d, h, w)).cuda()  # b, c, h, w, d
    with torch.no_grad():
        for z in range(d):
            if z == 0:
                slice = torch.stack([input[:, 0], input[:, 0], input[:, 1]], dim=1)
            elif z == d - 1:
                slice = torch.stack(
                    [input[:, d - 2], input[:, d - 1], input[:, d - 1]], dim=1
                )
            else:
                slice = input[:, z - 1 : z + 2]
            slice = slice.unsqueeze(0).cuda()
            logits = model(slice)  # b, c, d, h, w
            output[:, :, z] = logits[:, :, 0]
    return output