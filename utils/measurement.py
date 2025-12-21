import random
import re
import numpy as np

import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import directed_hausdorff
import sys


def calculate_hausdorff(val_outputs, val_labels):
    # 计算整个预测与标签之间的Hausdorff距离
    coords_pred = np.argwhere(val_outputs > 0)
    coords_target = np.argwhere(val_labels > 0)

    if coords_pred.size == 0 or coords_target.size == 0:
        return float("inf")

    d_AB = directed_hausdorff(coords_pred, coords_target)[0]
    d_BA = directed_hausdorff(coords_target, coords_pred)[0]
    hausdorff_distance = max(d_AB, d_BA)

    return hausdorff_distance


def calculate_dice(val_outputs, val_labels):
    # 计算整个预测与标签之间的Dice系数
    pred_class = (val_outputs > 0).astype(np.float32)
    target_class = (val_labels > 0).astype(np.float32)
    return cal_dice(pred_class, target_class)


def cal_dice(x, y):
    intersect = np.count_nonzero(x * y)
    x_sum = np.count_nonzero(x)
    y_sum = np.count_nonzero(y)

    return 2 * intersect / (x_sum + y_sum) if (x_sum + y_sum) > 0 else 0.0


def cal_sensitivity(predictions, labels):
    # 计算整体的敏感性
    TP = np.count_nonzero(predictions * labels)  # 真正例
    FN = np.count_nonzero(labels) - TP  # 假负例

    return TP / (TP + FN) if (TP + FN) > 0 else 1.0


def cal_specificity(predictions, labels):
    # 计算整体的特异性
    TN = np.count_nonzero((1 - predictions) * (1 - labels))  # 真负例
    FP = np.count_nonzero(predictions * (1 - labels))  # 假正例

    return TN / (TN + FP) if (TN + FP) > 0 else 1.0


def process_file(file_path, dir_1, dir_2, dir_3):

    val_labels = nib.load(file_path).get_fdata()
    name = file_path.name.replace("predict_", "")

    if (dir_1 / name).exists():
        val_outputs = nib.load(dir_1 / name).get_fdata()
    elif (dir_2 / name).exists():
        val_outputs = nib.load(dir_2 / name).get_fdata()
    else:
        val_outputs = nib.load(dir_3 / name).get_fdata()

    # val_labels = nib.load(
    #     "/data1/zzh/VesselSegModel/3DMISSFormer/test/predict_PuJian_20241109_011_0001.nii.gz"
    # ).get_fdata()
    # val_outputs = nib.load(
    #     "/data1/zzh/VesselSegModel/3DMISSFormer/test/label_PuJian_20241109_011_0001.nii.gz"
    # ).get_fdata()

    avg_hausdorff_distance = calculate_hausdorff(val_outputs, val_labels)
    # avg_hausdorff_distance = 0
    avg_dice_score = calculate_dice(val_outputs, val_labels)
    avg_sensitivity = cal_sensitivity(val_outputs, val_labels)
    avg_specificity = cal_specificity(val_outputs, val_labels)

    return (
        avg_hausdorff_distance,
        avg_dice_score,
        avg_sensitivity,
        avg_specificity,
        file_path.name,
    )


def process_wrapper(args):
    # 解包参数并调用 process_file
    return process_file(*args)


def test():

    # dir = Path("/data1/zzh/VesselSegModel/3DMISSFormer/test_private/")
    dir = Path("/data1/zzh/VesselSegModel/3DMISSFormer/test_IRCADb/")
    # dir = Path("/data1/zzh/VesselSegModel/3DMISSFormer/test_HepaticVessel/")
    dir_1 = Path("/data1/zzh/dataset_1210/Vessel/abdomen_3d_reconstruction/labelsTr/")
    dir_2 = Path("/data1/zzh/dataset_1210/Vessel/PuJian_20241109/labelsTr")
    dir_3 = Path("/data1/zzh/dataset_1210/Vessel/Task08_HepaticVessel/labelsTr")

    file_paths = list(dir.glob("predict_*.nii.gz"))

    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold3.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/HepaticVessel_Swin UNETR_fold1.txt"
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/3D_IRCADb_3D U-Net_fold2.txt"
    # )
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold1.txt"
    output_file = (
        "/home/zhaozihao/Vessel_Segmentation/utils/3D_IRCADb_HCFormer_fold5.txt"
    )
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/private_TransFusionNet_fold1.txt"
    # )
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/3D_IRCADb_Swin UNETR_fold4.txt"
    # )
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/3D_IRCADb_MISSFormer_fold2.txt"
    # )
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold2.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_MedNeXt_fold3.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold2.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold1.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/LOSS2_fold4.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/LOSS4_fold2.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/LOSS5_fold1.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_3D U-Net_fold3.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_MedNeXt_fold1.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/HepaticVessel_TransFusionNet_fold5.txt"
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/HepaticVessel_HCFormer_fold2.txt"
    # )
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold5.txt"

    # ablation
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/ablation_MSA_fold1.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/ablation_HCB_FFN_fold2.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/ablation_HCB_FFN_fold3.txt"
    # output_file = "/home/zhaozihao/Vessel_Segmentation/utils/ablation_MCB_FFN_fold5.txt"
    # output_file = (
    #     "/home/zhaozihao/Vessel_Segmentation/utils/ablation_MCB_MSFFN_fold1.txt"
    # )

    results = []
    with Pool(processes=1) as pool:
        with tqdm(total=len(file_paths)) as pbar:
            # 封装参数
            params = [(file_path, dir_1, dir_2, dir_3) for file_path in file_paths]
            for result in pool.imap_unordered(process_wrapper, params):
                results.append(result)
                pbar.update(1)

    dsc, hd, spe, sen = (
        0,
        0,
        0,
        0,
    )
    with open(output_file, "w") as f:
        for i in results:
            (
                avg_hausdorff_distance,
                avg_dice_score,
                avg_sensitivity,
                avg_specificity,
                file_name,
            ) = i
            dsc += avg_dice_score
            hd += avg_hausdorff_distance
            spe += avg_specificity
            sen += avg_sensitivity
            output = f"{file_name}: dice:{avg_dice_score:.16f}, hd: {avg_hausdorff_distance:.16f}, sen: {avg_sensitivity:.16f}, spe:{avg_specificity:.16f}\n"
            f.write(output)
        content = f"average dice: {dsc/len(results):.4f}\naverage hd: {hd/len(results):.4f}\naverage sen:{sen/len(results):.4f}\naverage spe:{spe/len(results):.4f}\n"
        f.write(content)


if __name__ == "__main__":
    test()
