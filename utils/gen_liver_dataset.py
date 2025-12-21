"""
对已经转换完成的数据集，生成decathlon challenge格式的json描述文件
"""

import argparse
import os
import json
import random
import sys
from tqdm import tqdm
from pathlib import Path

basic_info = {
    "description": "Liver Segmentation",
    "labels": {"0": "background", "1": "liver", "2": "tumor"},
    "licence": "yt",
    "modality": {"0": "CT"},
    "name": "liver_all",
    "numTest": 0,
    "numTraining": 0,
    "reference": "Zhejiang University",
    "release": "11/12/2024",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],  # need
    "validation": [],  # need
}


def get_list(data_dir):
    results = []

    images_dir_abs = data_dir
    labels_dir_abs = data_dir.replace("imagesTr", "labelsTr")

    image_names = os.listdir(images_dir_abs)
    for image_name in tqdm(image_names):
        label_name = image_name
        if "volume" in label_name:
            label_name = label_name.replace("volume", "segmentation")
        label_path_abs = os.path.join(labels_dir_abs, label_name)

        if not os.path.exists(label_path_abs):
            raise Exception(f"{label_path_abs} not exist")

        image_path = os.path.join(images_dir_abs, image_name)
        label_path = os.path.join(labels_dir_abs, label_name)
        results.append({"image": image_path, "label": label_path})

    return results


def main(args):
    data = basic_info.copy()
    data_list = []
    for i in args.train_dir:
        data_list.extend(get_list(i))
    val_index = random.sample(range(len(data_list)), int(len(data_list) * 0.1))
    index_len = len(val_index)
    data["training"] = [
        data_list[i] for i in range(len(data_list)) if i not in val_index
    ]
    data["validation"] = [
        data_list[i] for i in range(len(data_list)) if i in val_index[index_len // 2 :]
    ]
    data["test"] = [
        data_list[i] for i in range(len(data_list)) if i in val_index[: index_len // 2]
    ]
    print(
        f"train: {len(data['training'])}, val: {len(data['validation'])}, test: {len(data['test'])}"
    )
    # sys.exit()
    with open(args.output_path, "w") as f:
        json.dump(data, f)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument(
        "--output_path",
        default="/home/zhaozihao/Vessel_Segmentation/train_liver_seg_2.json",
    )
    args = parser.parse_args()

    args.train_dir = [
        "/data7/zzh/liver_volume_dataset/task03/Task03_Liver/imagesTr/",
        # "/data7/zzh/liver_volume_dataset/lits/train/imagesTr/",
    ]

    main(args)
