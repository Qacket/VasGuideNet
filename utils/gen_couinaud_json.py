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

# basic_info = {
#     "description": "Liver Vessel Segmentation",
#     "labels": {"0": "background", "1": "arterial", "2": "venous"},
#     "licence": "yt",
#     "modality": {"0": "CT"},
#     "name": "Vessel_all",
#     "numTest": 0,
#     "numTraining": 0,
#     "reference": "Zhejiang University",
#     "release": "11/12/2024",
#     "tensorImageSize": "3D",
#     "test": [],
#     "training": [],  # need
#     "validation": [],  # need
# }


basic_info = {
    "description": "Liver Couinaud Segmentation",
    "labels": {
        "0": "background",
        "1": "尾状叶",
        "2": "左外叶上段",
        "3": "左外叶下段",
        "4": "左内叶",
        "5": "右前叶下段",
        "6": "右后叶下段",
        "7": "右后叶上段",
        "8": "右前叶上段",
    },
    "licence": "yt",
    "modality": {"0": "CT"},
    "name": "Couinaud_all",
    "numTest": 0,
    "numVal": 0,
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
    if "imagesTr" not in images_dir_abs:
        labels_dir_abs = data_dir.replace("imagesTs", "labelsTr")
    else:
        labels_dir_abs = data_dir.replace("imagesTr", "labelsTr")
    count, count1 = 0, 0
    image_names = os.listdir(images_dir_abs)
    for image_name in tqdm(image_names):
        if not image_name.endswith(".nii.gz"):
            continue
        if image_name == "PuJian_20241109_027_0002.nii.gz":
            continue
        if "IRCADb" in image_name:
            continue
        label_name = image_name
        image_path = os.path.join(images_dir_abs, image_name)
        label_path = os.path.join(labels_dir_abs, label_name)
        label_vessel_path = label_path.replace("Couinaud", "Vessel")
        if not os.path.exists(label_path):
            continue
        if not os.path.exists(label_vessel_path):
            print(label_vessel_path, "not exists")
            continue
        results.append(
            {
                "image": image_path,
                "label": label_path,
                "label_vessel": label_vessel_path,
            }
        )
    print(len(results))
    return results


def main(args):
    data = basic_info.copy()
    data_list = []
    for i in args.train_dir:
        data_list.extend(get_list(i))
    random.shuffle(data_list)
    data_list = data_list
    val_index = random.sample(range(len(data_list)), int(len(data_list) * 0.2))
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
    data["numTraining"] = len(data["training"])
    data["numTest"] = len(data["test"])
    data["numVal"] = len(data["validation"])
    print(
        f"train: {len(data['training'])}, val: {len(data['validation'])}, test: {len(data['test'])}"
    )
    with open(args.output_path, "w") as f:
        json.dump(data, f)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform npz datasets to nii format")
    parser.add_argument(
        "--output_path",
        default="/home/zhaozihao/Vessel_Segmentation/lib/datasets/data_json/couinaud_dataset_public_fold3.json",
    )
    args = parser.parse_args()
    args.train_dir = [
        # "/data1/zzh/dataset_1210/Couinaud/abdomen_3d_reconstruction/imagesTr",
        # "/data1/zzh/dataset_1210/Couinaud/PuJian_20241109/imagesTr",
        "/data1/zzh/dataset_1210/Couinaud/Task08_HepaticVessel/imagesTr",
        "/data1/zzh/dataset_1210/Couinaud/Task08_HepaticVessel/imagesTs",
    ]
    main(args)
