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
    "description": "Liver Vessel Segmentation",
    "labels": {"0": "background", "1": "arterial", "2": "venous"},
    "licence": "yt",
    "modality": {"0": "CT"},
    "name": "Vessel_all",
    "numTest": 0,
    "numTraining": 0,
    "numValidation": 0,
    "reference": "Zhejiang University",
    "release": "11/12/2024",
    "tensorImageSize": "3D",
    "test": [],
    "training": [],  # need
    "validation": [],  # need
}

# basic_info = {
#     "description": "Liver Segmentation",
#     "labels": {"0": "background", "1": "liver", "2": "tumor"},
#     "licence": "yt",
#     "modality": {"0": "CT"},
#     "name": "liver_all",
#     "numTest": 0,
#     "numTraining": 0,
#     "reference": "Zhejiang University",
#     "release": "11/12/2024",
#     "tensorImageSize": "3D",
#     "test": [],
#     "training": [],  # need
#     "validation": [],  # need
# }


# basic_info = {
#     "description": "Liver Couinaud Segmentation",
#     "labels": {
#         "0": "background",
#         "1": "尾状叶",
#         "2": "左外叶上段",
#         "3": "左外叶下段",
#         "4": "左内叶",
#         "5": "右前叶下段",
#         "6": "右后叶下段",
#         "7": "右后叶上段",
#         "8": "右前叶上段",
#     },
#     "licence": "yt",
#     "modality": {"0": "CT"},
#     "name": "Couinaud_all",
#     "numTest": 0,
#     "numTraining": 0,
#     "reference": "Zhejiang University",
#     "release": "11/12/2024",
#     "tensorImageSize": "3D",
#     "test": [],
#     "training": [],  # need
#     "validation": [],  # need
# }


def get_list(data_dir):
    results = []

    images_dir_abs = data_dir
    labels_dir_abs = data_dir.replace("imagesTr", "labelsTr")

    image_names = os.listdir(images_dir_abs)
    for image_name in tqdm(image_names):
        label_name = image_name
        label_path_abs = os.path.join(labels_dir_abs, label_name)
        if not os.path.exists(label_path_abs):
            raise Exception(f"{label_path_abs} not exist")

        image_path = os.path.join(images_dir_abs, image_name)
        label_path = os.path.join(labels_dir_abs, label_name)
        results.append(
            {"image": image_path, "label": label_path, "label_vessel": label_path}
        )

    return results


def main(args):
    data = basic_info.copy()
    data_list = []
    for i in args.train_dir:
        data_list.extend(get_list(i))

    data_list = data_list
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
    data["numTraining"] = len(data["training"])
    data["numTest"] = len(data["test"])
    data["numValidation"] = len(data["validation"])

    # name_list = [
    #     "IRCADb_003.nii.gz",
    #     "IRCADb_005.nii.gz",
    #     "IRCADb_009.nii.gz",
    #     "IRCADb_013.nii.gz",
    #     "IRCADb_017.nii.gz",
    #     "IRCADb_018.nii.gz",
    # ]
    # name_list = ['hepaticvessel_083.nii.gz', 'hepaticvessel_293.nii.gz', 'hepaticvessel_102.nii.gz', 'hepaticvessel_274.nii.gz', 'hepaticvessel_131.nii.gz', 'hepaticvessel_296.nii.gz', 'hepaticvessel_386.nii.gz', 'hepaticvessel_194.nii.gz', 'hepaticvessel_321.nii.gz', 'hepaticvessel_092.nii.gz', 'hepaticvessel_023.nii.gz', 'hepaticvessel_200.nii.gz', 'hepaticvessel_096.nii.gz', 'hepaticvessel_307.nii.gz', 'hepaticvessel_079.nii.gz', 'hepaticvessel_340.nii.gz', 'hepaticvessel_279.nii.gz', 'hepaticvessel_398.nii.gz', 'hepaticvessel_385.nii.gz', 'hepaticvessel_019.nii.gz', 'hepaticvessel_286.nii.gz', 'hepaticvessel_309.nii.gz', 'hepaticvessel_089.nii.gz', 'hepaticvessel_425.nii.gz', 'hepaticvessel_262.nii.gz', 'hepaticvessel_443.nii.gz', 'hepaticvessel_256.nii.gz', 'hepaticvessel_039.nii.gz', 'hepaticvessel_409.nii.gz', 'hepaticvessel_384.nii.gz', 'hepaticvessel_050.nii.gz', 'hepaticvessel_160.nii.gz', 'hepaticvessel_290.nii.gz', 'hepaticvessel_422.nii.gz', 'hepaticvessel_407.nii.gz', 'hepaticvessel_369.nii.gz', 'hepaticvessel_280.nii.gz', 'hepaticvessel_284.nii.gz', 'hepaticvessel_433.nii.gz', 'hepaticvessel_456.nii.gz', 'hepaticvessel_423.nii.gz', 'hepaticvessel_111.nii.gz', 'hepaticvessel_208.nii.gz', 'hepaticvessel_322.nii.gz', 'hepaticvessel_027.nii.gz', 'hepaticvessel_011.nii.gz', 'hepaticvessel_234.nii.gz', 'hepaticvessel_248.nii.gz', 'hepaticvessel_406.nii.gz', 'hepaticvessel_053.nii.gz', 'hepaticvessel_282.nii.gz', 'hepaticvessel_225.nii.gz', 'hepaticvessel_255.nii.gz', 'hepaticvessel_368.nii.gz', 'hepaticvessel_044.nii.gz', 'hepaticvessel_110.nii.gz', 'hepaticvessel_161.nii.gz', 'hepaticvessel_094.nii.gz', 'hepaticvessel_215.nii.gz', 'hepaticvessel_240.nii.gz', 'hepaticvessel_165.nii.gz', 'hepaticvessel_291.nii.gz', 'hepaticvessel_217.nii.gz', 'hepaticvessel_013.nii.gz', 'hepaticvessel_442.nii.gz', 'hepaticvessel_085.nii.gz', 'hepaticvessel_336.nii.gz', 'hepaticvessel_371.nii.gz', 'hepaticvessel_416.nii.gz', 'hepaticvessel_259.nii.gz', 'hepaticvessel_061.nii.gz', 'hepaticvessel_411.nii.gz', 'hepaticvessel_177.nii.gz', 'hepaticvessel_404.nii.gz', 'hepaticvessel_098.nii.gz', 'hepaticvessel_420.nii.gz', 'hepaticvessel_258.nii.gz', 'hepaticvessel_042.nii.gz', 'hepaticvessel_195.nii.gz', 'hepaticvessel_314.nii.gz', 'hepaticvessel_051.nii.gz', 'hepaticvessel_223.nii.gz', 'hepaticvessel_270.nii.gz', 'hepaticvessel_184.nii.gz', 'hepaticvessel_159.nii.gz', 'hepaticvessel_196.nii.gz', 'hepaticvessel_318.nii.gz', 'hepaticvessel_088.nii.gz', 'hepaticvessel_246.nii.gz', 'hepaticvessel_341.nii.gz']
    # name_list = ['PuJian_20241109_022_0001.nii.gz', 'PuJian_20241109_088_0002.nii.gz', 'PuJian_20241109_006_0001.nii.gz', 'PuJian_20241109_001_0002.nii.gz', 'PuJian_20241109_021_0002.nii.gz', 'PuJian002_0002.nii.gz', 'PuJian044_0001.nii.gz', 'PuJian033_0001.nii.gz', 'PuJian043_0002.nii.gz', 'PuJian014_0002.nii.gz', 'PuJian_20241109_067_0001.nii.gz', 'PuJian_20241109_010_0002.nii.gz', 'PuJian_20241109_047_0001.nii.gz', 'PuJian037_0002.nii.gz', 'PuJian004_0002.nii.gz', 'PuJian_20241109_034_0001.nii.gz', 'PuJian_20241109_072_0001.nii.gz', 'PuJian021_0002.nii.gz', 'PuJian_20241109_026_0002.nii.gz', 'PuJian_20241109_019_0002.nii.gz', 'PuJian_20241109_080_0002.nii.gz', 'PuJian_20241109_090_0002.nii.gz', 'PuJian005_0002.nii.gz', 'PuJian018_0002.nii.gz', 'PuJian_20241109_046_0002.nii.gz', 'PuJian_20241109_050_0002.nii.gz', 'PuJian_20241109_039_0001.nii.gz', 'PuJian006_0001.nii.gz', 'PuJian053_0002.nii.gz', 'PuJian_20241109_005_0001.nii.gz', 'PuJian012_0002.nii.gz', 'PuJian_20241109_073_0002.nii.gz', 'PuJian_20241109_076_0001.nii.gz', 'PuJian_20241109_058_0002.nii.gz', 'PuJian_20241109_051_0002.nii.gz', 'PuJian022_0002.nii.gz', 'PuJian029_0001.nii.gz', 'PuJian_20241109_028_0001.nii.gz', 'PuJian006_0002.nii.gz', 'PuJian031_0001.nii.gz', 'PuJian032_0002.nii.gz', 'PuJian049_0002.nii.gz', 'PuJian027_0001.nii.gz', 'PuJian_20241109_082_0001.nii.gz', 'PuJian_20241109_070_0002.nii.gz', 'PuJian_20241109_044_0001.nii.gz', 'PuJian019_0001.nii.gz', 'PuJian_20241109_006_0002.nii.gz', 'PuJian_20241109_091_0001.nii.gz', 'PuJian_20241109_041_0001.nii.gz', 'PuJian_20241109_064_0001.nii.gz', 'PuJian_20241109_073_0001.nii.gz', 'PuJian053_0001.nii.gz', 'PuJian_20241109_078_0002.nii.gz', 'PuJian040_0001.nii.gz', 'PuJian_20241109_025_0002.nii.gz', 'PuJian_20241109_036_0001.nii.gz', 'PuJian_20241109_063_0001.nii.gz', 'PuJian_20241109_074_0001.nii.gz', 'PuJian007_0001.nii.gz', 'PuJian046_0002.nii.gz', 'PuJian_20241109_067_0002.nii.gz', 'PuJian028_0001.nii.gz', 'PuJian044_0002.nii.gz',  'PuJian010_0002.nii.gz', 'PuJian_20241109_043_0002.nii.gz', 'PuJian_20241109_091_0002.nii.gz', 'PuJian025_0002.nii.gz', 'PuJian052_0001.nii.gz', 'PuJian_20241109_012_0002.nii.gz', 'PuJian_20241109_079_0001.nii.gz', 'PuJian048_0002.nii.gz', 'PuJian_20241109_054_0002.nii.gz', 'PuJian_20241109_005_0002.nii.gz', 'PuJian035_0001.nii.gz', 'PuJian_20241109_007_0001.nii.gz', 'PuJian_20241109_070_0001.nii.gz', 'PuJian_20241109_050_0001.nii.gz', 'PuJian_20241109_033_0001.nii.gz', 'PuJian040_0002.nii.gz', 'PuJian_20241109_033_0002.nii.gz', 'PuJian_20241109_087_0001.nii.gz', 'PuJian_20241109_019_0001.nii.gz', 'PuJian_20241109_012_0001.nii.gz', 'PuJian051_0001.nii.gz',  'PuJian_20241109_000_0001.nii.gz', 'PuJian_20241109_077_0002.nii.gz', 'PuJian_20241109_030_0002.nii.gz']

    # data["test"] = [i for i in data_list if i["image"].split("/")[-1] in name_list]

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
        # default="/home/zhaozihao/Vessel_Segmentation/1212_Couinaud_dataset.json",
        # default="/home/zhaozihao/Vessel_Segmentation/1212_Vessel_dataset.json",
        # default="/home/zhaozihao/Vessel_Segmentation/PuJian_Vessel_dataset.json",
        # default="/home/zhaozihao/Vessel_Segmentation/test_private.json",
        # default="/home/zhaozihao/Vessel_Segmentation/test_HepaticVessel.json",
        default="/home/zhaozihao/Vessel_Segmentation/lib/datasets/data_json/PuJian_Vessel_dataset_new.json",
    )
    args = parser.parse_args()

    args.train_dir = [
        "/data1/zzh/dataset_1210/Vessel/abdomen_3d_reconstruction/imagesTr",
        "/data1/zzh/dataset_1210/Vessel/PuJian_20241109/imagesTr",
        # "/data1/zzh/dataset_1210/Vessel/Task08_HepaticVessel/imagesTr",
    ]

    # args.train_dir = [
    #     "/data1/zzh/dataset_1210/Couinaud/abdomen_3d_reconstruction/imagesTr",
    #     "/data1/zzh/dataset_1210/Couinaud/PuJian_20241109/imagesTr",
    # ]
    main(args)
