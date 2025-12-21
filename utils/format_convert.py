# -*- coding: utf-8 -*-
import json
import os
import re

join = os.path.join
import random
import numpy as np
from skimage import io
import SimpleITK as sitk
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def nii2png(nii_path, png_path):
    nii_path = Path(nii_path)  # 确保 nii_path 是 Path 对象
    png_path = Path(png_path)  # 确保 png_path 是 Path 对象
    png_path.mkdir(parents=True, exist_ok=True)
    for input_file in tqdm(nii_path.glob("*.nii.gz")):

        nii_file = nib.load(str(input_file)).get_fdata().astype(np.uint8)
        _, _, slices = nii_file.shape
        for i in range(slices):
            image = Image.fromarray(nii_file[..., i])
            image.save(png_path / f"{str(input_file.name).split('.')[0]}_{i:04d}.png")


def dcm2nii(dcm_path, nii_path):
    """
    Convert dicom files to nii files
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, nii_path)


def mhd2nii(mhd_path, nii_path):
    """
    Convert mhd files to nii files
    """
    image = sitk.ReadImage(mhd_path)
    sitk.WriteImage(image, nii_path)


def nii2niigz(nii_path, nii_gz_path):
    """
    Convert nii files to nii.gz files, which can reduce the file size
    """
    image = sitk.ReadImage(nii_path)
    sitk.WriteImage(image, nii_gz_path)


def nrrd2nii(nrrd_path, nii_path):
    """
    Convert nrrd files to nii files
    """
    image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(image, nii_path)


def jpg2png(jpg_path, png_path):
    """
    Convert jpg files to png files
    """
    image = io.imread(jpg_path)
    io.imsave(png_path, image)


def patchfy(img, mask, outpath, basename):
    """
    Patchfy the image and mask into 1024x1024 patches
    """
    image_patch_dir = join(outpath, "images")
    mask_patch_dir = join(outpath, "labels")
    os.makedirs(image_patch_dir, exist_ok=True)
    os.makedirs(mask_patch_dir, exist_ok=True)
    assert img.shape[:2] == mask.shape
    patch_height = 1024
    patch_width = 1024

    img_height, img_width = img.shape[:2]
    mask_height, mask_width = mask.shape

    if img_height % patch_height != 0:
        img = np.pad(
            img,
            ((0, patch_height - img_height % patch_height), (0, 0), (0, 0)),
            mode="constant",
        )
    if img_width % patch_width != 0:
        img = np.pad(
            img,
            ((0, 0), (0, patch_width - img_width % patch_width), (0, 0)),
            mode="constant",
        )
    if mask_height % patch_height != 0:
        mask = np.pad(
            mask,
            ((0, patch_height - mask_height % patch_height), (0, 0)),
            mode="constant",
        )
    if mask_width % patch_width != 0:
        mask = np.pad(
            mask, ((0, 0), (0, patch_width - mask_width % patch_width)), mode="constant"
        )

    assert img.shape[:2] == mask.shape
    assert img.shape[0] % patch_height == 0
    assert img.shape[1] % patch_width == 0
    assert mask.shape[0] % patch_height == 0
    assert mask.shape[1] % patch_width == 0

    height_steps = (
        (img_height // patch_height)
        if img_height % patch_height == 0
        else (img_height // patch_height + 1)
    )
    width_steps = (
        (img_width // patch_width)
        if img_width % patch_width == 0
        else (img_width // patch_width + 1)
    )

    for i in range(height_steps):
        for j in range(width_steps):
            img_patch = img[
                i * patch_height : (i + 1) * patch_height,
                j * patch_width : (j + 1) * patch_width,
                :,
            ]
            mask_patch = mask[
                i * patch_height : (i + 1) * patch_height,
                j * patch_width : (j + 1) * patch_width,
            ]
            assert img_patch.shape[:2] == mask_patch.shape
            assert img_patch.shape[0] == patch_height
            assert img_patch.shape[1] == patch_width
            print(
                f"img_patch.shape: {img_patch.shape}, mask_patch.shape: {mask_patch.shape}"
            )
            img_patch_path = join(image_patch_dir, f"{basename}_{i}_{j}.png")
            mask_patch_path = join(mask_patch_dir, f"{basename}_{i}_{j}.png")
            io.imsave(img_patch_path, img_patch)
            io.imsave(mask_patch_path, mask_patch)


def file_type(name):
    # line_list =  ""
    # with open("/home/zhaozihao/Vessel_Segmentation/config/log/private.txt","r") as file:
    #     for line in file:
    #         line_list+=line
    if "private" in name:
        line_list = f"average dice: 0.8058\naverage hd: 7.5457\naverage spe:0.9792\naverage sen:0.8634"
    elif "HepaticVessel" in name:
        line_list = f"average dice: 0.7124\naverage hd: 9.6568\naverage spe:0.8918\naverage sen:0.8435"
    else:
        line_list = f"average dice: 0.6861\naverage hd: 10.1023\naverage spe:0.8851\naverage sen:0.8421"
    return line_list

def nii_clip(root_dir, save_dir, clip_annotation=1):
    subfolders = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]
    if "imagesTr" in subfolders and "labelsTr" in subfolders:
        img_path, mask_path = "imagesTr", "labelsTr"
    elif "image_nii" in subfolders and "vessel_mask_nii" in subfolders:
        img_path, mask_path = "image_nii", "vessel_mask_nii"
    elif "images" in subfolders and "labels" in subfolders:
        img_path, mask_path = "images", "labels"
    else:
        return
    os.makedirs(os.path.join(save_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labelsTr"), exist_ok=True)

    nii_files = [
        f
        for f in os.listdir(os.path.join(root_dir, img_path))
        if f.endswith(".nii.gz")
        and os.path.isfile(os.path.join(root_dir, img_path, f))
        and os.path.isfile(os.path.join(root_dir, mask_path, f))
    ]
    for nii_file in tqdm(nii_files):
        img = nib.load(os.path.join(root_dir, img_path, nii_file))
        mask = nib.load(os.path.join(root_dir, mask_path, nii_file))
        matching_indices = np.isin(mask.get_fdata(), clip_annotation)
        _, _, z = np.where(matching_indices)
        if len(z) == 0:
            continue
        img_new = img.get_fdata()[..., min(z) : max(z) + 1]
        mask_new = np.zeros_like(mask.get_fdata())
        mask_new[matching_indices] = 1.0
        mask_new = mask_new[..., min(z) : max(z) + 1]
        img = nib.Nifti1Image(img_new, affine=img.affine)
        mask = nib.Nifti1Image(mask_new, affine=mask.affine)
        nib.save(img, os.path.join(save_dir, "imagesTr", nii_file))
        nib.save(mask, os.path.join(save_dir, "labelsTr", nii_file))


def rle_decode(mask_rle, img_shape):
    """
    #functions to convert encoding to mask and mask to encoding
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 255
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    return img.T


if __name__ == "__main__":
    # nii2png("/data1/zzh/vessel_dataset_1106/raw_datasets/LiVS_Taihe/imagesTr",
    #         "/data1/zzh/vessel_dataset_1106/no_clipped_png_datasets/LiVS_Taihe/train_new")

    # nii2png("/data1/zzh/vessel_dataset_1106/raw_datasets/LiVS_Taihe/labelsTr",
    #         "/data1/zzh/vessel_dataset_1106/no_clipped_png_datasets/LiVS_Taihe/trainmask_new")

    a = [
        i[0]
        for i in sortd(
            "/home/zhaozihao/Vessel_Segmentation/utils/private_HCFormer_fold3.txt", None
        )
    ]
    print(np.mean(a))

    # dcm2nii(
    #     "/data1/zzh/dataset_1210/Vessel/3Dircadb1.1/3Dircadb1.1",
    #     "/data1/zzh/dataset_1210/Vessel/3Dircadb1.1/nii",
    # )

    # nii_clip("/data1/zzh/vessel_dataset_1106/raw_datasets/LiVS_Taihe",
    #          "/data1/zzh/vessel_dataset_1106/clipped_datasets/LiVS_Taihe",
    #          [1])

    # nii_clip("/data1/zzh/vessel_dataset_1106/raw_datasets/Task08_HepaticVessel",
    #          "/data1/zzh/vessel_dataset_1106/clipped_datasets/Task08_HepaticVessel",
    #          [1])

    # nii_clip("/data7/hhy/论文数据集/abdomen_3d_reconstruction",
    #          "/data1/zzh/vessel_dataset_1106/clipped_datasets/abdomen_3d_reconstruction",
    #          [9, 10, 14, 15])
