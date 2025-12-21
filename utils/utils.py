import errno
import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.distributed as dist
import shutil
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import sys





def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def gen_448(image, label):
    """
    cut a 512*512 ct img to 448 * 448

    """
    if image.shape[-1] == 512:
        if len(image.shape) == 2 and len(label.shape) == 2:
            image1 = image[32:480, 32:480]
            label1 = label[32:480, 32:480]
        else:
            image1 = image[..., 32:480, 32:480]
            label1 = label[..., 32:480, 32:480]
    else:
        image1, label1 = image, label
    return image1, label1

def slice_private_data(input_dir, output_dir, name):
    number = random.random()
    if number >= 0.9:
        output_dir = output_dir + "/test"
        shutil.copy(os.path.join(input_dir, name),os.path.join(output_dir, name))
        print(os.path.join(input_dir, name), "test done~")
    else:
        output_dir = output_dir + ("/val" if 0.8 <= number < 0.9 else "/train")
        data = np.load(os.path.join(input_dir, name), allow_pickle=True)
        ct, liver, tumor = data["ct"], data["liver_mask"], data["tumor_mask"]
        # ct, meta = load_ct(os.path.join(input_dir, name), return_meta=True)
        # liver, mask, seg_num = load_mask(os.path.join(input_dir, name[3:]), return_meta=True)

        # liver[tumor > 0] = tumor[tumor > 0]
        ct = ct.transpose((1, 0, 2, 3))
        # mask = liver.transpose((1, 0, 2, 3))
        mask = tumor.transpose((1, 0, 2, 3))
        ct, mask = gen_448(ct, mask)
        # mask[mask == 7] = 3
        # assert seg_num[0] < seg_num[1] < seg_num[2]

        for i, (ct_i, mask_i) in enumerate(zip(ct, mask)):
            np.savez_compressed(
                os.path.join(output_dir, name.strip(".npz") + "_" + str(i) + ".npz"),
                ct=ct_i,
                mask=mask_i,
            )
        print(os.path.join(input_dir, name), "train / val done~")


def setup(rank, world_size, addr="127.0.0.1", port="1895"):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def load_ct_to_numpy(data_path):
    if type(data_path) != str:
        data_path = data_path.name

    image = nib.load(data_path)
    data = image.get_fdata()

    data = np.rot90(data, k=1, axes=(0, 1))

    data[data < -150] = -150
    data[data > 250] = 250

    data = data - np.amin(data)
    data = data / np.amax(data) * 255
    data = data.astype("uint8")

    print(data.shape)
    return [data[..., i] for i in range(data.shape[-1])]


def load_pred_volume_to_numpy(data_path):
    if type(data_path) != str:
        data_path = data_path.name

    image = nib.load(data_path)
    data = image.get_fdata()

    data = np.rot90(data, k=1, axes=(0, 1))
    
    # data[(data !=9) &(data!=10)&(data!=14)] = 0
    # data[(data == 9) | (data == 10) | (data == 14)] = 1

    # data[data ==1] = 0
    # data[data==2] = 1

    data[data >=9] = 0
    data = data.astype("uint8")

    print(data.shape)
    return [data[..., i] for i in range(data.shape[-1])]


if __name__ == "__main__":

    # path = Path("/data1/zzh/Vessel/abdomen_3d_reconstruction/labelsTr/")
    # save_path = Path("/data1/zzh/Vessel/abdomen_3d_reconstruction/trainmask/")
    # a = list(path.glob("*nii.gz"))

    path = Path("/data1/zzh/Vessel/abdomen_3d_reconstruction/labelsTr/")
    save_path = Path("/data1/zzh/Vessel/abdomen_3d_reconstruction/trainmask2/")
    a = list(path.glob("*nii.gz"))
    
    print(len(a))
    for i in tqdm(a):
        img_list = load_pred_volume_to_numpy(str(i))
        for j, img in enumerate(img_list):
            img = Image.fromarray(img)
            img.save(save_path / (i.stem + f"_{j}.png"))
        # sys.exit()
