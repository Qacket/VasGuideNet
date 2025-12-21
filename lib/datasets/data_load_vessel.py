import os
import random
import sys

from scipy import ndimage
import torch
sys.path.append(os.path.dirname(sys.path[0]))
import json
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import numpy as np
from PIL import Image,ImageFile
from torchvision import transforms
from tqdm import tqdm
import cv2

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        label[(label !=9) &(label!=10)&(label!=14)] = 0
        label[(label == 9) | (label == 10) | (label == 14)] = 1
        # -1000 - 1000 =>  (-1 1)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class PNG_DATA_LOADER_VESSEL(Dataset):
    def __init__(self, cfg, is_train) -> None:
        self.cfg = cfg
        self.is_train = is_train

        self.images = []
        self.labels = []

        data_sources = cfg.DATA.TRAIN_DATA if is_train is True else cfg.DATA.VAL_DATA
        

        for data_source in data_sources:
            print("load :\t", data_source)
            
            _images =[]
            _labels = []

            data_root, json_path, json_name = data_source
            path = os.path.join(data_root, json_path, json_name)

            data_ano_list = json.load(open(path, 'r'))
            for data in data_ano_list:
                if 'img' not in data or 'label' not in data:
                    print("image or label not in data")
                    continue
                else:
                    _images.append(data['img'])
                    _labels.append(data['label'])


            self.images.extend(_images)
            self.labels.extend(_labels)
            print("num of data:{}".format(len(_images)))
        
        self.perm = np.random.RandomState(123).permutation(len(self.images))    
        perm = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]

        print('+' *20)
        for i in range(10):
            print(self.images[i], self.labels[i])
        
        data_shape = cfg.INPUT.DATA_SHAPE

        if is_train:
            self.transforms = transforms.Compose([
                RandomGenerator(output_size=[data_shape[1], data_shape[2]])
            ])
            # self.transforms =transforms.Compose([
            #     transforms.Resize((data_shape[1], data_shape[2])),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomVerticalFlip(),
            #     transforms.RandomRotation(30),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # ])
            print("train transforms", self.transforms)
        else:
            self.transforms = transforms.Compose([
                RandomGenerator(output_size=[data_shape[1], data_shape[2]])
            ])
            # self.transforms = transforms.Compose([
            #     transforms.Resize((data_shape[1], data_shape[2])),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # ])
            print("test transforms", self.transforms)

    def __len__(self):
        return len(self.images)
    
    def _rand_another(self):
        return np.random.randint(0, len(self))
    
    def get_data(self, img_fname, label_fname):
        try:
            img = Image.open(img_fname)
            label = Image.open(label_fname)
            return {'success':True, 'img':img, 'label':label}
        except Exception as e:
            print('get image failed pillow, {}, {} '.format(img_fname, e),flush=True)
            return {'success':False, img:None, label:None}
        
    
    def __getitem__(self, index):
        success = False
        times = 0
        while not success and times < 10:
            times+=1
            img_fname, label_fname = self.images[index], self.labels[index]
            try:
                ret = self.get_data(img_fname, label_fname)
                success= ret['success']
                img = ret['img']
                label = ret['label']
            except Exception as e:
                print(e, flush=True)
            if not success:
                index = self._rand_another()
                continue

        sample = {'image': np.array(img), 'label': np.array(label)}
        sample = self.transforms(sample)
        # img, label = self.transforms(img), self.transforms(label)
        if times>=10:
            raise Exception("Failed to load image and label")
        return sample["image"].repeat((3,1,1)), sample["label"]