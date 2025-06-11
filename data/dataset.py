'''
Dataloader to process Adobe Image Matting Dataset.

From GCA_Matting(https://github.com/Yaoyi-Li/GCA-Matting/tree/master/dataloader)
'''
import os
import glob
import logging
import os.path as osp
import functools
import numpy as np
import torch
import cv2
import math
import numbers
import random
import pickle
from   torch.utils.data import Dataset, DataLoader
from   torch.nn import functional as F
from   torchvision import transforms
from easydict import EasyDict

class Prefetcher():
    """
    Modified from the data_prefetcher in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.orig_loader = loader
        self.stream = torch.cuda.Stream()
        self.next_sample = None

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_sample.items():
                if isinstance(value, torch.Tensor):
                    self.next_sample[key] = value.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        sample = self.next_sample
        if sample is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key].record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            # throw stop exception if there is no more data to perform as a default dataloader
            raise StopIteration("No samples in loader. example: `iterator = iter(Prefetcher(loader)); "
                                "data = next(iterator)`")
        return sample

    def __iter__(self):
        self.loader = iter(self.orig_loader)
        self.preload()
        return self


class GetData(object):
    def __init__(self,
                 img_dir="",
                 label_dir="",
                 ):
        img_names = os.listdir(img_dir)
        label_names = os.listdir(label_dir)
        self.images = [os.path.join(img_dir, name) for name in img_names if name.lower().endswith('.jpg')]
        self.labels = [os.path.join(label_dir, name) for name in label_names if name.lower().endswith('.png')]
        
        
    def __len__(self):
        return len(self.images)

class Resize:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        image_resized = cv2.resize(image, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, self.crop_size[::-1], interpolation=cv2.INTER_NEAREST)
        sample['image'] = image_resized
        sample['label'] = mask_resized
        return sample
    
class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, sample):
        image = sample['image'] / 255.0
        # image = (image - self.mean) / self.std
        sample['image'] = image
        return sample

class RandomCrop:
    def __init__(self, crop_size):
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']
        _, H, W = image.shape
        if H < self.crop_height or W < self.crop_width:
            image_resized = cv2.resize(image, (self.crop_width, self.crop_height), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (self.crop_width, self.crop_height), interpolation=cv2.INTER_NEAREST)
            sample['image'] = image_resized
            sample['label'] = mask_resized
            return sample

        top = random.randint(0, H - self.crop_height)
        left = random.randint(0, W - self.crop_width)

        image_cropped = image[:, top:top + self.crop_height, left:left + self.crop_width]
        if mask.ndim == 3:
            mask_cropped = mask[:, top:top + self.crop_height, left:left + self.crop_width]
        else:
            mask_cropped = mask[top:top + self.crop_height, left:left + self.crop_width]

        sample['image'] = image_cropped
        sample['label'] = mask_cropped
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
     
        image = image.transpose((2, 0, 1)).astype("float32")
        label = np.expand_dims(label.astype("float32"), axis=0)
        
        sample['image'], sample['label'] = torch.from_numpy(image), torch.from_numpy(label)
        
        return sample



class DataGenerator(Dataset):
    def __init__(self, data, crop_size, phase):
        self.crop_size = crop_size
        self.images = data.images
        self.labels = data.labels

        train_trans = [
            Normalize(),
            RandomCrop(self.crop_size),
            ToTensor() ]

        test_trans = [
            Normalize(),
            Resize(self.crop_size),
            ToTensor() ]
        
        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'test':
                transforms.Compose(test_trans)
        }[phase]


    def __getitem__(self, idx):
        idx = idx % len(self.images)
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.labels[idx], 0)
        H, W = label.shape
        image_name = os.path.basename(self.images[idx])
    
        sample = {'image': image, 'label': label, 'image_name': image_name, 'hw': (H, W)}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

   