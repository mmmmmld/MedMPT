# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 16:38

import copy
import json
import os
import random
import re
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

import pydicom
import scipy.ndimage
from utils.preprocess import resize_frames

def build_cv_dataset(root_dir, task):
    data_root = os.path.join(root_dir, 'datasets', 'mosmeddata')
    dataset = []
    labels = []
    if task == 'diagnosis':
        load_data = json.load(open(os.path.join(data_root, f"data_dict_in_diagnosis.json")))
        for k, data_list in load_data.items():
            for data in data_list:
                data["label"] = int(k)
                dataset.append(data)
                labels.append(int(k))
    elif task == 'stage':
        load_data = json.load(open(os.path.join(data_root, f"data_dict_in_stage.json")))
        for k, data_list in load_data.items():
            if k == "CT-4":
                continue
            for data in data_list:
                data["label"] = int(k[-1])
                dataset.append(data)
                labels.append(int(k[-1]))
    else:
        raise ValueError
    return dataset, labels

class MosMedDataset(data.Dataset):
    """
    crossvalidation
    """
    def __init__(self, args, data_list, pct=1.0, stage='train'):
        super(MosMedDataset, self).__init__()
        self.stage = stage
        self.args = args
        self.target_z_spacing = 5
        self.WindowWidth = [1500, 2000, 400]
        self.WindowLocation = [-400, -500, 60]
        self.data_root = os.path.join(self.args.root_dir, 'datasets', 'mosmeddata')
        self.dataset = data_list[:int(len(data_list) * pct)]
        self.length_of_dataset = len(self.dataset)
        print(f"{stage} data = {self.length_of_dataset}")

        if self.stage == 'train':
            self.visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.args.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=[.1, 3.])], p=0.5),
            ])
        else:
            self.visual_transform = transforms.Compose([
                transforms.Resize([self.args.image_size, self.args.image_size]),
            ])
        self.normalize = transforms.Normalize(mean=[0.5,], std=[0.5,])

    def __getitem__(self, index):
        data = self.load_sample(index)
        while data is None:
            # print('the item is wrong')
            index = (index + 1) % self.length_of_dataset
            data = self.load_sample(index)
        return data

    def load_sample(self, index):
        data_id = self.dataset[index]['study_id']
        label = self.dataset[index]['label']
        img_path = self.dataset[index]['img_path']
        image = self.load_nii_data(img_path, resample=self.args.resample)

        # convert image from (window num, slice num, h, w) to (slice num, window num, h, w)
        image = image.transpose(1, 0, 2, 3)

        ## sample to certain length for input
        inds = resize_frames(list(range(image.shape[0])), ex_fnum=self.args.slice_num)
        image = image[inds]

        image_tensor = torch.from_numpy(image)  # tensor with shape: (slice num, window num, h, w)
        image_tensor = self.visual_transform(image_tensor)  # aug tensor in (0-255)
        image_tensor = image_tensor / 255  # normalize to 0-1
        image_tensor = self.normalize(image_tensor)  # normalize with mean and std

        return {'image': image_tensor, 'label': label, 'image_path': img_path, 'data_id': str(data_id)}

    def __len__(self):
        """Return the total number of images."""
        return self.length_of_dataset
    
    def load_nii_data(self, p, resample=False):
        nii_img = nib.load(p)
        img_data = nii_img.get_fdata()  # (h, w, d)
        if resample:
            space_dim = nii_img.header['pixdim'][1:4]  # (px, py, pz)
            real_shape_z = img_data.shape[-1] * space_dim[-1]
            new_shape_z = np.round(real_shape_z / self.target_z_spacing)
            real_resize_factor_z = new_shape_z / img_data.shape[-1]
            real_resize_factor = [1, 1, real_resize_factor_z]
            img_data = scipy.ndimage.interpolation.zoom(img_data, real_resize_factor, mode='nearest')
        ## window transform
        window_image = []
        for ww, wl in zip(self.WindowWidth, self.WindowLocation):
            w_img = self.window_transform(img_data, ww, wl)  # (h, w, d)
            window_image.append(w_img)
        window_image = np.stack(window_image, axis=0)  # (window num, h, w, d)
        window_image = np.transpose(window_image, (0, 3, 1, 2))
        window_image = np.rot90(window_image, 1, (-2, -1))
        return window_image


    def window_transform(self, image, WindowWidth=1500, WindowLocation=-400, normal_to_one=False):
        """
        return: truncate image according to window center and window width
        and normalized to [0,1]
        """
        raw_image = copy.deepcopy(image)
        minv = float(WindowLocation) - 0.5 * float(WindowWidth)
        maxv = float(WindowLocation) + 0.5 * float(WindowWidth)
        raw_image = np.where(raw_image > maxv, maxv, raw_image)
        raw_image = np.where(raw_image < minv, minv, raw_image)
        newimg = (raw_image - minv) / (maxv - minv)
        return (newimg * 255).astype('uint8')

