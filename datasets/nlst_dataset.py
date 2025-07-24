# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 16:38

import json
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from utils.preprocess import resize_frames

class NLSTDataset(data.Dataset):
    def __init__(self, args, pct=1.0, stage='train'):
        super(NLSTDataset, self).__init__()
        self.stage = stage
        self.args = args
        self.data_root = os.path.join(self.args.root_dir, 'datasets', 'nlst')
        if stage == 'train':
            self.dataset = json.load(open(os.path.join(self.data_root, f"cancer_classification_train_balanced.json")))
        elif stage == 'valid':
            self.dataset = json.load(open(os.path.join(self.data_root, f"cancer_classification_valid_balanced.json")))
        else:
            self.dataset = json.load(open(os.path.join(self.data_root, f"cancer_classification_test_balanced.json")))

        self.dataset = self.dataset[:int(len(self.dataset) * pct)]
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
        label = self.dataset[index]['label']
        path = self.dataset[index]['dcm_paths']

        tmp_dir = os.path.split(path[0])[0]
        save_dir, save_name = os.path.split(tmp_dir)
        save_dir = save_dir.replace('data/NLST', 'data/maliangdi/nlst/buffer', 1)
        buffer_path = os.path.join(save_dir, save_name + '.npz')

        try:
            image = np.load(buffer_path)['data']
        except:
            return None

        # convert image from (window num, slice num, h, w) to (slice num, window num, h, w)
        image = image.transpose(1, 0, 2, 3)

        ## sample to certain length for input
        inds = resize_frames(list(range(image.shape[0])), ex_fnum=self.args.slice_num)
        image = image[inds]

        image_tensor = torch.from_numpy(image)  # tensor with shape: (slice num, window num, h, w)
        image_tensor = self.visual_transform(image_tensor)  # aug tensor in (0-255)
        image_tensor = image_tensor / 255  # normalize to 0-1
        image_tensor = self.normalize(image_tensor)  # normalize with mean and std

        return {'image': image_tensor, 'label': label, 'image_path': os.path.split(path[0])[0]}

    def __len__(self):
        """Return the total number of images."""
        return self.length_of_dataset
