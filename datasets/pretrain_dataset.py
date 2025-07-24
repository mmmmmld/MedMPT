# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 16:38

import copy
import json
import os
import random
import re
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
import pydicom
import scipy.ndimage
from utils.preprocess import resize_frames


class PretrainDataset(data.Dataset):
    def __init__(self, args, stage='train'):
        super(PretrainDataset, self).__init__()
        self.stage = stage
        self.args = args
        self.dataset = json.load(open("/path/to/pretraining/dataset"))

        if stage == 'train':
            self.dataset = self.dataset[:int(len(self.dataset) * 0.95)]
            self.dataset = self.dataset[:int(len(self.dataset) * self.args.train_pct)]
        else:
            self.dataset = self.dataset[int(len(self.dataset) * 0.95):]
            self.dataset = self.dataset[:int(len(self.dataset) * self.args.val_pct)]

        self.length_of_dataset = len(self.dataset)

        if self.stage == 'train':
            self.visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.args.image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
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
            index = (index + 1) % self.length_of_dataset
            data = self.load_sample(index)
        return data

    def load_sample(self, index):
        report = self.dataset[index]['findings']
        path = self.dataset[index]['buffer_path']
        slice_num = self.dataset[index]['num']

        report = self.clean_report(report)

        ### read from buffer path
        image = np.load(path)['data']

        # convert image from (window num, slice num, h, w) to (slice num, window num, h, w)
        image = image.transpose(1, 0, 2, 3)

        ## sample to certain length for input
        inds = resize_frames(list(range(image.shape[0])), ex_fnum=self.args.slice_num)
        image = image[inds]
        image_tensor = torch.from_numpy(image)  # tensor with shape: (slice num, window num, h, w)

        if self.stage == 'train':
            aug_image_0 = self.visual_transform(image_tensor)  # aug tensor in (0-255)
            aug_image_1 = self.visual_transform(image_tensor)  # aug tensor in (0-255)
            aug_image_0 = self.normalize(aug_image_0 / 255)  # normalize to 0-1, normalize with mean and std
            aug_image_1 = self.normalize(aug_image_1 / 255)  # normalize to 0-1, normalize with mean and std
            if self.args.sentence_shuffle:
                raise NotImplementedError
            else:
                shuffle_report_0 = report
                shuffle_report_1 = report
            return {'image': [aug_image_0, aug_image_1], 'caption': [shuffle_report_0, shuffle_report_1], 'image_path': path}
        else:
            aug_image_0 = self.visual_transform(image_tensor)  # aug tensor in (0-255)
            aug_image_0 = self.normalize(aug_image_0 / 255)
            return {'image': [aug_image_0], 'caption': [report], 'image_path': path}

    def __len__(self):
        """Return the total number of images."""
        return self.length_of_dataset

    def clean_report(self, content):
        report = copy.deepcopy(content)
        # remove special sign
        report = report.replace('\n', '')
        report = re.sub(r'\s', '', report)
        report = re.sub(r' ', '', report)
        report = re.sub('\'', '', report)
        # convert english sign to chinese sign
        report = re.sub('[:]+', '：', report)
        report = re.sub('[;]+', '；', report)
        report = re.sub('[?]+', '？', report)
        report = re.sub('[,]+', '，', report)
        report = re.sub('[!]+', '！', report)
        # remove repeat signs
        report = re.sub('[！]{2,}', '！', report)
        report = re.sub('[？]{2,}', '？', report)
        report = re.sub('[；]{2,}', '；', report)
        report = re.sub('[、]{2,}', '、', report)
        report = re.sub('[，]{2,}', '，', report)
        report = re.sub('[。]{2,}', '。', report)

        sentences = report.split('：')
        if '对比' in sentences[0]:
            del sentences[0]
        report = '：'.join(sentences)
        return report

