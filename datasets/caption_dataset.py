# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/25 16:38

import copy
import json
import os
import re
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
from utils.preprocess import resize_frames

class CaptionDataset(data.Dataset):
    def __init__(self, args, pct=1.0, stage="train"):
        super(CaptionDataset, self).__init__()
        self.args = args
        self.stage = stage
        self.image_slice_num = args.slice_num
        self.dataset_root = args.dataset_root
        self.buffer_root = args.buffer_root
        if stage == 'train':
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', '55', 'train_val', "multimodal", 'train.json')))

        elif stage == 'valid':
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', '55', 'train_val', "multimodal", 'valid.json')))
        else:
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', '55', 'test', 'test.json')))

        if pct != 1:
            self.dataset = self.dataset[:int(len(self.dataset) * pct)]

        if self.stage == 'train':
            self.visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=args.image_size, scale=(0.6, 1.0)),
            ])
        else:
            self.visual_transform = transforms.Compose([
                transforms.Resize([args.image_size, args.image_size]),
            ])

        self.normalize = transforms.Normalize(mean=[0.5,], std=[0.5,])
        print(f"+ {stage} data num = {len(self.dataset)}")


    def __len__(self):
        """Return the total number of images."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.load_sample(index)
        while data is None:
            index = (index + 1) % len(self.dataset)
            data = self.load_sample(index)
        return data

    def load_sample(self, index):
        data_id = self.dataset[index]['dataID']

        img_path = self.dataset[index]['ct']
        buffer_path = os.path.join(self.buffer_root, '/'.join(img_path.split('/')[-3:]) + '.npz')
        image = np.load(buffer_path)['data']
        image = image.transpose(1, 0, 2, 3)
        ## sample to certain length for input
        inds = resize_frames(list(range(image.shape[0])), ex_fnum=self.image_slice_num)
        image = image[inds]

        image_tensor = torch.from_numpy(image)  # tensor with shape: (slice num, window num, h, w)
        image_tensor = self.visual_transform(image_tensor)  # aug tensor in (0-255)
        image_tensor = image_tensor / 255  # normalize to 0-1
        image_tensor = self.normalize(image_tensor)  # normalize with mean and std

        report = self.dataset[index]['report']
        caption = self.clean_report(report)

        return {'image': image_tensor, 'caption': caption, 'name': data_id, 'image_path': img_path}

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

    def make_one_hot_label(self, label_list, num_classes):
        """
        input:
        scalar label list, like[2, 3]

        output:
        one_hot_label: list in one-hot-like format label, like [0,0,1,1]
        """
        one_hot_label = [0 for _ in range(num_classes)]
        for label in label_list:
            one_hot_label[label] = 1
        return one_hot_label

