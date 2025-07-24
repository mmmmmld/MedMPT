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
import torch.distributed as dist
from utils.preprocess import resize_frames


class MedDataset(data.Dataset):
    def __init__(self, input="ct,report,biomarker", class_num=55, image_size=224, image_slice_num=32,
                 bio_num=100, bio_v_null=0, bio_discrete=False, bio_normalize=False,
                 pct=1.0, stage="train", dataset_root="", buffer_root="", **kwargs):
        super(MedDataset, self).__init__()
        self.stage = stage
        self.input = input
        self.class_num = class_num
        self.image_slice_num = image_slice_num
        self.biomarker_num = bio_num
        self.biomarker_v_null = bio_v_null
        self.biomarker_discrete = bio_discrete
        self.biomarker_normalize = bio_normalize
        self.dataset_root = dataset_root
        self.buffer_root = buffer_root

        self.medicine_ids = json.load(open(os.path.join(self.dataset_root, 'medicines', str(self.class_num), 'medicine_to_label.json')))
        self.medicine_ids = dict([(int(v), k) for k, v in self.medicine_ids.items()])

        self.target_biomarkers = json.load(
            open(os.path.join(self.dataset_root, 'biomarkers', str(self.biomarker_num), 'target_biomarkers.json')))

        if stage == 'train':
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', str(self.class_num), 'train_val', "multimodal", 'train.json')))

        elif stage == 'valid':
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', str(self.class_num), 'train_val', "multimodal", 'valid.json')))
        else:
            self.dataset = json.load(open(
                os.path.join(self.dataset_root, 'medicines', str(self.class_num), 'test', 'test.json')))

        if pct != 1:
            self.dataset = self.dataset[:int(len(self.dataset) * pct)]

        if self.stage == 'train':
            self.visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=[.1, 3.])], p=0.5),
            ])
        else:
            self.visual_transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
            ])

        self.normalize = transforms.Normalize(mean=[0.5,], std=[0.5,])


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
        label = self.dataset[index]['label']  # (list of ind)
        one_hot_label = self.make_one_hot_label(label, num_classes=self.class_num)
        one_hot_label = torch.Tensor(one_hot_label)
        img_path = self.dataset[index]['ct']

        if 'ct' in self.input:
            buffer_path = os.path.join(self.buffer_root, '/'.join(img_path.split('/')[-3:]) + '.npz')
            image = np.load(buffer_path)['data']
            image = image.transpose(1, 0, 2, 3)
            ## sample to certain length for input
            inds = resize_frames(list(range(image.shape[0])), ex_fnum=self.image_slice_num)
            image = image[inds]
            image_tensor = torch.from_numpy(image)
            image_tensor = self.visual_transform(image_tensor)
            image_tensor = image_tensor / 255
            image_tensor = self.normalize(image_tensor)
        else:
            image_tensor = 0

        if 'report' in self.input:
            report = self.dataset[index]['report']
            caption = self.clean_report(report)
        else:
            caption = 0

        if 'biomarker' in self.input:
            biomarkers = self.dataset[index]['biomarker']
            test_inputs = []
            missing_mask = []
            for target in self.target_biomarkers:
                is_missing = False
                if biomarkers[target]['value'] is None:
                    test_inputs.append(self.biomarker_v_null)
                    is_missing = True
                elif self.biomarker_discrete:
                    if biomarkers[target]['prompt'] == 'normal':
                        test_inputs.append(0)
                    elif biomarkers[target]['prompt'] == 'high':
                        test_inputs.append(1)
                    elif biomarkers[target]['prompt'] == 'low':
                        test_inputs.append(-1)
                    elif biomarkers[target]['prompt'] is None:
                        test_inputs.append(biomarkers[target]['value'])
                    else:
                        raise ValueError
                elif self.biomarker_normalize:
                    if biomarkers[target]['normal_range'][0] is None or biomarkers[target]['normal_range'][1] is None or \
                            biomarkers[target]['normal_range'][1] == biomarkers[target]['normal_range'][0]:
                        test_inputs.append(biomarkers[target]['value'])
                    else:
                        norm_value = (biomarkers[target]['value'] - biomarkers[target]['normal_range'][0]) / (
                                biomarkers[target]['normal_range'][1] - biomarkers[target]['normal_range'][0])
                        test_inputs.append(norm_value)
                else:
                    test_inputs.append(biomarkers[target]['value'])
                missing_mask.append(is_missing)
            test_inputs = torch.Tensor(test_inputs)
            missing_mask = torch.Tensor(missing_mask).float()
        else:
            test_inputs = 0
            missing_mask = 0

        return {'image': image_tensor, 'caption': caption, 'biomarker': test_inputs, 'biomarker_missing_mask': missing_mask,
                'one_hot_label': one_hot_label, 'name': data_id, 'image_path': img_path}


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

    def onehot_to_medicine(self, idx_list):
        """idx_list: [0, 1, 1, 0, ...]"""
        m_list = []
        for idx, v in enumerate(idx_list):
            if v > 0:
                m_list.append(self.medicine_ids[idx])
        return m_list

