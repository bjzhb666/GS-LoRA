#!/usr/bin/env python
# encoding: utf-8
'''
@author: yaoyaozhong
@contact: zhongyaoyao@bupt.edu.cn
@file: image_iter_yy.py
@time: 2020/06/03
@desc: training dataset loader for .rec
'''

import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import logging
import numbers
import random
from torch.utils.data import Dataset

logger = logging.getLogger()

from IPython import embed


class FaceDataset(data.Dataset):

    def __init__(self, path_imgrec, rand_mirror):
        self.rand_mirror = rand_mirror
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...', path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec,
                                                     'r')
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print('header0 label', header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]),
                                          int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print('id2range', len(self.id2range))
            else:
                self.imgidx = list(self.imgrec.keys)
            self.seq = self.imgidx

    def __getitem__(self, index):
        idx = self.seq[index]
        s = self.imgrec.read_idx(idx)
        header, s = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        _data = mx.image.imdecode(s)
        if self.rand_mirror:
            _rd = random.randint(0, 1)
            if _rd == 1:
                _data = mx.ndarray.flip(data=_data, axis=1)

        _data = nd.transpose(_data, axes=(2, 0, 1))
        _data = _data.asnumpy()
        img = torch.from_numpy(_data)

        return img, label

    def __len__(self):
        return len(self.seq)


class Wrapper10Dataset(Dataset):
    '''
    该数据集类用于将原始数据集的样本数量减少到原来的10%
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = self._count_classes()
        self.class_counts = self._count_per_class()
        self.new_length = self._calculate_new_length()

        # 使用缓存存储类别样本列表
        self.class_samples_cache = {}

    def _count_classes(self):
        classes = set(label for _, label in self.dataset)
        return len(classes)

    def _count_per_class(self):
        class_counts = {}
        for _, label in self.dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
        return list(class_counts.values())

    def _calculate_new_length(self):
        return sum(int(count * 0.1) for count in self.class_counts)

    def __len__(self):
        return self.new_length

    def __getitem__(self, index):
        target_count = int(index / 0.1)
        selected_class = None
        for class_index, count in enumerate(self.class_counts):
            if target_count < count:
                selected_class = class_index
                break
            target_count -= count

        # 检查缓存中是否有该类别的样本列表
        if selected_class in self.class_samples_cache:
            class_samples = self.class_samples_cache[selected_class]
        else:
            # 如果缓存中没有，则遍历数据集，获取该类别的样本列表
            class_samples = []
            for image, label in self.dataset:
                if label == selected_class:
                    class_samples.append((image, label))
            # 将样本列表存入缓存
            self.class_samples_cache[selected_class] = class_samples

        return class_samples[target_count]


if __name__ == '__main__':
    root = './data/faces_webface_112x112/train.rec'
    embed()
    dataset = FaceDataset(path_imgrec=root, rand_mirror=False)
    trainloader = data.DataLoader(dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=False)
    print(len(dataset))  # 490623
    for data, label in trainloader:
        print(data.shape,
              label)  # torch.Size([32, 3, 112, 112]) torch.Size([32])
