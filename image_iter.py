#!/usr/bin/env python
# encoding: utf-8
"""
@author: yaoyaozhong, hongbozhao
@contact: zhongyaoyao@bupt.edu.cn
@file: image_iter_yy.py
@time: 2020/06/03, 2025/7/14
@desc: training dataset loader for .rec
"""

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
from torch.utils.data import Subset
from PIL import Image

logger = logging.getLogger()

from IPython import embed


class FaceDataset(data.Dataset):

    def __init__(self, path_imgrec, rand_mirror):
        self.rand_mirror = rand_mirror
        assert path_imgrec
        if path_imgrec:
            logging.info("loading recordio %s...", path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            print(path_imgrec, path_imgidx)
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag > 0:
                print("header0 label", header.label)
                self.header0 = (int(header.label[0]), int(header.label[1]))
                # assert(header.flag==1)
                # self.imgidx = range(1, int(header.label[0]))
                self.imgidx = []
                self.id2range = {}
                self.seq_identity = range(int(header.label[0]), int(header.label[1]))
                for identity in self.seq_identity:
                    s = self.imgrec.read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a, b = int(header.label[0]), int(header.label[1])
                    count = b - a
                    self.id2range[identity] = (a, b)
                    self.imgidx += range(a, b)
                print("id2range", len(self.id2range))
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


class CLDatasetWrapper(Dataset):
    """
    modify the label of the original dataset to make it different from the original label
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        image, label = self.original_dataset[index]

        # Modify the label
        modified_label = self.modify_label(label)

        return image, modified_label

    def __len__(self):
        return len(self.original_dataset)

    def modify_label(self, label):
        # Modify the label to make it different from the original label
        # Randomly generate a positive integer
        random_int = random.randint(1, 100)
        modified_label = (
            label + random_int
        )  # You can define your own modification rule here

        # Ensure the modified label is not equal to the original label
        modified_label = modified_label % len(self.original_dataset.classes)
        if modified_label == label:
            modified_label = (label + 1) % len(self.original_dataset.classes)

        return modified_label


class CustomSubset(Subset):
    """A custom subset class"""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets  # keep the targets attribute
        self.classes = dataset.classes  # keep the classes attribute

    def __getitem__(self, idx):  # support the indexing such as dataset[0]
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self):  # support the len() function
        return len(self.indices)


class ImageNet900Dataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # 下面这两行是 ImageNet 预处理的标准 normalize
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, duplication_factor=20, transform=None):
        """
        Args:
            base_dataset (Dataset): The few-shot dataset.
            duplication_factor (int): Number of times to replicate each sample.
            transform (callable, optional): Transform to apply to each duplicated image.
        """
        self.base_dataset = base_dataset
        self.duplication_factor = duplication_factor
        self.transform = transform

        self.expanded_indices = [
            idx for idx in range(len(base_dataset)) for _ in range(duplication_factor)
        ]

    def __getitem__(self, idx):
        base_idx = self.expanded_indices[idx]
        image, label = self.base_dataset[base_idx]
        # import pdb; pdb.set_trace()
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.expanded_indices)
    

class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    root = "./data/faces_webface_112x112/train.rec"
    embed()
    dataset = FaceDataset(path_imgrec=root, rand_mirror=False)
    trainloader = data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False
    )
    print(len(dataset))  # 490623
    for data, label in trainloader:
        print(data.shape, label)  # torch.Size([32, 3, 112, 112]) torch.Size([32])
