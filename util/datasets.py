# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyhdf.SD import SD, SDC

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def hdf_aod_loader(path: str, code: str, order: str) -> Any:
    # preprocessing the aod value
    hdf_file = SD(path, SDC.READ)
    data = hdf_file.select(f'Optical_Depth_{code}')[:][order]
    data = data.astype(np.float16)
    data[data == -28672] = np.nan
    # data = data * 0.001
    data = data[np.newaxis, ...]
    return torch.from_numpy(data)

class AODDataset(Dataset):
    """Dataset type for MODIS AOD MCD19A2"""
    
    def __init__(
        self, 
        root: str, 
        table_file: str, 
        aod_code: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = hdf_aod_loader,
        ):
        """
        Args:
            root (string):Root directory path.
            transform (callable, optional): Optional transform to be applied on samples.
        """
        # super().__init__(root, transform=transform, target_transform=target_transform)        
        # for backwards-compatibility
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        data_df = pd.read_csv(table_file, index_col=0)
        samples = self.make_dataset(self.root, data_df)
        
        self.root = root
        self.code = aod_code
        self.loader = loader
        
        self.samples = samples
        self.targets = [s[2] for s in samples]
    
    @staticmethod
    def make_dataset(directory: str, data_df:pd.DataFrame):
        """Generate a list of smaples of a form (path_to_sample, class)

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            table_file (DataFrame): cover ratio data table for files
        """
        samples = []
        for i, row in data_df.iterrows():
            path = os.path.join(directory, row['File Name'])
            samples.append((path, row['Order'], row['Ratio']))

        return samples
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            Tuple[Any, Any]: (sample, target) where target is class_index of the target class.
        """
        path, order, target = self.samples[index]
        sample = self.loader(path, self.code, order)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def get_aod_data(file_path, code='055'):
    # preprocessing the aod value
    hdf_file = SD(file_path, SDC.READ)
    data = hdf_file.select(f'Optical_Depth_{code}')[:]
    data = data.astype(np.float32)
    data[data == -28672] = np.nan
    data = data * 0.001
    return data
