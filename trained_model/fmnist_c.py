import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import numpy as np
import torch
from PIL import Image

from torchvision.datasets.vision import VisionDataset


class FMNISTC(VisionDataset):
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]


    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if self.train:
            self.new_root = os.path.join(self.root, 'train')
            downloaded_list = os.listdir(self.new_root)
            self.target_data = np.load(os.path.join(self.root, 'label', 'label_train.npy')).tolist()
        else:
            self.new_root = os.path.join(self.root, 'test')
            downloaded_list = os.listdir(self.new_root)
            self.target_data = np.load(os.path.join(self.root, 'label', 'label_test.npy')).tolist()

        self.data: Any = []
        self.targets = []

        # now load the numpy arrays
        # data shape:
        # <class 'numpy.ndarray'>
        # (50000, 32, 32, 3)
        # label shape:
        # <class 'list'>
        # 50000
        for file_name in downloaded_list:
            file_path = os.path.join(self.new_root, file_name)
            # self.data.append(np.load(file_path))
            self.data.extend(np.load(file_path))
            self.targets.extend(self.target_data)
        
        # self.data = np.vstack(self.data) 
        self.data = np.asarray(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)



class FMNISTCLEAN(VisionDataset):
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]


    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if self.train:
            self.new_root = os.path.join(self.root, 'train_clean')
            downloaded_list = os.listdir(self.new_root)
            self.target_data = np.load(os.path.join(self.root, 'label', 'label_train.npy')).tolist()
        else:
            self.new_root = os.path.join(self.root, 'test_clean')
            downloaded_list = os.listdir(self.new_root)
            self.target_data = np.load(os.path.join(self.root, 'label', 'label_test.npy')).tolist()

        self.data: Any = []
        self.targets = []

        # now load the numpy arrays
        # data shape:
        # <class 'numpy.ndarray'>
        # (50000, 32, 32, 3)
        # label shape:
        # <class 'list'>
        # 50000
        for file_name in downloaded_list:
            file_path = os.path.join(self.new_root, file_name)
            # self.data.append(np.load(file_path))
            self.data.extend(np.load(file_path))
            self.targets.extend(self.target_data)
        
        # self.data = np.vstack(self.data) 
        self.data = np.asarray(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


# class FashionMNIST(MNIST):
#     """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

#     Args:
#         root (str or ``pathlib.Path``): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
#             and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
#         train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
#             otherwise from ``t10k-images-idx3-ubyte``.
#         download (bool, optional): If True, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#         transform (callable, optional): A function/transform that  takes in a PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#     """

#     mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

#     resources = [
#         ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
#         ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
#         ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
#         ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
#     ]
#     classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

