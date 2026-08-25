import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset


class LEAFC(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set  

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



class LEAFCLEAN(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set     

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
