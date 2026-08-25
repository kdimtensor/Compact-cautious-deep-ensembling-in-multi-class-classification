# -*- coding: utf-8 -*-

import os
import torchvision.transforms as trn
import numpy as np

# /////////////// Distortion Helpers ///////////////

from torchvision import datasets
from tqdm import tqdm
import warnings
import collections
from PIL import Image
warnings.simplefilter("ignore", UserWarning)

# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

# /////////////// End Distortions ///////////////

#########################

def main():

    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise

    ####
    data_name = "LEAF"
    print(f'Using {data_name} data')

    folder_path = f'../data/{data_name}-C/gaussian_noise_1_6/'
    os.makedirs(folder_path, exist_ok=True)

    if os.path.isdir("../data/LEAF/"):
        print("existing data")
    else:
        print("Need to download data first")
        import sys
        sys.exit()

    dataset = datasets.ImageFolder(root=f'../data/{data_name}/')
    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])
    print(list(dataset.class_to_idx))

    with open(folder_path+"output.txt", "w") as f:
        for item in dataset.class_to_idx:
            f.write(str(item) + "\n")


    data = []
    targets = []

    for img, label in tqdm(dataset):

        img = img.resize((224, 224), Image.LANCZOS)

        data.append(img)     
        targets.append(label)  # label is int

    dataset.data = data
    dataset.targets = targets


    for method_name in d.keys():
        print('Creating images for the corruption', method_name)
        data_clean, data_c, labels = [], [], []

        for severity in range(5,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in tqdm(zip(dataset.data, dataset.targets)):
                labels.append(label)
                a = np.uint8(img)
                data_clean.append(a)
                data_c.append(np.uint8(corruption(convert_img(img))))


        np.save(folder_path+ d[method_name].__name__ + '_clean_train.npy', np.array(data_clean).astype(np.uint8))
        np.save(folder_path + d[method_name].__name__ + '_c_train.npy', np.array(data_c).astype(np.uint8))

        # This replaces not appends
        np.save(folder_path + 'label_train.npy',
                np.array(labels).astype(np.uint8))

if __name__ == "__main__":
    
    main()