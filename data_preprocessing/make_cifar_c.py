# -*- coding: utf-8 -*-

import os
import torchvision.datasets as dset
import torchvision.transforms as trn
import numpy as np

# /////////////// Distortion Helpers ///////////////

import warnings
import collections

warnings.simplefilter("ignore", UserWarning)

# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

# /////////////// End Distortions ///////////////

def main():
    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise

    #############
    data_path = '../data/CIFAR-10-C/gaussian_noise_1_6/'
    os.makedirs(data_path, exist_ok=True)
    ####
    if os.path.isdir("../data/cifar-10-batches-py/"):
        download = False
    else:
        download = True
    # # Only train data
    data_clean = dset.CIFAR10('../data', train=True, download = download)

    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

    for method_name in d.keys():
        print('Creating train images for the corruption', method_name)
        cifar_clean, cifar_c, labels = [], [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(data_clean.data, data_clean.targets):
                labels.append(label)
                cifar_clean.append(np.uint8(img))
                cifar_c.append(np.uint8(corruption(convert_img(img))))

        np.save(data_path + d[method_name].__name__ + '_clean_train.npy', np.array(cifar_clean).astype(np.uint8))
        np.save(data_path + d[method_name].__name__ + '_c_train.npy', np.array(cifar_c).astype(np.uint8))

        # This replaces not appends
        np.save(data_path + 'label_train.npy',
                np.array(labels).astype(np.uint8))

    # Only test data
    data_clean = dset.CIFAR10('../data', train=False, download = download)
    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

    for method_name in d.keys():
        print('Creating test images for the corruption', method_name)
        cifar_clean, cifar_c, labels = [], [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(data_clean.data, data_clean.targets):
                labels.append(label)
                cifar_clean.append(np.uint8(img))
                cifar_c.append(np.uint8(corruption(convert_img(img))))

        np.save(data_path + d[method_name].__name__ + '_clean_test.npy', np.array(cifar_clean).astype(np.uint8))
        np.save(data_path + d[method_name].__name__ + '_c_test.npy', np.array(cifar_c).astype(np.uint8))

        # This replaces not appends
        np.save(data_path +'label_test.npy',
                np.array(labels).astype(np.uint8))

if __name__ == "__main__":
    
    print('Using CIFAR-10 data')
    main()