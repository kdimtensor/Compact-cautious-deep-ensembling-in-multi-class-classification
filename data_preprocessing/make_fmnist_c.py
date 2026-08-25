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

    ####
    folder_path =  '../data/FMNIST-10-C/gaussian_noise_1_6/'
    os.makedirs(folder_path, exist_ok=True)

    if os.path.isdir("../data/FashionMNIST/"):
        download = False
        print("existing data")
    else:
        download = True
        print("Downloading data")
    ####

    # Only test data
    
    data_clean = dset.FashionMNIST('../data', train=False, download=download)

    # Convert data_clean.data and data_clean.targets
    # from torch.tensor to numpy and list
    # The convert function loves numpy, not torch tensor
    data_clean_data_numpy = data_clean.data.numpy()
    data_clean_targets_list = data_clean.targets.tolist()

    idx_to_label = {i: label for i, label in enumerate(data_clean.classes)}

    with open(folder_path + "label_list.txt", "w") as f:

        for key, value in idx_to_label.items():
            f.write(f"{key}: {value}\n")

    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

    for method_name in d.keys():
        print('Creating test images for the corruption', method_name)
        data_clean_list, data_c, labels = [], [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(data_clean_data_numpy, data_clean_targets_list):

                labels.append(label)
                data_clean_list.append(np.uint8(img))
                data_c.append(np.uint8(corruption(convert_img(img))))

        np.save(folder_path + d[method_name].__name__ + '_clean_test.npy', np.array(data_clean_list).astype(np.uint8))
        np.save(folder_path + d[method_name].__name__ + '_c_test.npy', np.array(data_c).astype(np.uint8))

        # This replaces not appends
        np.save(folder_path + 'label_test.npy',
                np.array(labels).astype(np.uint8))


    # # # Only train data

    data_clean = dset.FashionMNIST('../data', train=True, download = download)

    data_clean_data_numpy = data_clean.data.numpy()
    data_clean_targets_list = data_clean.targets.tolist()

    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

    for method_name in d.keys():
        print('Creating train images for the corruption', method_name)
        data_clean_list, data_c, labels = [], [], []

        for severity in range(1,6):
            corruption = lambda clean_img: d[method_name](clean_img, severity)

            for img, label in zip(data_clean_data_numpy, data_clean_targets_list):

                labels.append(label)
                data_clean_list.append(np.uint8(img))
                data_c.append(np.uint8(corruption(convert_img(img))))

        np.save(folder_path + d[method_name].__name__ + '_clean_train.npy', np.array(data_clean_list).astype(np.uint8))
        np.save(folder_path + d[method_name].__name__ + '_c_train.npy', np.array(data_c).astype(np.uint8))

        # This replaces not appends
        np.save(folder_path + 'label_train.npy',
            np.array(labels).astype(np.uint8))

if __name__ == "__main__":
    
    print('Using f-minst data')
    main()