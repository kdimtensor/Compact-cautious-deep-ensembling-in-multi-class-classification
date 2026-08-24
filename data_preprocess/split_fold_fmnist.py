import os
import numpy as np
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold

from fmnist_c import FMNISTC

# >>> Get severity 6 >>>

# fmnist original dataset:
# train: 60,000
# test: 10,000

# # ###

if __name__ == "__main__":

    root = "../"
    os.makedirs(root + 'data/FMNIST-10-C/gaussian_noise_6/train_clean', exist_ok=True)
    os.makedirs(root + 'data/FMNIST-10-C/gaussian_noise_6/train', exist_ok=True)
    os.makedirs(root + 'data/FMNIST-10-C/gaussian_noise_6/test_clean', exist_ok=True)
    os.makedirs(root + 'data/FMNIST-10-C/gaussian_noise_6/test', exist_ok=True)
    os.makedirs(root + 'data/FMNIST-10-C/gaussian_noise_6/label', exist_ok=True)
    # ####

    data_clean_train = np.load(root + 'data/FMNIST-10-C/gaussian_noise_1_6/gaussian_noise_clean_train.npy')
    data_c_train = np.load(root + 'data/FMNIST-10-C/gaussian_noise_1_6/gaussian_noise_c_train.npy')
    data_clean_test = np.load(root + 'data/FMNIST-10-C/gaussian_noise_1_6/gaussian_noise_clean_test.npy')
    data_c_test = np.load(root + 'data/FMNIST-10-C/gaussian_noise_1_6/gaussian_noise_c_test.npy')
    label_train = np.load(root +'data/FMNIST-10-C/gaussian_noise_1_6/label_train.npy')
    label_test = np.load(root + 'data/FMNIST-10-C/gaussian_noise_1_6/label_test.npy')


    data_clean_train_6  = data_clean_train[240000:]
    data_c_train_6      = data_c_train[240000:]
    data_clean_test_6   = data_clean_test[40000:]
    data_c_test_6       = data_c_test[40000:]
    label_train_6       = label_train[240000:]
    label_test_6        = label_test[40000:]

    print(len(data_clean_train))
    print(len(data_c_train))
    print(len(data_clean_test))
    print(len(data_c_test))
    print(len(label_train))
    print(len(label_test))
    print('-'*11)
    print(data_clean_train_6.shape)
    print(data_c_train_6.shape)
    print(data_clean_test_6.shape)
    print(data_c_test_6.shape )
    print(label_train_6.shape     )
    print(label_test_6.shape     )

    np.save(root + 'data/FMNIST-10-C/gaussian_noise_6/train_clean/gaussian_noise_train_clean.npy', np.array(data_clean_train_6).astype(np.uint8))
    np.save(root + 'data/FMNIST-10-C/gaussian_noise_6/train/gaussian_noise_train.npy', np.array(data_c_train_6).astype(np.uint8))
    np.save(root + 'data/FMNIST-10-C/gaussian_noise_6/test_clean/gaussian_noise_test_clean.npy', np.array(data_clean_test_6).astype(np.uint8))
    np.save(root + 'data/FMNIST-10-C/gaussian_noise_6/test/gaussian_noise_test.npy', np.array(data_c_test_6).astype(np.uint8))
    np.save(root + 'data/FMNIST-10-C/gaussian_noise_6/label/label_train.npy', np.array(label_train_6).astype(np.uint8))
    np.save(root +'data/FMNIST-10-C/gaussian_noise_6/label/label_test.npy', np.array(label_test_6).astype(np.uint8))

    # sys.exit()

    # <<< Get severity 6 <<<

    # >>> Split folds >>>

    k_folds = 3

    train_dataset = FMNISTC(
            root=root +'data/FMNIST-10-C/gaussian_noise_6/',
            train=True,
            download=False)

    test_dataset = FMNISTC(
        root=root +'data/FMNIST-10-C/gaussian_noise_6/',
        train=False,
        download=False)

    dataset_all = ConcatDataset([train_dataset, test_dataset])

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state = 42)

    os.makedirs(root + f'data/FMNIST-10-C/gaussian_clean_fold', exist_ok=True)
    os.makedirs(root + f'data/FMNIST-10-C/gaussian_noise_fold', exist_ok=True)
    os.makedirs(root + f'data/FMNIST-10-C/k_fold_id', exist_ok=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_all)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        np.save(root + f'data/FMNIST-10-C/k_fold_id/train_ids_fold_{fold}.npy', train_ids)
        np.save(root + f'data/FMNIST-10-C/k_fold_id/test_ids_fold_{fold}.npy', test_ids)
