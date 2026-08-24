import os

import numpy as np
from sklearn.model_selection import KFold

from leaf_c import LEAFC
from collections import Counter

if __name__ == "__main__":

    #############
    root = "../"
    data_name = "LEAF-C"
    #############
    ####

    os.makedirs(root + f'data/{data_name}/gaussian_noise_6/train_clean', exist_ok=True)
    os.makedirs(root + f'data/{data_name}/gaussian_noise_6/train', exist_ok=True)
    os.makedirs(root + f'data/{data_name}/gaussian_noise_6/label', exist_ok=True)
    ####
    # >>> Get severity 6 >>>

    data_clean_train = np.load(root + f'data/{data_name}/gaussian_noise_1_6/gaussian_noise_clean_train.npy')
    data_c_train = np.load(root + f'data/{data_name}/gaussian_noise_1_6/gaussian_noise_c_train.npy')
    label_train = np.load(root + f'data/{data_name}/gaussian_noise_1_6/label_train.npy')

    print(len(data_clean_train))
    print(len(data_c_train))
    print(len(label_train))
    print('-'*8)

    np.save(root + f'data/{data_name}/gaussian_noise_6/train_clean/gaussian_noise_train_clean.npy', np.array(data_clean_train_6).astype(np.uint8))

    np.save(root+ f'data/{data_name}/gaussian_noise_6/train/gaussian_noise_train.npy', np.array(data_c_train_6).astype(np.uint8))

    np.save(root + f'data/{data_name}/gaussian_noise_6/label/label_train.npy',
            np.array(label_train_6).astype(np.uint8))

    # <<< Get severity 6 <<<

    # >>> Split folds >>>
    ########### no need split noise and clean
    k_folds = 3
    train_dataset = LEAFC(
            root= root + f'data/{data_name}/gaussian_noise_6/',
            train=True,
            )

    dataset_all = train_dataset

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state = 42)

    os.makedirs(root + f'data/{data_name}/gaussian_clean_fold', exist_ok=True)
    os.makedirs(root + f'data/{data_name}/gaussian_noise_fold', exist_ok=True)
    os.makedirs(root + f'data/{data_name}/k_fold_id', exist_ok=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_all)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        np.save(root + f'data/{data_name}/k_fold_id/train_ids_fold_{fold}.npy', train_ids)
        np.save(root + f'data/{data_name}/k_fold_id/test_ids_fold_{fold}.npy', test_ids)
