
import sys
import os
import json

import numpy as np
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE


# >>> CALIBRATION ERRORS >>>
#########
root_dir = ""
#########
data = "LEAF"
model_type = "Drop_out"
train_type = "noise"

k_folds = 3
# [[SED_c L1D_c KLD_c SED_clean L1D_clean KLD_clean], [fold 2], [fold 3]]
kfold_ece_class = []
kfold_mce_class = []
kfold_ece_confi = []
kfold_mce_confi = []

save_dir = root_dir + f'/data/{data}-C/gaussian_{train_type}_fold/{model_type}/'
out_dir =  root_dir + f'/Compact-credal-deep-ensembling-in-multi-class-classification/Visualize_output/{data}-C/gaussian_{train_type}_fold/{model_type}/set_output/'
for fold in range(k_folds):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # [SED_c L1D_c KLD_c SED_clean L1D_clean KLD_clean]
    ece_class = []
    mce_class = []
    ece_confi = []
    mce_confi = []
    
    save_file = f'{out_dir}test_calibration_result.txt'
    print(save_file)
    # if not os.path.exists(f'{save_dir}calibra/'):
    #     print(f"Path does not exist. Creating: {f'{save_dir}/calibra/'}")
    #     os.makedirs(f'{save_dir}calibra')

    # if not os.path.exists(save_file):
    #     print('File not exists.')
    # #     sys.exit()

    ########
    all_p_star_SED_c =  np.load(f'{save_dir}all_p_star_SED_c_fold_{fold}.npy')
    all_p_star_L1D_c =  np.load(f'{save_dir}all_p_star_L1_c_fold_{fold}.npy')
    all_p_star_KLD_c =  np.load(f'{save_dir}all_p_star_KLD_c_fold_{fold}.npy')

    predictions_SED_c = np.load(f'{save_dir}predictions_SED_sensitive_c_fold_{fold}.npy')
    predictions_L1D_c = np.load(f'{save_dir}predictions_L1_sensitive_c_fold_{fold}.npy')
    predictions_KLD_c = np.load(f'{save_dir}predictions_KLD_sensitive_c_fold_{fold}.npy')

    all_p_star_SED_clean =  np.load(f'{save_dir}all_p_star_SED_clean_fold_{fold}.npy')
    all_p_star_L1D_clean =  np.load(f'{save_dir}all_p_star_L1_clean_fold_{fold}.npy')
    all_p_star_KLD_clean =  np.load(f'{save_dir}all_p_star_KLD_clean_fold_{fold}.npy')

    predictions_SED_clean = np.load(f'{save_dir}predictions_SED_sensitive_clean_fold_{fold}.npy')
    predictions_L1D_clean = np.load(f'{save_dir}predictions_L1_sensitive_clean_fold_{fold}.npy')
    predictions_KLD_clean = np.load(f'{save_dir}predictions_KLD_sensitive_clean_fold_{fold}.npy')
    
    
    # all_p_star_SED_c =  np.load(f'{save_dir}cala_no_reward/all_p_star_SED_c_fold_{fold}.npy')
    # all_p_star_L1D_c =  np.load(f'{save_dir}cala_no_reward/all_p_star_L1_c_fold_{fold}.npy')
    # all_p_star_KLD_c =  np.load(f'{save_dir}cala_no_reward/all_p_star_KLD_c_fold_{fold}.npy')

    # predictions_SED_c = np.load(f'{save_dir}cala_no_reward/predictions_SED_sensitive_c_fold_{fold}.npy')
    # predictions_L1D_c = np.load(f'{save_dir}cala_no_reward/predictions_L1_sensitive_c_fold_{fold}.npy')
    # predictions_KLD_c = np.load(f'{save_dir}cala_no_reward/predictions_KLD_sensitive_c_fold_{fold}.npy')

    # all_p_star_SED_clean =  np.load(f'{save_dir}cala_no_reward/all_p_star_SED_clean_fold_{fold}.npy')
    # all_p_star_L1D_clean =  np.load(f'{save_dir}cala_no_reward/all_p_star_L1_clean_fold_{fold}.npy')
    # all_p_star_KLD_clean =  np.load(f'{save_dir}cala_no_reward/all_p_star_KLD_clean_fold_{fold}.npy')

    # predictions_SED_clean = np.load(f'{save_dir}cala_no_reward/predictions_SED_sensitive_clean_fold_{fold}.npy')
    # predictions_L1D_clean = np.load(f'{save_dir}cala_no_reward/predictions_L1_sensitive_clean_fold_{fold}.npy')
    # predictions_KLD_clean = np.load(f'{save_dir}cala_no_reward/predictions_KLD_sensitive_clean_fold_{fold}.npy')

    labels_c = np.load(f'{save_dir}test_labels_c_fold_{fold}.npy')
    labels_clean = np.load(f'{save_dir}test_labels_clean_fold_{fold}.npy')
    labels_onehot_c = np.load(f'{save_dir}test_labels_onehot_c_fold_{fold}.npy')
    labels_onehot_clean = np.load(f'{save_dir}test_labels_onehot_clean_fold_{fold}.npy')

    print('acc_SED_c:', (predictions_SED_c == labels_c).mean() * 100)
    print('acc_L1D_c:',  (predictions_L1D_c == labels_c).mean() * 100)
    print('acc_KLD_c:', (predictions_KLD_c == labels_c).mean() * 100)

    print('acc_SED_clean:', (predictions_SED_clean == labels_clean).mean() * 100)
    print('acc_L1D_clean:',  (predictions_L1D_clean == labels_clean).mean() * 100)
    print('acc_KLD_clean:', (predictions_KLD_clean == labels_clean).mean() * 100)

    # print('binary_ECE_SED:', binary_ECE(labels_onehot, all_p_star_SED, bins=30))
    # print('binary_ECE_L1:', binary_ECE(labels_onehot, all_p_star_L1, bins=30))
    # print('binary_ECE_KLD:', binary_ECE(labels_onehot, all_p_star_KLD, bins=30))

    # print('binary_MCE_SED:', binary_MCE(labels_onehot, all_p_star_SED, bins=30))
    # print('binary_MCE_L1:', binary_MCE(labels_onehot, all_p_star_L1, bins=30))
    # print('binary_MCE_KLD:', binary_MCE(labels_onehot, all_p_star_KLD, bins=30))

    # WARNING!!! 
    # Inputs for calibrations functions in the shape [n_instance, n_class], [10000, 10]
    # The pycalib API starts with shape [n_class, n_instance] and use transpose [].T
    # print('classwise_ECE_SED_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
    # print('classwise_MCE_SED_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
    # print('conf_ECE_SED_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
    # print('conf_MCE_SED_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))  

    # print('-'*11)

    # print('classwise_ECE_L1_c:',  np.round(classwise_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
    # print('classwise_MCE_L1_c:',  np.round(classwise_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
    # print('conf_ECE_L1_c:     ',  np.round(conf_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
    # print('conf_MCE_L1_c:     ',  np.round(conf_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))    

    # print('-'*11)

    # print('classwise_ECE_KLD_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
    # print('classwise_MCE_KLD_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
    # print('conf_ECE_KLD_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
    # print('conf_MCE_KLD_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))    

    # print('-'*22)

    # print('classwise_ECE_SED_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
    # print('classwise_MCE_SED_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
    # print('conf_ECE_SED_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
    # print('conf_MCE_SED_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))    

    # print('-'*11)

    # print('classwise_ECE_L1_clean:',  np.round(classwise_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
    # print('classwise_MCE_L1_clean:',  np.round(classwise_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
    # print('conf_ECE_L1_clean:     ',  np.round(conf_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
    # print('conf_MCE_L1_clean:     ',  np.round(conf_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))    

    # print('-'*11)

    # print('classwise_ECE_KLD_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
    # print('classwise_MCE_KLD_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
    # print('conf_ECE_KLD_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
    # print('conf_MCE_KLD_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))

    ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    
    ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_L1D_c, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_L1D_c, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_L1D_c, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_L1D_c, bins=30)*100)
    
    ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    
    ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    
    ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_L1D_clean, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_L1D_clean, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_L1D_clean, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_L1D_clean, bins=30)*100)
    
    ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
    mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
    ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
    mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)

    kfold_ece_class.append(ece_class)
    kfold_mce_class.append(mce_class)
    kfold_ece_confi.append(ece_confi)
    kfold_mce_confi.append(mce_confi)

print(np.array(kfold_ece_class))
print(np.array(kfold_mce_class))
print(np.array(kfold_ece_confi))
print(np.array(kfold_mce_confi))

ece_class_mean = np.round(np.mean(np.array(kfold_ece_class), axis=0), 2)
mce_class_mean = np.round(np.mean(np.array(kfold_mce_class), axis=0), 2)
ece_confi_mean = np.round(np.mean(np.array(kfold_ece_confi), axis=0), 2)
mce_confi_mean = np.round(np.mean(np.array(kfold_mce_confi), axis=0), 2)

ece_class_std = np.round(np.std(np.array(kfold_ece_class), axis=0), 2)
mce_class_std = np.round(np.std(np.array(kfold_mce_class), axis=0), 2)
ece_confi_std = np.round(np.std(np.array(kfold_ece_confi), axis=0), 2)
mce_confi_std = np.round(np.std(np.array(kfold_mce_confi), axis=0), 2)

with open(save_file, "w") as f:
    f.write('(=.=)'*11)
    f.write('\nECE Classwise')
    f.write('\nMCE Classwise')
    f.write('\nECE Confidence')
    f.write('\nMCE Confidence')
    f.write('\n[[SED_c L1D_c KLD_c SED_clean L1D_clean KLD_clean], [fold 2], [fold 3]]')
    f.write('\n' + '(=.=)'*11)
    f.write(f'\n{kfold_ece_class}')
    f.write(f'\n{kfold_mce_class}')
    f.write(f'\n{kfold_ece_confi}')
    f.write(f'\n{kfold_mce_confi}')
    f.write('\n' + '(=.=)'*11)
    f.write(f'\nece_class_mean: {ece_class_mean}')
    f.write(f'\nmce_class_mean: {mce_class_mean}')
    f.write(f'\nece_confi_mean: {ece_confi_mean}')
    f.write(f'\nmce_confi_mean: {mce_confi_mean}')
    f.write(f'\nece_class_std: {ece_class_std}')
    f.write(f'\nmce_class_std: {mce_class_std}')
    f.write(f'\nece_confi_std: {ece_confi_std}')
    f.write(f'\nmce_confi_std: {mce_confi_std}')

# sys.exit()

# <<< CALIBRATION ERRORS <<<
