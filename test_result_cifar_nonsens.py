import sys
import os
import json

import numpy as np

from scipy.optimize import minimize
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE

k_folds = 3

# >>> PRECISE PREDICTIONS >>>

def evaluate(output, labels, n_class, save_dir, fold, test_type):        
    all_p_star_L1 = np.zeros([len(labels), n_class])
    all_p_star_KLD = np.zeros([len(labels), n_class])
    eval_results = []

    # >>> Squared Euclidean distance >>>

    # pred_mean shape ([10000, 10])
    # all_p_star_SED = pred_mean
    all_p_star_SED = np.mean(output, axis=0)    

    # <<< Squared Euclidean distance <<<

    for i in range(len(labels)):
        probability_set = output[:,i,:] #(100,10)

        # >>> L1_distance >>>

        def L1_distance(p_star_L1, probability_set):
            return np.sum(np.abs(probability_set - p_star_L1))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

        result = minimize(L1_distance, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)    
        p_star_L1 = result.x

        # all_p_star_L1 shape [10000, 10]
        all_p_star_L1[i] = p_star_L1

        # <<< L1_distance <<<

        # >>> KL_divergence >>>

        def KL_divergence(p_star_KLD, probability_set):
            epsilon = 1e-10
            p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)
            probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
            return np.sum(p_star_KLD * np.log(p_star_KLD/probability_set))
            # return np.sum(probability_set * np.log(probability_set/p_star_KLD))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                {'type': 'ineq', 'fun': lambda x: x}]
        
        result = minimize(KL_divergence, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)
        p_star_KLD = result.x
        epsilon = 1e-10
        p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)

        # all_p_star_KLD shape [10000, 10]
        all_p_star_KLD[i] = p_star_KLD

        # <<< KL_divergence <<<         
    
    predictions_SED = np.argmax(all_p_star_SED, axis=1)
    acc_SED = (predictions_SED == labels).mean() * 100    
    
    predictions_L1 = np.argmax(all_p_star_L1, axis=1)
    acc_L1 = (predictions_L1 == labels).mean() * 100
    
    predictions_KLD = np.argmax(all_p_star_KLD, axis=1)    
    acc_KLD = (predictions_KLD == labels).mean() * 100
    
    eval_results.append(acc_SED)
    eval_results.append(acc_L1)
    eval_results.append(acc_KLD)

    # >>> Save test outputs >>>

    np.save(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy', all_p_star_SED)
    np.save(f'{save_dir}predictions_SED_{test_type}_fold_{fold}.npy', predictions_SED)

    np.save(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy', all_p_star_L1)
    np.save(f'{save_dir}predictions_L1_{test_type}_fold_{fold}.npy', predictions_L1)

    np.save(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy', all_p_star_KLD)
    np.save(f'{save_dir}predictions_KLD_{test_type}_fold_{fold}.npy', predictions_KLD)

    # <<< Save test output <<<

    return eval_results


kfold_results_c = []
kfold_results_clean = []

for fold in range(k_folds):
    print(f'FOLD {fold}')
    print('--------------------------------')

    save_dir = './data/CIFAR-10-C/gaussian_noise_clean_fold_nonsensitive/'
        
    # output shape ([100, 10000, 10])
    output_c = np.load(f'{save_dir}tensor_output_c_fold_{fold}.npy')
    labels_c = np.load(f'{save_dir}cifar_test_labels_c_fold_{fold}.npy')
    output_clean = np.load(f'{save_dir}tensor_output_clean_fold_{fold}.npy')
    labels_clean = np.load(f'{save_dir}cifar_test_labels_clean_fold_{fold}.npy')
    labels_onehot_c = np.load(f'{save_dir}cifar_test_labels_onehot_c_fold_{fold}.npy')
    labels_onehot_clean = np.load(f'{save_dir}cifar_test_labels_onehot_clean_fold_{fold}.npy')
    with open('./iukm/cifar_test_data_class_to_idx.txt', 'r') as f:
        class_to_idx = json.load(f)
    n_class = len(class_to_idx)
    
    kfold_results_c.append(evaluate(output_c, labels_c, n_class, save_dir, fold, 'c'))
    kfold_results_clean.append(evaluate(output_clean, labels_clean, n_class, save_dir, fold, 'clean'))

print(np.round(kfold_results_c, 2))
print('-'*8)
print(np.round(kfold_results_clean, 2))


sys.exit()

# kfold_results_c = np.array([[80.32, 80.31, 80.26],
#                             [80.14, 80.00, 80.01],
#                             [79.96, 79.88, 79.82]])

# kfold_results_clean = np.array([[78.56, 78.44, 78.46],
#                                 [78.33, 78.24, 78.24],
#                                 [78.80, 78.26, 78.64]])

# kfold_results_c_mean = np.round(np.mean(kfold_results_c, axis=0),2)
# kfold_results_c_std  = np.round(np.std(kfold_results_c, axis=0),2)

# kfold_results_clean_mean = np.round(np.mean(kfold_results_clean, axis=0),2)
# kfold_results_clean_std  = np.round(np.std(kfold_results_clean, axis=0),2)

# print(kfold_results_c_mean)
# print(kfold_results_c_std)

# print('-'*11)

# print(kfold_results_clean_mean)
# print(kfold_results_clean_std)

# sys.exit()


# <<< PRECISE PREDICTIONS <<<

# >>> CALIBRATION ERRORS >>>

# # [[SED_c L1_c KLD_c SED_clean L1_clean KLD_clean], [fold 2], [fold 3]]
# kfold_ece_class = []
# kfold_mce_class = []
# kfold_ece_confi = []
# kfold_mce_confi = []

# for fold in range(k_folds):
#     print(f'FOLD {fold}')
#     print('--------------------------------')

#     # [SED_c L1_c KLD_c SED_clean L1_clean KLD_clean]
#     ece_class = []
#     mce_class = []
#     ece_confi = []
#     mce_confi = []
    
#     # fold = 0
#     save_dir = './data/CIFAR-10-C/gaussian_noise_c_fold_nonsensitive/'

#     if not os.path.exists(f'{save_dir}calibration_error.txt'):
#         print('File not exists.')
#         sys.exit()

#     all_p_star_SED_c =  np.load(f'{save_dir}all_p_star_SED_c_fold_{fold}.npy')
#     all_p_star_L1_c =   np.load(f'{save_dir}all_p_star_L1_c_fold_{fold}.npy')
#     all_p_star_KLD_c =  np.load(f'{save_dir}all_p_star_KLD_c_fold_{fold}.npy')

#     predictions_SED_c = np.load(f'{save_dir}predictions_SED_c_fold_{fold}.npy')
#     predictions_L1_c =  np.load(f'{save_dir}predictions_L1_c_fold_{fold}.npy')
#     predictions_KLD_c = np.load(f'{save_dir}predictions_KLD_c_fold_{fold}.npy')

#     all_p_star_SED_clean =  np.load(f'{save_dir}all_p_star_SED_clean_fold_{fold}.npy')
#     all_p_star_L1_clean =   np.load(f'{save_dir}all_p_star_L1_clean_fold_{fold}.npy')
#     all_p_star_KLD_clean =  np.load(f'{save_dir}all_p_star_KLD_clean_fold_{fold}.npy')

#     predictions_SED_clean = np.load(f'{save_dir}predictions_SED_clean_fold_{fold}.npy')
#     predictions_L1_clean =  np.load(f'{save_dir}predictions_L1_clean_fold_{fold}.npy')
#     predictions_KLD_clean = np.load(f'{save_dir}predictions_KLD_clean_fold_{fold}.npy')

#     labels_c = np.load(f'{save_dir}cifar_test_labels_c_fold_{fold}.npy')
#     labels_clean = np.load(f'{save_dir}cifar_test_labels_clean_fold_{fold}.npy')
#     labels_onehot_c = np.load(f'{save_dir}cifar_test_labels_onehot_c_fold_{fold}.npy')
#     labels_onehot_clean = np.load(f'{save_dir}cifar_test_labels_onehot_clean_fold_{fold}.npy')

#     print('acc_SED_c:', (predictions_SED_c == labels_c).mean() * 100)
#     print('acc_L1_c:',  (predictions_L1_c == labels_c).mean() * 100)
#     print('acc_KLD_c:', (predictions_KLD_c == labels_c).mean() * 100)

#     print('acc_SED_clean:', (predictions_SED_clean == labels_clean).mean() * 100)
#     print('acc_L1_clean:',  (predictions_L1_clean == labels_clean).mean() * 100)
#     print('acc_KLD_clean:', (predictions_KLD_clean == labels_clean).mean() * 100)

#     # print('binary_ECE_SED:', binary_ECE(labels_onehot, all_p_star_SED, bins=30))
#     # print('binary_ECE_L1:', binary_ECE(labels_onehot, all_p_star_L1, bins=30))
#     # print('binary_ECE_KLD:', binary_ECE(labels_onehot, all_p_star_KLD, bins=30))

#     # print('binary_MCE_SED:', binary_MCE(labels_onehot, all_p_star_SED, bins=30))
#     # print('binary_MCE_L1:', binary_MCE(labels_onehot, all_p_star_L1, bins=30))
#     # print('binary_MCE_KLD:', binary_MCE(labels_onehot, all_p_star_KLD, bins=30))

#     # WARNING!!! 
#     # Inputs for calibrations functions in the shape [n_instance, n_class], [10000, 10]
#     # The pycalib API starts with shape [n_class, n_instance] and use transpose [].T
#     # print('classwise_ECE_SED_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('classwise_MCE_SED_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('conf_ECE_SED_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))
#     # print('conf_MCE_SED_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100, 2))  

#     # print('-'*11)

#     # print('classwise_ECE_L1_c:',  np.round(classwise_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('classwise_MCE_L1_c:',  np.round(classwise_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('conf_ECE_L1_c:     ',  np.round(conf_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))
#     # print('conf_MCE_L1_c:     ',  np.round(conf_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_KLD_c:', np.round(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('classwise_MCE_KLD_c:', np.round(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('conf_ECE_KLD_c:     ', np.round(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))
#     # print('conf_MCE_KLD_c:     ', np.round(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100, 2))    

#     # print('-'*22)

#     # print('classwise_ECE_SED_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('classwise_MCE_SED_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('conf_ECE_SED_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))
#     # print('conf_MCE_SED_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_L1_clean:',  np.round(classwise_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('classwise_MCE_L1_clean:',  np.round(classwise_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('conf_ECE_L1_clean:     ',  np.round(conf_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))
#     # print('conf_MCE_L1_clean:     ',  np.round(conf_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100, 2))    

#     # print('-'*11)

#     # print('classwise_ECE_KLD_clean:', np.round(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('classwise_MCE_KLD_clean:', np.round(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('conf_ECE_KLD_clean:     ', np.round(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))
#     # print('conf_MCE_KLD_clean:     ', np.round(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100, 2))

#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_SED_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_L1_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_c, all_p_star_KLD_c, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_SED_clean, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_L1_clean, bins=30)*100)
    
#     ece_class.append(classwise_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     mce_class.append(classwise_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     ece_confi.append(conf_ECE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)
#     mce_confi.append(conf_MCE(labels_onehot_clean, all_p_star_KLD_clean, bins=30)*100)

#     kfold_ece_class.append(ece_class)
#     kfold_mce_class.append(mce_class)
#     kfold_ece_confi.append(ece_confi)
#     kfold_mce_confi.append(mce_confi)

# print(np.array(kfold_ece_class))
# print(np.array(kfold_mce_class))
# print(np.array(kfold_ece_confi))
# print(np.array(kfold_mce_confi))

# ece_class_mean = np.round(np.mean(np.array(kfold_ece_class), axis=0), 2)
# mce_class_mean = np.round(np.mean(np.array(kfold_mce_class), axis=0), 2)
# ece_confi_mean = np.round(np.mean(np.array(kfold_ece_confi), axis=0), 2)
# mce_confi_mean = np.round(np.mean(np.array(kfold_mce_confi), axis=0), 2)

# ece_class_std = np.round(np.std(np.array(kfold_ece_class), axis=0), 2)
# mce_class_std = np.round(np.std(np.array(kfold_mce_class), axis=0), 2)
# ece_confi_std = np.round(np.std(np.array(kfold_ece_confi), axis=0), 2)
# mce_confi_std = np.round(np.std(np.array(kfold_mce_confi), axis=0), 2)

# print(f'& {ece_class_mean[0]}\(\pm\){ece_class_std[0]} & {ece_class_mean[1]}\(\pm\){ece_class_std[1]} & {ece_class_mean[2]}\(\pm\){ece_class_std[2]} & {ece_class_mean[3]}\(\pm\){ece_class_std[3]} & {ece_class_mean[4]}\(\pm\){ece_class_std[4]} & {ece_class_mean[5]}\(\pm\){ece_class_std[5]}')

# print(f'& {mce_class_mean[0]}\(\pm\){mce_class_std[0]} & {mce_class_mean[1]}\(\pm\){mce_class_std[1]} & {mce_class_mean[2]}\(\pm\){mce_class_std[2]} & {mce_class_mean[3]}\(\pm\){mce_class_std[3]} & {mce_class_mean[4]}\(\pm\){mce_class_std[4]} & {mce_class_mean[5]}\(\pm\){ece_class_std[5]}')

# print(f'& {ece_confi_mean[0]}\(\pm\){ece_confi_std[0]} & {ece_confi_mean[1]}\(\pm\){ece_confi_std[1]} & {ece_confi_mean[2]}\(\pm\){ece_confi_std[2]} & {ece_confi_mean[3]}\(\pm\){ece_confi_std[3]} & {ece_confi_mean[4]}\(\pm\){ece_confi_std[4]} & {ece_confi_mean[5]}\(\pm\){ece_confi_std[5]}')

# print(f'& {mce_confi_mean[0]}\(\pm\){mce_confi_std[0]} & {mce_confi_mean[1]}\(\pm\){mce_confi_std[1]} & {mce_confi_mean[2]}\(\pm\){mce_confi_std[2]} & {mce_confi_mean[3]}\(\pm\){mce_confi_std[3]} & {mce_confi_mean[4]}\(\pm\){mce_confi_std[4]} & {mce_confi_mean[5]}\(\pm\){ece_confi_std[5]}')

# # sys.exit()

# <<< CALIBRATION ERRORS <<<

# >>> SET-VALUED PREDICTIONS >>>

def define_discount_ratio(k, discount_type):
    if discount_type == 'f1':
        return 2/(1+k)
    elif discount_type == 'u80':
        return -1.2/(k**2) + 2.2/k
    else:
        discount_type = 'u65'
        return -0.6/(k**2) + 1.6/k
    

def ndc(p_star, discount_type, classes, n_class):
    class_order = np.argsort(-p_star)
    max_eu = 0
    top_k = 0
    for k in range(1,n_class+1):
        # sum(p(y|x))
        # probability_sum = np.sum(p_star[class_order[0:k]])
        # eu = g(|y_bar|) * sum(p(y|x))
        eu = define_discount_ratio(k, discount_type) * np.sum(p_star[class_order[0:k]])
        if eu < max_eu:
            break
        else:
            max_eu = eu
            top_k = k
    
    return list(classes[class_order[0:top_k]])


def cal_set_value_prediction(all_p_star_SED,
                                all_p_star_L1,
                                all_p_star_KLD,
                                classes, n_class):
    u65_SED_predictions = []
    u80_SED_predictions = []
    u65_L1_predictions = []
    u80_L1_predictions = []
    u65_KLD_predictions = []
    u80_KLD_predictions = []

    for i in range(len(all_p_star_SED)):
        # probability_set = output[:,i,:] #(100,10)

        # >>> Squared Euclidean distance >>>

        # p_star_SED = probability_set.mean(axis=0)
        u65_SED = ndc(all_p_star_SED[i], 'u65', classes, n_class)
        u80_SED = ndc(all_p_star_SED[i], 'u80', classes, n_class)
        u65_SED_predictions.append(u65_SED)
        u80_SED_predictions.append(u80_SED)

        # <<< Squared Euclidean distance <<<
        # >>> L1_distance >>>

        u65_L1 = ndc(all_p_star_L1[i], 'u65', classes, n_class)
        u80_L1 = ndc(all_p_star_L1[i], 'u80', classes, n_class)
        u65_L1_predictions.append(u65_L1)
        u80_L1_predictions.append(u80_L1)

        # <<< L1_distance <<<

        # >>> KL_divergence >>>
            
        u65_KLD = ndc(all_p_star_KLD[i], 'u65', classes, n_class)
        u80_KLD = ndc(all_p_star_KLD[i], 'u80', classes, n_class)
        u65_KLD_predictions.append(u65_KLD)
        u80_KLD_predictions.append(u80_KLD)

        # <<< KL_divergence <<<
    return [[u65_SED_predictions, u80_SED_predictions],
            [u65_L1_predictions, u80_L1_predictions],
            [u65_KLD_predictions, u80_KLD_predictions]]

# >>> Remember to load the right files >>>

# [[u65_u65_SED,u65_u80_SED,u80_u80_SED,u80_u65_SED],
# [u65_u65_L1D,u65_u80_L1D,,u80_u80_L1D,u80_u65_L1D],
# [u65_u65_KLD,u65_u80_KLD,u80_u80_KLD,u80_u65_KLD], [fold 1], [fold 2]]
all_set_results_kfold = []

# [[u65_SED_beta_size, u80_SED_beta_size,
# u65_L1_beta_size,  u80_L1_beta_size, 
# u65_KLD_beta_size, u80_KLD_beta_size], [fold 1], [fold 2]]
beta_size_kfold = []

# Structure same as beta_size_kfold
proportion_kfold = []

# [[[corr_prec_u65_SED_corr_prec,
# corr_prec_u65_SED_corr_sett,
# inco_prec_u65_SED_corr_sett,
# inco_prec_u65_SED_inco_sett,
# inco_prec_u65_SED_inco_prec],
# [corr_prec_u65_L1D_corr_prec,
# corr_prec_u65_L1D_corr_sett,
# inco_prec_u65_L1D_corr_sett,
# inco_prec_u65_L1D_inco_sett,
# inco_prec_u65_L1D_inco_prec],
# [corr_prec_u65_KLD_corr_prec,
# corr_prec_u65_KLD_corr_sett,
# inco_prec_u65_KLD_corr_sett,
# inco_prec_u65_KLD_inco_sett,
# inco_prec_u65_KLD_inco_prec]], [fold 1], [fold 2]]
u65_prec_sett_kfold = []

# Structure same as u65_prec_sett_kfold
u80_prec_sett_kfold = []

for fold in range(k_folds):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # fold = 0
    # test_type = 'c'
    test_type = 'clean'
    save_dir = './data/CIFAR-10-C/gaussian_noise_c_fold_nonsensitive/'
    save_file = f'{save_dir}set_value_prediction.txt'
    if not os.path.exists(save_file):
        print('File not exists.')
        sys.exit()

    labels = np.load(f'{save_dir}cifar_test_labels_{test_type}_fold_{fold}.npy')
    with open('./iukm/cifar_test_data_class_to_idx.txt', 'r') as f:
        class_to_idx = json.load(f)
    n_class = len(class_to_idx)
    classes = np.fromiter(class_to_idx.values(), dtype=int)

    all_p_star_SED = np.load(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy')
    all_p_star_L1 =  np.load(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy')
    all_p_star_KLD = np.load(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy')

    all_precise_results = np.ndarray(shape=(3,len(labels)))
    all_precise_results[0] = np.load(f'{save_dir}predictions_SED_{test_type}_fold_{fold}.npy')
    all_precise_results[1] = np.load(f'{save_dir}predictions_L1_{test_type}_fold_{fold}.npy') 
    all_precise_results[2] = np.load(f'{save_dir}predictions_KLD_{test_type}_fold_{fold}.npy')

    # <<< Remember to load the right files <<<

    # class_name = list(class_to_idx.keys())

    # [[SED], [L1], [KLD]]        
    # [[[1],[2]], [[3],[4]], [[5],[6]]]        
    all_set_predictions = cal_set_value_prediction(all_p_star_SED,
                                                    all_p_star_L1,
                                                    all_p_star_KLD,
                                                    classes, n_class)
    # [[u65_u65, u65_u80, u80_u80, u80_u65], [L1], [KLD]]
    all_set_results = np.ndarray(shape=(3,4))
    # [[SED], [L1], [KLD]]
    # [[u65_u65_uni, u65_u65_mul, 
    #   u80_u80_uni, u80_u80_mul], [L1], [KLD]]
    all_set_count = np.zeros([3, 4])
    # [[precise_u65_uni, precise_u65_mul,
    #   precise_u80_uni, precise_u80_mul], [L1], [KLD]]
    all_precise_count = np.zeros([3, 4])
    # [[SED], [L1], [KLD]]
    # [[u65_set_count, u65_element_count,
    #   u80_set_count, u80_element_count], [L1], [KLD]]
    avg_beta_size = np.zeros([3, 4])


    # >>> From set-valued perspective >>>

    for j in range(len(all_set_predictions)):
        u65_u65 = 0
        u65_u80 = 0
        u80_u80 = 0
        u80_u65 = 0            
        for i in range(len(labels)):
            u65_prediction = all_set_predictions[j][0][i]
            u80_prediction = all_set_predictions[j][1][i]

            if len(u65_prediction) == 1:
                if u65_prediction[0] == labels[i]:
                    u65_u65 += 1
                    u65_u80 += 1
                    all_set_count[j][0] += 1

                if all_precise_results[j][i] == labels[i]:
                    all_precise_count[j][0] += 1
            else:
                all_set_count[j][1] += 1
                avg_beta_size[j][0] += 1
                avg_beta_size[j][1] += len(u65_prediction)

                if labels[i] in u65_prediction:
                    u65_u65 += -0.6/(len(u65_prediction)**2) + 1.6/len(u65_prediction)
                    u65_u80 += -1.2/(len(u65_prediction)**2) + 2.2/len(u65_prediction)
                    
                
                if all_precise_results[j][i] == labels[i]:
                    all_precise_count[j][1] += 1

            if len(u80_prediction) == 1:
                if u80_prediction[0] == labels[i]:
                    u80_u80 += 1
                    u80_u65 += 1
                    all_set_count[j][2] += 1

                if all_precise_results[j][i] == labels[i]:
                    all_precise_count[j][2] += 1
            else:
                all_set_count[j][3] += 1
                avg_beta_size[j][2] += 1
                avg_beta_size[j][3] += len(u80_prediction)

                if labels[i] in u80_prediction:
                    u80_u80 += (-1.2/(len(u80_prediction)**2) + 2.2/len(u80_prediction))
                    u80_u65 += (-0.6/(len(u80_prediction)**2) + 1.6/len(u80_prediction))                
                    

                if all_precise_results[j][i] == labels[i]:
                    all_precise_count[j][3] += 1

        all_set_results[j][0] = u65_u65
        all_set_results[j][1] = u65_u80
        all_set_results[j][2] = u80_u80
        all_set_results[j][3] = u80_u65

    all_set_results_kfold.append((all_set_results/len(labels))*100)
    all_set_results = np.round((all_set_results/len(labels))*100, 2)
    all_set_count /= len(labels)
    all_precise_count /= len(labels)

    u65_SED_beta_size = avg_beta_size[0][1]/avg_beta_size[0][0]
    u80_SED_beta_size = avg_beta_size[0][3]/avg_beta_size[0][2]
    u65_L1_beta_size =  avg_beta_size[1][1]/avg_beta_size[1][0]
    u80_L1_beta_size =  avg_beta_size[1][3]/avg_beta_size[1][2]
    u65_KLD_beta_size = avg_beta_size[2][1]/avg_beta_size[2][0]
    u80_KLD_beta_size = avg_beta_size[2][3]/avg_beta_size[2][2]

    beta_size_kfold.append([u65_SED_beta_size,
                            u80_SED_beta_size,
                            u65_L1_beta_size ,
                            u80_L1_beta_size ,
                            u65_KLD_beta_size,
                            u80_KLD_beta_size])

    u65_SED_proportion = (avg_beta_size[0][0]/len(labels))*100
    u80_SED_proportion = (avg_beta_size[0][2]/len(labels))*100
    u65_L1_proportion =  (avg_beta_size[1][0]/len(labels))*100
    u80_L1_proportion =  (avg_beta_size[1][2]/len(labels))*100
    u65_KLD_proportion = (avg_beta_size[2][0]/len(labels))*100
    u80_KLD_proportion = (avg_beta_size[2][2]/len(labels))*100

    proportion_kfold.append([u65_SED_proportion,
                             u80_SED_proportion,
                             u65_L1_proportion ,
                             u80_L1_proportion ,
                             u65_KLD_proportion,
                             u80_KLD_proportion])

    # print('u65_SED_u65:', all_set_results[0][0])
    # print('u65_SED_u80:', all_set_results[0][1])
    # print('u80_SED_u80:', all_set_results[0][2])
    # print('u80_SED_u65:', all_set_results[0][3])

    # print('u65_L1_u65:', all_set_results[1][0])
    # print('u65_L1_u80:', all_set_results[1][1])
    # print('u80_L1_u80:', all_set_results[1][2])
    # print('u80_L1_u65:', all_set_results[1][3])

    # print('u65_KLD_u65:', all_set_results[2][0])
    # print('u65_KLD_u80:', all_set_results[2][1])
    # print('u80_KLD_u80:', all_set_results[2][2])
    # print('u80_KLD_u65:', all_set_results[2][3])

    # with open(save_file, "a") as f:
    #     f.write(f'\n>>> Fold {fold} >>>')
    #     f.write('\n>>> Set predictions >>>')
    #     f.write(f'\nu65_SED_u65: {all_set_results[0][0]}')
    #     f.write(f'\nu65_SED_u80: {all_set_results[0][1]}')
    #     f.write(f'\nu80_SED_u80: {all_set_results[0][2]}')
    #     f.write(f'\nu80_SED_u65: {all_set_results[0][3]}')
    #     f.write(f'\nu65_L1_u65:  {all_set_results[1][0]}')
    #     f.write(f'\nu65_L1_u80:  {all_set_results[1][1]}')
    #     f.write(f'\nu80_L1_u80:  {all_set_results[1][2]}')
    #     f.write(f'\nu80_L1_u65:  {all_set_results[1][3]}')
    #     f.write(f'\nu65_KLD_u65: {all_set_results[2][0]}')
    #     f.write(f'\nu65_KLD_u80: {all_set_results[2][1]}')
    #     f.write(f'\nu80_KLD_u80: {all_set_results[2][2]}')
    #     f.write(f'\nu80_KLD_u65: {all_set_results[2][3]}')
    #     f.write('\n<<< Set predictions <<<')
    #     f.write('\n>>> Beta size >>>')
    #     f.write(f'\nu65_SED_beta_size: {u65_SED_beta_size}')
    #     f.write(f'\nu80_SED_beta_size: {u80_SED_beta_size}')
    #     f.write(f'\nu65_L1_beta_size : {u65_L1_beta_size }')
    #     f.write(f'\nu80_L1_beta_size : {u80_L1_beta_size }')
    #     f.write(f'\nu65_KLD_beta_size: {u65_KLD_beta_size}')
    #     f.write(f'\nu80_KLD_beta_size: {u80_KLD_beta_size}')
    #     f.write(f'\nu65_SED_proportion: {u65_SED_proportion}')
    #     f.write(f'\nu80_SED_proportion: {u80_SED_proportion}')
    #     f.write(f'\nu65_L1_proportion : {u65_L1_proportion }')
    #     f.write(f'\nu80_L1_proportion : {u80_L1_proportion }')
    #     f.write(f'\nu65_KLD_proportion: {u65_KLD_proportion}')
    #     f.write(f'\nu80_KLD_proportion: {u80_KLD_proportion}')
    #     f.write('\n<<< Beta size <<<')


    # >>> Set size >>>

    # print('u65_SED_beta_size:', u65_SED_beta_size)
    # print('u80_SED_beta_size:', u80_SED_beta_size)
    # print('u65_L1_beta_size :', u65_L1_beta_size )
    # print('u80_L1_beta_size :', u80_L1_beta_size )
    # print('u65_KLD_beta_size:', u65_KLD_beta_size)
    # print('u80_KLD_beta_size:', u80_KLD_beta_size)

    # print('u65_SED_proportion:', u65_SED_proportion)
    # print('u80_SED_proportion:', u80_SED_proportion)
    # print('u65_L1_proportion :', u65_L1_proportion )
    # print('u80_L1_proportion :', u80_L1_proportion )
    # print('u65_KLD_proportion:', u65_KLD_proportion)
    # print('u80_KLD_proportion:', u80_KLD_proportion)

    # <<< Set size <<<

    # <<< From set-valued perspective <<<

    # >>> from precise perspective >>>        

    # [[SED], [L1], [KLD]]
    # [[precise_correct, precise_incorrect], [L1], [KLD]]
    precise_cor_inc_count = np.zeros([3, 2])
    # precise       precise_correct*         precise_incorrect*
    # set_valued    0 precise_correct*       2 precise_incorrect
    #               1 set_correct            3 set_correct*
    #                                        4 set_incorrect
    # [[SED], [L1], [KLD]]
    # [[u65_precise_correct, u65_set_correct, 
    #   u65_precise_incorrect, u65_set_correct, u65_set_incorrect,
    #   u80_precise_correct, u80_set_correct, 
    #   u80_precise_incorrect, u80_set_correct, u80_set_incorrect],
    #   [L1], [KLD]]
    set_cor_inc_count = np.zeros([3, 10])
    # all_set_predictions = [[u65_SED_predictions, u80_SED_predictions],
    #                        [u65_L1_predictions, u80_L1_predictions],
    #                        [u65_KLD_predictions, u80_KLD_predictions]]
    # If True label = 2:
    # Precise prediction = 1
    # u65 = [1], singleton --> pre_inc_u65_sing_incor + 1
    # u80 = [1, 3], set    --> pre_inc_u80_set_incor + 1
    # Therefore: pre_inc_u80_set_incor larger than pre_inc_u65_set_incor
    for j in range(len(all_precise_results)):
        for i in range(len(labels)):
            # precise + correct
            if all_precise_results[j][i] == labels[i]:
                precise_cor_inc_count[j][0] += 1
                # u65
                if len(all_set_predictions[j][0][i]) == 1:
                    set_cor_inc_count[j][0] += 1
                else:
                    set_cor_inc_count[j][1] += 1
                # u80
                if len(all_set_predictions[j][1][i]) == 1:
                    set_cor_inc_count[j][5] += 1
                else:
                    set_cor_inc_count[j][6] += 1

            else: # precise + incorrect
                precise_cor_inc_count[j][1] += 1
                # u65
                if len(all_set_predictions[j][0][i]) == 1:                
                    set_cor_inc_count[j][2] += 1
                elif labels[i] not in all_set_predictions[j][0][i]:
                    set_cor_inc_count[j][4] += 1
                else:
                    set_cor_inc_count[j][3] += 1
                # u80
                if len(all_set_predictions[j][1][i]) == 1:
                    set_cor_inc_count[j][7] += 1
                elif labels[i] not in all_set_predictions[j][1][i]:
                    set_cor_inc_count[j][9] += 1
                else:
                    set_cor_inc_count[j][8] += 1
            


    precise_cor_inc_count_labels = precise_cor_inc_count / len(labels)
    set_cor_inc_count_labels = set_cor_inc_count / len(labels)

    # [[SED], [L1], [KLD]]
    # [[u65_precise_correct, u65_set_correct, 
    #   u65_precise_incorrect, u65_set_correct, u65_set_incorrect,
    #   u80_precise_correct, u80_set_correct, 
    #   u80_precise_incorrect, u80_set_correct, u80_set_incorrect],
    #   [L1], [KLD]]
    # [[precise_correct, precise_incorrect], [L1], [KLD]]
    corr_prec_u65_SED_corr_prec = (set_cor_inc_count[0][0]/precise_cor_inc_count[0][0])*100
    corr_prec_u65_SED_corr_sett = (set_cor_inc_count[0][1]/precise_cor_inc_count[0][0])*100
    inco_prec_u65_SED_corr_sett = (set_cor_inc_count[0][3]/precise_cor_inc_count[0][1])*100
    inco_prec_u65_SED_inco_sett = (set_cor_inc_count[0][4]/precise_cor_inc_count[0][1])*100
    inco_prec_u65_SED_inco_prec = (set_cor_inc_count[0][2]/precise_cor_inc_count[0][1])*100
    
    corr_prec_u65_L1_corr_prec =  (set_cor_inc_count[1][0]/precise_cor_inc_count[1][0])*100
    corr_prec_u65_L1_corr_sett =  (set_cor_inc_count[1][1]/precise_cor_inc_count[1][0])*100
    inco_prec_u65_L1_corr_sett =  (set_cor_inc_count[1][3]/precise_cor_inc_count[1][1])*100
    inco_prec_u65_L1_inco_sett =  (set_cor_inc_count[1][4]/precise_cor_inc_count[1][1])*100
    inco_prec_u65_L1_inco_prec =  (set_cor_inc_count[1][2]/precise_cor_inc_count[1][1])*100
    
    corr_prec_u65_KLD_corr_prec = (set_cor_inc_count[2][0]/precise_cor_inc_count[2][0])*100
    corr_prec_u65_KLD_corr_sett = (set_cor_inc_count[2][1]/precise_cor_inc_count[2][0])*100
    inco_prec_u65_KLD_corr_sett = (set_cor_inc_count[2][3]/precise_cor_inc_count[2][1])*100
    inco_prec_u65_KLD_inco_sett = (set_cor_inc_count[2][4]/precise_cor_inc_count[2][1])*100
    inco_prec_u65_KLD_inco_prec = (set_cor_inc_count[2][2]/precise_cor_inc_count[2][1])*100

    u65_prec_sett_kfold.append([[corr_prec_u65_SED_corr_prec,
                                corr_prec_u65_SED_corr_sett,
                                inco_prec_u65_SED_corr_sett,
                                inco_prec_u65_SED_inco_sett,
                                inco_prec_u65_SED_inco_prec],
                                [corr_prec_u65_L1_corr_prec,
                                corr_prec_u65_L1_corr_sett,
                                inco_prec_u65_L1_corr_sett,
                                inco_prec_u65_L1_inco_sett,
                                inco_prec_u65_L1_inco_prec],
                                [corr_prec_u65_KLD_corr_prec,
                                corr_prec_u65_KLD_corr_sett,
                                inco_prec_u65_KLD_corr_sett,
                                inco_prec_u65_KLD_inco_sett,
                                inco_prec_u65_KLD_inco_prec]])

    corr_prec_u80_SED_corr_prec = (set_cor_inc_count[0][5]/precise_cor_inc_count[0][0])*100
    corr_prec_u80_SED_corr_sett = (set_cor_inc_count[0][6]/precise_cor_inc_count[0][0])*100
    inco_prec_u80_SED_corr_sett = (set_cor_inc_count[0][8]/precise_cor_inc_count[0][1])*100
    inco_prec_u80_SED_inco_sett = (set_cor_inc_count[0][9]/precise_cor_inc_count[0][1])*100
    inco_prec_u80_SED_inco_prec = (set_cor_inc_count[0][7]/precise_cor_inc_count[0][1])*100
    
    corr_prec_u80_L1_corr_prec =  (set_cor_inc_count[1][5]/precise_cor_inc_count[1][0])*100
    corr_prec_u80_L1_corr_sett =  (set_cor_inc_count[1][6]/precise_cor_inc_count[1][0])*100
    inco_prec_u80_L1_corr_sett =  (set_cor_inc_count[1][8]/precise_cor_inc_count[1][1])*100
    inco_prec_u80_L1_inco_sett =  (set_cor_inc_count[1][9]/precise_cor_inc_count[1][1])*100
    inco_prec_u80_L1_inco_prec =  (set_cor_inc_count[1][7]/precise_cor_inc_count[1][1])*100
    
    corr_prec_u80_KLD_corr_prec = (set_cor_inc_count[2][5]/precise_cor_inc_count[2][0])*100
    corr_prec_u80_KLD_corr_sett = (set_cor_inc_count[2][6]/precise_cor_inc_count[2][0])*100
    inco_prec_u80_KLD_corr_sett = (set_cor_inc_count[2][8]/precise_cor_inc_count[2][1])*100
    inco_prec_u80_KLD_inco_sett = (set_cor_inc_count[2][9]/precise_cor_inc_count[2][1])*100
    inco_prec_u80_KLD_inco_prec = (set_cor_inc_count[2][7]/precise_cor_inc_count[2][1])*100

    u80_prec_sett_kfold.append([[corr_prec_u80_SED_corr_prec,
                                corr_prec_u80_SED_corr_sett,
                                inco_prec_u80_SED_corr_sett,
                                inco_prec_u80_SED_inco_sett,
                                inco_prec_u80_SED_inco_prec],
                                [corr_prec_u80_L1_corr_prec,
                                corr_prec_u80_L1_corr_sett,
                                inco_prec_u80_L1_corr_sett,
                                inco_prec_u80_L1_inco_sett,
                                inco_prec_u80_L1_inco_prec],
                                [corr_prec_u80_KLD_corr_prec,
                                corr_prec_u80_KLD_corr_sett,
                                inco_prec_u80_KLD_corr_sett,
                                inco_prec_u80_KLD_inco_sett,
                                inco_prec_u80_KLD_inco_prec]])

    # <<< from precise persepctive <<<

    # # >>> Set-valued predictions unielement or multielement >>>

    # print('u65_SED_uni:', all_set_count[0][0])
    # print('u65_SED_mul:', all_set_count[0][1])
    # print('u80_SED_uni:', all_set_count[0][2])
    # print('u80_SED_mul:', all_set_count[0][3])

    # print('precise_u65_SED_uni:', all_precise_count[0][0])
    # print('precise_u65_SED_mul:', all_precise_count[0][1])
    # print('precise_u80_SED_uni:', all_precise_count[0][2])
    # print('precise_u80_SED_mul:', all_precise_count[0][3])

    # print('u65_L1_uni:', all_set_count[1][0])
    # print('u65_L1_mul:', all_set_count[1][1])
    # print('u80_L1_uni:', all_set_count[1][2])
    # print('u80_L1_mul:', all_set_count[1][3])

    # print('precise_u65_L1_uni:', all_precise_count[1][0])
    # print('precise_u65_L1_mul:', all_precise_count[1][1])
    # print('precise_u80_L1_uni:', all_precise_count[1][2])
    # print('precise_u80_L1_mul:', all_precise_count[1][3])

    # print('u65_KLD_uni:', all_set_count[2][0])
    # print('u65_KLD_mul:', all_set_count[2][1])
    # print('u80_KLD_uni:', all_set_count[2][2])
    # print('u80_KLD_mul:', all_set_count[2][3])

    # print('precise_u65_KLD_uni:', all_precise_count[2][0])
    # print('precise_u65_KLD_mul:', all_precise_count[2][1])
    # print('precise_u80_KLD_uni:', all_precise_count[2][2])
    # print('precise_u80_KLD_mul:', all_precise_count[2][3])

    # # <<< Set-valued predictions unielement or multielement <<<

    # >>> from precise perspective >>>
    # >>> Over labels >>>

    # print('precise_correct_count_SED:', precise_cor_inc_count_labels[0][0])
    # print('precise_incorrect_count_SED:', precise_cor_inc_count_labels[0][1])
    # print('precise_correct_count_L1:', precise_cor_inc_count_labels[1][0])
    # print('precise_incorrect_count_L1:', precise_cor_inc_count_labels[1][1])
    # print('precise_correct_count_KLD:', precise_cor_inc_count_labels[2][0])
    # print('precise_incorrect_count_KLD:', precise_cor_inc_count_labels[2][1])

    # print('u65_precise_correct_count_SED:', set_cor_inc_count_labels[0][0])
    # print('u65_set_correct_count_SED:', set_cor_inc_count_labels[0][1])
    # print('u65_precise_incorrect_count_SED:', set_cor_inc_count_labels[0][2])
    # print('u65_set_correct_count_SED:', set_cor_inc_count_labels[0][3])
    # print('u65_set_incorrect_count_SED:', set_cor_inc_count_labels[0][4])

    # print('u80_precise_correct_count_SED:', set_cor_inc_count_labels[0][5])
    # print('u80_set_correct_count_SED:', set_cor_inc_count_labels[0][6])
    # print('u80_precise_incorrect_count_SED:', set_cor_inc_count_labels[0][7])
    # print('u80_set_correct_count_SED:', set_cor_inc_count_labels[0][8])
    # print('u80_set_incorrect_count_SED:', set_cor_inc_count_labels[0][9])

    # print('u65_precise_correct_count_L1:', set_cor_inc_count_labels[1][0])
    # print('u65_set_correct_count_L1:', set_cor_inc_count_labels[1][1])
    # print('u65_precise_incorrect_count_L1:', set_cor_inc_count_labels[1][2])
    # print('u65_set_correct_count_L1:', set_cor_inc_count_labels[1][3])
    # print('u65_set_incorrect_count_L1:', set_cor_inc_count_labels[1][4])

    # print('u80_precise_correct_count_L1:', set_cor_inc_count_labels[1][5])
    # print('u80_set_correct_count_L1:', set_cor_inc_count_labels[1][6])
    # print('u80_precise_incorrect_count_L1:', set_cor_inc_count_labels[1][7])
    # print('u80_set_correct_count_L1:', set_cor_inc_count_labels[1][8])
    # print('u80_set_incorrect_count_L1:', set_cor_inc_count_labels[1][9])

    # print('u65_precise_correct_count_KLD:', set_cor_inc_count_labels[2][0])
    # print('u65_set_correct_count_KLD:', set_cor_inc_count_labels[2][1])
    # print('u65_precise_incorrect_count_KLD:', set_cor_inc_count_labels[2][2])
    # print('u65_set_correct_count_KLD:', set_cor_inc_count_labels[2][3])
    # print('u65_set_incorrect_count_KLD:', set_cor_inc_count_labels[2][4])

    # print('u80_precise_correct_count_KLD:', set_cor_inc_count_labels[2][5])
    # print('u80_set_correct_count_KLD:', set_cor_inc_count_labels[2][6])
    # print('u80_precise_incorrect_count_KLD:', set_cor_inc_count_labels[2][7])
    # print('u80_set_correct_count_KLD:', set_cor_inc_count_labels[2][8])
    # print('u80_set_incorrect_count_KLD:', set_cor_inc_count_labels[2][9])

    # <<< Over labels <<<

    # with open(save_file, "a") as f:
    #     f.write(f'\ncorr_prec_u65_SED_corr_prec: {corr_prec_u65_SED_corr_prec}')
    #     f.write(f'\ncorr_prec_u65_SED_corr_sett: {corr_prec_u65_SED_corr_sett}')
    #     f.write(f'\ninco_prec_u65_SED_corr_sett: {inco_prec_u65_SED_corr_sett}')
    #     f.write(f'\ninco_prec_u65_SED_inco_sett: {inco_prec_u65_SED_inco_sett}')
    #     f.write(f'\ninco_prec_u65_SED_inco_prec: {inco_prec_u65_SED_inco_prec}')
    #     f.write(f'\ncorr_prec_u65_L1_corr_prec:  {corr_prec_u65_L1_corr_prec }')
    #     f.write(f'\ncorr_prec_u65_L1_corr_sett:  {corr_prec_u65_L1_corr_sett }')
    #     f.write(f'\ninco_prec_u65_L1_corr_sett:  {inco_prec_u65_L1_corr_sett }')
    #     f.write(f'\ninco_prec_u65_L1_inco_sett:  {inco_prec_u65_L1_inco_sett }')
    #     f.write(f'\ninco_prec_u65_L1_inco_prec:  {inco_prec_u65_L1_inco_prec }')
    #     f.write(f'\ncorr_prec_u65_KLD_corr_prec: {corr_prec_u65_KLD_corr_prec}')
    #     f.write(f'\ncorr_prec_u65_KLD_corr_sett: {corr_prec_u65_KLD_corr_sett}')
    #     f.write(f'\ninco_prec_u65_KLD_corr_sett: {inco_prec_u65_KLD_corr_sett}')
    #     f.write(f'\ninco_prec_u65_KLD_inco_sett: {inco_prec_u65_KLD_inco_sett}')
    #     f.write(f'\ninco_prec_u65_KLD_inco_prec: {inco_prec_u65_KLD_inco_prec}')
    #     f.write(f'\ncorr_prec_u80_SED_corr_prec: {corr_prec_u80_SED_corr_prec}')
    #     f.write(f'\ncorr_prec_u80_SED_corr_sett: {corr_prec_u80_SED_corr_sett}')
    #     f.write(f'\ninco_prec_u80_SED_corr_sett: {inco_prec_u80_SED_corr_sett}')
    #     f.write(f'\ninco_prec_u80_SED_inco_sett: {inco_prec_u80_SED_inco_sett}')
    #     f.write(f'\ninco_prec_u80_SED_inco_prec: {inco_prec_u80_SED_inco_prec}')
    #     f.write(f'\ncorr_prec_u80_L1_corr_prec:  {corr_prec_u80_L1_corr_prec }')
    #     f.write(f'\ncorr_prec_u80_L1_corr_sett:  {corr_prec_u80_L1_corr_sett }')
    #     f.write(f'\ninco_prec_u80_L1_corr_sett:  {inco_prec_u80_L1_corr_sett }')
    #     f.write(f'\ninco_prec_u80_L1_inco_sett:  {inco_prec_u80_L1_inco_sett }')
    #     f.write(f'\ninco_prec_u80_L1_inco_prec:  {inco_prec_u80_L1_inco_prec }')
    #     f.write(f'\ncorr_prec_u80_KLD_corr_prec: {corr_prec_u80_KLD_corr_prec}')
    #     f.write(f'\ncorr_prec_u80_KLD_corr_sett: {corr_prec_u80_KLD_corr_sett}')
    #     f.write(f'\ninco_prec_u80_KLD_corr_sett: {inco_prec_u80_KLD_corr_sett}')
    #     f.write(f'\ninco_prec_u80_KLD_inco_sett: {inco_prec_u80_KLD_inco_sett}')
    #     f.write(f'\ninco_prec_u80_KLD_inco_prec: {inco_prec_u80_KLD_inco_prec}')


# >>> Set predictions mean and std >>>

all_set_results_mean = np.round(np.mean(all_set_results_kfold, axis=0), 2)
all_set_results_std  = np.round(np.std(all_set_results_kfold, axis=0), 2)

# <<< Set prediction mean and std <<<

# >>> Beta size and proportion mean and std >>>

beta_size_mean = np.round(np.mean(beta_size_kfold, axis=0), 2)
beta_size_std  = np.round(np.std(beta_size_kfold, axis=0), 2)
proportion_mean = np.round(np.mean(proportion_kfold, axis=0), 2)
proportion_std = np.round(np.std(proportion_kfold, axis=0), 2)

# <<< Beta size and proportion mean and std <<<

# >>> Precise and set correct and incorrect >>>

u65_prec_sett_mean = np.round(np.mean(u65_prec_sett_kfold, axis=0), 2)
u65_prec_sett_std  = np.round(np.std(u65_prec_sett_kfold, axis=0), 2)

u80_prec_sett_mean = np.round(np.mean(u80_prec_sett_kfold, axis=0), 2)
u80_prec_sett_std  = np.round(np.std(u80_prec_sett_kfold, axis=0), 2)

# <<< Precise and set correct and incorrect <<<

with open(save_file, "a") as f:
        f.write('\n>>> Set predictions >>>')
        f.write(f'\nu65_SED_u65_mean: {all_set_results_mean[0][0]}')
        f.write(f'\nu65_SED_u80_mean: {all_set_results_mean[0][1]}')
        f.write(f'\nu80_SED_u80_mean: {all_set_results_mean[0][2]}')
        f.write(f'\nu80_SED_u65_mean: {all_set_results_mean[0][3]}')
        f.write(f'\nu65_L1D_u65_mean: {all_set_results_mean[1][0]}')
        f.write(f'\nu65_L1D_u80_mean: {all_set_results_mean[1][1]}')
        f.write(f'\nu80_L1D_u80_mean: {all_set_results_mean[1][2]}')
        f.write(f'\nu80_L1D_u65_mean: {all_set_results_mean[1][3]}')
        f.write(f'\nu65_KLD_u65_mean: {all_set_results_mean[2][0]}')
        f.write(f'\nu65_KLD_u80_mean: {all_set_results_mean[2][1]}')
        f.write(f'\nu80_KLD_u80_mean: {all_set_results_mean[2][2]}')
        f.write(f'\nu80_KLD_u65_mean: {all_set_results_mean[2][3]}')
        f.write(f'\nu65_SED_u65_std: {all_set_results_std[0][0]}')
        f.write(f'\nu65_SED_u80_std: {all_set_results_std[0][1]}')
        f.write(f'\nu80_SED_u80_std: {all_set_results_std[0][2]}')
        f.write(f'\nu80_SED_u65_std: {all_set_results_std[0][3]}')
        f.write(f'\nu65_L1D_u65_std: {all_set_results_std[1][0]}')
        f.write(f'\nu65_L1D_u80_std: {all_set_results_std[1][1]}')
        f.write(f'\nu80_L1D_u80_std: {all_set_results_std[1][2]}')
        f.write(f'\nu80_L1D_u65_std: {all_set_results_std[1][3]}')
        f.write(f'\nu65_KLD_u65_std: {all_set_results_std[2][0]}')
        f.write(f'\nu65_KLD_u80_std: {all_set_results_std[2][1]}')
        f.write(f'\nu80_KLD_u80_std: {all_set_results_std[2][2]}')
        f.write(f'\nu80_KLD_u65_std: {all_set_results_std[2][3]}')
        f.write('\n<<< Set predictions <<<')
        f.write('\n>>> Beta size >>>')
        f.write(f'\nu65_SED_beta_size_mean: {beta_size_mean[0]}')
        f.write(f'\nu80_SED_beta_size_mean: {beta_size_mean[1]}')
        f.write(f'\nu65_L1D_beta_size_mean: {beta_size_mean[2]}')
        f.write(f'\nu80_L1D_beta_size_mean: {beta_size_mean[3]}')
        f.write(f'\nu65_KLD_beta_size_mean: {beta_size_mean[4]}')
        f.write(f'\nu80_KLD_beta_size_mean: {beta_size_mean[5]}')
        f.write(f'\nu65_SED_beta_size_std: {beta_size_std[0]}')
        f.write(f'\nu80_SED_beta_size_std: {beta_size_std[1]}')
        f.write(f'\nu65_L1D_beta_size_std: {beta_size_std[2]}')
        f.write(f'\nu80_L1D_beta_size_std: {beta_size_std[3]}')
        f.write(f'\nu65_KLD_beta_size_std: {beta_size_std[4]}')
        f.write(f'\nu80_KLD_beta_size_std: {beta_size_std[5]}')
        f.write('\n<<< Beta size <<<')
        f.write('\n>>> Proportion >>>')
        f.write(f'\nu65_SED_proportion_mean: {proportion_mean[0]}')
        f.write(f'\nu80_SED_proportion_mean: {proportion_mean[1]}')
        f.write(f'\nu65_L1D_proportion_mean: {proportion_mean[2]}')
        f.write(f'\nu80_L1D_proportion_mean: {proportion_mean[3]}')
        f.write(f'\nu65_KLD_proportion_mean: {proportion_mean[4]}')
        f.write(f'\nu80_KLD_proportion_mean: {proportion_mean[5]}')
        f.write(f'\nu65_SED_proportion_std: {proportion_std[0]}')
        f.write(f'\nu80_SED_proportion_std: {proportion_std[1]}')
        f.write(f'\nu65_L1D_proportion_std: {proportion_std[2]}')
        f.write(f'\nu80_L1D_proportion_std: {proportion_std[3]}')
        f.write(f'\nu65_KLD_proportion_std: {proportion_std[4]}')
        f.write(f'\nu80_KLD_proportion_std: {proportion_std[5]}')
        f.write('\n<<< Proportion <<<')
        f.write('\n>>> Precise and set correct and incorrect >>>')
        f.write(f'\ncorr_prec_u65_SED_corr_prec_mean: {u65_prec_sett_mean[0][0]}')
        f.write(f'\ncorr_prec_u65_SED_corr_sett_mean: {u65_prec_sett_mean[0][1]}')
        f.write(f'\ninco_prec_u65_SED_corr_sett_mean: {u65_prec_sett_mean[0][2]}')
        f.write(f'\ninco_prec_u65_SED_inco_sett_mean: {u65_prec_sett_mean[0][3]}')
        f.write(f'\ninco_prec_u65_SED_inco_prec_mean: {u65_prec_sett_mean[0][4]}')
        f.write(f'\ncorr_prec_u65_L1_corr_prec_mean:  {u65_prec_sett_mean[1][0]}')
        f.write(f'\ncorr_prec_u65_L1_corr_sett_mean:  {u65_prec_sett_mean[1][1]}')
        f.write(f'\ninco_prec_u65_L1_corr_sett_mean:  {u65_prec_sett_mean[1][2]}')
        f.write(f'\ninco_prec_u65_L1_inco_sett_mean:  {u65_prec_sett_mean[1][3]}')
        f.write(f'\ninco_prec_u65_L1_inco_prec_mean:  {u65_prec_sett_mean[1][4]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_prec_mean: {u65_prec_sett_mean[2][0]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_sett_mean: {u65_prec_sett_mean[2][1]}')
        f.write(f'\ninco_prec_u65_KLD_corr_sett_mean: {u65_prec_sett_mean[2][2]}')
        f.write(f'\ninco_prec_u65_KLD_inco_sett_mean: {u65_prec_sett_mean[2][3]}')
        f.write(f'\ninco_prec_u65_KLD_inco_prec_mean: {u65_prec_sett_mean[2][4]}')
        f.write(f'\ncorr_prec_u80_SED_corr_prec_mean: {u80_prec_sett_mean[0][0]}')
        f.write(f'\ncorr_prec_u80_SED_corr_sett_mean: {u80_prec_sett_mean[0][1]}')
        f.write(f'\ninco_prec_u80_SED_corr_sett_mean: {u80_prec_sett_mean[0][2]}')
        f.write(f'\ninco_prec_u80_SED_inco_sett_mean: {u80_prec_sett_mean[0][3]}')
        f.write(f'\ninco_prec_u80_SED_inco_prec_mean: {u80_prec_sett_mean[0][4]}')
        f.write(f'\ncorr_prec_u80_L1_corr_prec_mean:  {u80_prec_sett_mean[1][0]}')
        f.write(f'\ncorr_prec_u80_L1_corr_sett_mean:  {u80_prec_sett_mean[1][1]}')
        f.write(f'\ninco_prec_u80_L1_corr_sett_mean:  {u80_prec_sett_mean[1][2]}')
        f.write(f'\ninco_prec_u80_L1_inco_sett_mean:  {u80_prec_sett_mean[1][3]}')
        f.write(f'\ninco_prec_u80_L1_inco_prec_mean:  {u80_prec_sett_mean[1][4]}')
        f.write(f'\ncorr_prec_u80_KLD_corr_prec_mean: {u80_prec_sett_mean[2][0]}')
        f.write(f'\ncorr_prec_u80_KLD_corr_sett_mean: {u80_prec_sett_mean[2][1]}')
        f.write(f'\ninco_prec_u80_KLD_corr_sett_mean: {u80_prec_sett_mean[2][2]}')
        f.write(f'\ninco_prec_u80_KLD_inco_sett_mean: {u80_prec_sett_mean[2][3]}')
        f.write(f'\ninco_prec_u80_KLD_inco_prec_mean: {u80_prec_sett_mean[2][4]}')
        f.write(f'\ncorr_prec_u65_SED_corr_prec_std:  {u65_prec_sett_std[0][0]}')
        f.write(f'\ncorr_prec_u65_SED_corr_sett_std:  {u65_prec_sett_std[0][1]}')
        f.write(f'\ninco_prec_u65_SED_corr_sett_std:  {u65_prec_sett_std[0][2]}')
        f.write(f'\ninco_prec_u65_SED_inco_sett_std:  {u65_prec_sett_std[0][3]}')
        f.write(f'\ninco_prec_u65_SED_inco_prec_std:  {u65_prec_sett_std[0][4]}') 
        f.write(f'\ncorr_prec_u65_L1_corr_prec_std:   {u65_prec_sett_std[1][0]}')
        f.write(f'\ncorr_prec_u65_L1_corr_sett_std:   {u65_prec_sett_std[1][1]}')
        f.write(f'\ninco_prec_u65_L1_corr_sett_std:   {u65_prec_sett_std[1][2]}')
        f.write(f'\ninco_prec_u65_L1_inco_sett_std:   {u65_prec_sett_std[1][3]}')
        f.write(f'\ninco_prec_u65_L1_inco_prec_std:   {u65_prec_sett_std[1][4]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_prec_std:  {u65_prec_sett_std[2][0]}')
        f.write(f'\ncorr_prec_u65_KLD_corr_sett_std:  {u65_prec_sett_std[2][1]}')
        f.write(f'\ninco_prec_u65_KLD_corr_sett_std:  {u65_prec_sett_std[2][2]}')
        f.write(f'\ninco_prec_u65_KLD_inco_sett_std:  {u65_prec_sett_std[2][3]}')
        f.write(f'\ninco_prec_u65_KLD_inco_prec_std:  {u65_prec_sett_std[2][4]}')
        f.write(f'\ncorr_prec_u80_SED_corr_prec_std:  {u80_prec_sett_std[0][0]}')
        f.write(f'\ncorr_prec_u80_SED_corr_sett_std:  {u80_prec_sett_std[0][1]}')
        f.write(f'\ninco_prec_u80_SED_corr_sett_std:  {u80_prec_sett_std[0][2]}')
        f.write(f'\ninco_prec_u80_SED_inco_sett_std:  {u80_prec_sett_std[0][3]}')
        f.write(f'\ninco_prec_u80_SED_inco_prec_std:  {u80_prec_sett_std[0][4]}')
        f.write(f'\ncorr_prec_u80_L1_corr_prec_std:   {u80_prec_sett_std[1][0]}')
        f.write(f'\ncorr_prec_u80_L1_corr_sett_std:   {u80_prec_sett_std[1][1]}')
        f.write(f'\ninco_prec_u80_L1_corr_sett_std:   {u80_prec_sett_std[1][2]}')
        f.write(f'\ninco_prec_u80_L1_inco_sett_std:   {u80_prec_sett_std[1][3]}')
        f.write(f'\ninco_prec_u80_L1_inco_prec_std:   {u80_prec_sett_std[1][4]}')
        f.write(f'\ncorr_prec_u80_KLD_corr_prec_std:  {u80_prec_sett_std[2][0]}')
        f.write(f'\ncorr_prec_u80_KLD_corr_sett_std:  {u80_prec_sett_std[2][1]}')
        f.write(f'\ninco_prec_u80_KLD_corr_sett_std:  {u80_prec_sett_std[2][2]}')
        f.write(f'\ninco_prec_u80_KLD_inco_sett_std:  {u80_prec_sett_std[2][3]}')
        f.write(f'\ninco_prec_u80_KLD_inco_prec_std:  {u80_prec_sett_std[2][4]}')
        f.write('\n<<< Precise and set correct and incorrect <<<')


# print('corr_prec_u65_SED_corr_prec:', corr_prec_u65_SED_corr_prec)
# print('corr_prec_u65_SED_corr_sett:', corr_prec_u65_SED_corr_sett)
# print('inco_prec_u65_SED_corr_sett:', inco_prec_u65_SED_corr_sett)
# print('inco_prec_u65_SED_inco_sett:', inco_prec_u65_SED_inco_sett)
# print('inco_prec_u65_SED_inco_prec:', inco_prec_u65_SED_inco_prec)

# print('corr_prec_u65_L1_corr_prec:', corr_prec_u65_L1_corr_prec)
# print('corr_prec_u65_L1_corr_sett:', corr_prec_u65_L1_corr_sett)
# print('inco_prec_u65_L1_corr_sett:', inco_prec_u65_L1_corr_sett)
# print('inco_prec_u65_L1_inco_sett:', inco_prec_u65_L1_inco_sett)
# print('inco_prec_u65_L1_inco_prec:', inco_prec_u65_L1_inco_prec)

# print('corr_prec_u65_KLD_corr_prec:', corr_prec_u65_KLD_corr_prec)
# print('corr_prec_u65_KLD_corr_sett:', corr_prec_u65_KLD_corr_sett)
# print('inco_prec_u65_KLD_corr_sett:', inco_prec_u65_KLD_corr_sett)
# print('inco_prec_u65_KLD_inco_sett:', inco_prec_u65_KLD_inco_sett)
# print('inco_prec_u65_KLD_inco_prec:', inco_prec_u65_KLD_inco_prec)

# print('corr_prec_u80_SED_corr_prec:', corr_prec_u80_SED_corr_prec)
# print('corr_prec_u80_SED_corr_sett:', corr_prec_u80_SED_corr_sett)
# print('inco_prec_u80_SED_corr_sett:', inco_prec_u80_SED_corr_sett)
# print('inco_prec_u80_SED_inco_sett:', inco_prec_u80_SED_inco_sett)
# print('inco_prec_u80_SED_inco_prec:', inco_prec_u80_SED_inco_prec)

# print('corr_prec_u80_L1_corr_prec:', corr_prec_u80_L1_corr_prec)
# print('corr_prec_u80_L1_corr_sett:', corr_prec_u80_L1_corr_sett)
# print('inco_prec_u80_L1_corr_sett:', inco_prec_u80_L1_corr_sett)
# print('inco_prec_u80_L1_inco_sett:', inco_prec_u80_L1_inco_sett)
# print('inco_prec_u80_L1_inco_prec:', inco_prec_u80_L1_inco_prec)

# print('corr_prec_u80_KLD_corr_prec:', corr_prec_u80_KLD_corr_prec)
# print('corr_prec_u80_KLD_corr_sett:', corr_prec_u80_KLD_corr_sett)
# print('inco_prec_u80_KLD_corr_sett:', inco_prec_u80_KLD_corr_sett)
# print('inco_prec_u80_KLD_inco_sett:', inco_prec_u80_KLD_inco_sett)
# print('inco_prec_u80_KLD_inco_prec:', inco_prec_u80_KLD_inco_prec)

# <<< from precise perspective <<<

# <<< SET-VALUED PREDICTIONS <<<