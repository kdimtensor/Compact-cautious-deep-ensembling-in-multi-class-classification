import sys
import json

import numpy as np

from scipy.optimize import minimize
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE


# >>> PRECISE PREDICTIONS >>>

# # output shape ([100, 10000, 10])
# output = np.load('./cifar_ensemble_100_output_best.npy')
# labels = np.load('./cifar_test_labels_best.npy')
# labels_onehot = np.load('./labels_onehot.npy')
# with open('./cifar_test_data_class_to_idxxt', 'r') as f:
#         class_to_idx = json.load(f)
# n_class = len(class_to_idx)
# predictions_L1 = np.zeros(len(labels))
# predictions_KLD = np.zeros(len(labels))
# all_p_star_L1 = np.zeros([len(labels), n_class])
# all_p_star_KLD = np.zeros([len(labels), n_class])

# predictions_L1 = np.zeros(len(labels))
# predictions_KLD = np.zeros(len(labels))
# all_p_star_L1 = np.zeros([len(labels), n_class])
# all_p_star_KLD = np.zeros([len(labels), n_class])

# >>> Squared Euclidean distance >>>

# # pred_mean shape ([10000, 10])
# pred_mean = np.mean(output, axis=0)
# Y_pred = np.argmax(pred_mean, axis=1)

# # <<< Squared Euclidean distance <<<

# for i in range(len(labels)):
#     probability_set = output[:,i,:] #(100,10)

#     # >>> L1_distance >>>

#     def L1_distance(p_star_L1, probability_set):
#         return np.sum(np.abs(probability_set - p_star_L1))

#     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

#     result = minimize(L1_distance, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)    
#     p_star_L1 = result.x

#     # all_p_star_L1 shape [10000, 10]
#     all_p_star_L1[i] = p_star_L1
#     predictions_L1[i] = p_star_L1.argmax()

#     # <<< L1_distance <<<

#     # >>> KL_divergence >>>

#     def KL_divergence(p_star_KLD, probability_set):
#         epsilon = 1e-10
#         p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)
#         probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
#         return np.sum(p_star_KLD * np.log(p_star_KLD/probability_set))
#         # return np.sum(probability_set * np.log(probability_set/p_star_KLD))

#     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
#                             {'type': 'ineq', 'fun': lambda x: x}]
    
#     result = minimize(KL_divergence, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)
#     p_star_KLD = result.x
#     epsilon = 1e-10
#     p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)

#     # all_p_star_KLD shape [10000, 10]
#     all_p_star_KLD[i] = p_star_KLD
#     predictions_KLD[i] = p_star_KLD.argmax()

#     # <<< KL_divergence <<<            

# # >>> Save test outputs >>>

# np.save('./cifar_all_p_star_SED.npy', pred_mean)
# np.save('./cifar_all_p_star_L1.npy', all_p_star_L1)
# np.save('./cifar_all_p_star_KLD.npy', all_p_star_KLD)

# np.save('./cifar_predictions_SED.npy', Y_pred)
# np.save('./cifar_predictions_L1.npy', predictions_L1)
# np.save('./cifar_predictions_KLD.npy', predictions_KLD)

# # <<< Save test output <<<

# print('precise_SED_acc:',(Y_pred == labels).mean()*100)
# print('precise_L1_acc:', (predictions_L1 == labels).mean() * 100)
# print('precise_KLD_acc:', (predictions_KLD == labels).mean() * 100)

# <<< PRECISE PREDICTIONS <<<

# >>> CALIBRATION ERRORS >>>

# all_p_star_SED = np.load('./cifar_all_p_star_SED.npy')
# all_p_star_L1 = np.load('./cifar_all_p_star_L1.npy')
# all_p_star_KLD = np.load('./cifar_all_p_star_KLD.npy')
# predictions_SED = np.load('./cifar_predictions_SED.npy')
# predictions_L1 = np.load('./cifar_predictions_L1.npy')
# predictions_KLD = np.load('./cifar_predictions_KLD.npy')

# labels = np.load('./cifar_test_labels_best.npy')
# labels_onehot = np.load('./cifar_test_labels_onehot_best.npy')

# print('precise_SED_acc:', (predictions_SED == labels).mean() * 100)
# print('precise_L1_acc:', (predictions_L1 == labels).mean() * 100)
# print('precise_KLD_acc:', (predictions_KLD == labels).mean() * 100)

# # print('binary_ECE_SED:', binary_ECE(labels_onehot, all_p_star_SED, bins=30))
# # print('binary_ECE_L1:', binary_ECE(labels_onehot, all_p_star_L1, bins=30))
# # print('binary_ECE_KLD:', binary_ECE(labels_onehot, all_p_star_KLD, bins=30))

# # print('binary_MCE_SED:', binary_MCE(labels_onehot, all_p_star_SED, bins=30))
# # print('binary_MCE_L1:', binary_MCE(labels_onehot, all_p_star_L1, bins=30))
# # print('binary_MCE_KLD:', binary_MCE(labels_onehot, all_p_star_KLD, bins=30))

# # WARNING!!! 
# # Inputs for calibrations functions in the shape [n_instance, n_class], [10000, 10]
# # The pycalib API starts with shape [n_class, n_instance] and use transpose [].T
# print('classwise_ECE_SED:', classwise_ECE(labels_onehot, all_p_star_SED, bins=30))
# print('classwise_ECE_L1:', classwise_ECE(labels_onehot, all_p_star_L1, bins=30))
# print('classwise_ECE_KLD:', classwise_ECE(labels_onehot, all_p_star_KLD, bins=30))

# print('classwise_MCE_SED:', classwise_MCE(labels_onehot, all_p_star_SED, bins=30))
# print('classwise_MCE_L1:', classwise_MCE(labels_onehot, all_p_star_L1, bins=30))
# print('classwise_MCE_KLD:', classwise_MCE(labels_onehot, all_p_star_KLD, bins=30))

# print('conf_ECE_SED:', conf_ECE(labels_onehot, all_p_star_SED, bins=30))
# print('conf_ECE_L1:', conf_ECE(labels_onehot, all_p_star_L1, bins=30))
# print('conf_ECE_KLD:', conf_ECE(labels_onehot, all_p_star_KLD, bins=30))

# print('conf_MCE_SED:', conf_MCE(labels_onehot, all_p_star_SED, bins=30))
# print('conf_MCE_L1:', conf_MCE(labels_onehot, all_p_star_L1, bins=30))
# print('conf_MCE_KLD:', conf_MCE(labels_onehot, all_p_star_KLD, bins=30))

# <<< CALIBRATION ERRORS <<<

# >>> SET-VALUED PREDICTIONS >>>

# output shape ([100, 10000, 10])
output = np.load('./iukm/cifar_ensemble_100_output_best.npy')
labels = np.load('./iukm/cifar_test_labels_best.npy')
with open('./iukm/cifar_test_data_class_to_idx.txt', 'r') as f:
        class_to_idx = json.load(f)
n_class = len(class_to_idx)
classes = np.fromiter(class_to_idx.values(), dtype=int)
# class_name = list(class_to_idx.keys())

u65_SED_predictions = []
u80_SED_predictions = []
u65_L1_predictions = []
u80_L1_predictions = []
u65_KLD_predictions = []
u80_KLD_predictions = []


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
        if eu <= max_eu:
            break
        else:
            max_eu = eu
            top_k = k
    
    return list(classes[class_order[0:top_k]])


for i in range(len(labels)):
    probability_set = output[:,i,:] #(100,10)

    # >>> Squared Euclidean distance >>>

    p_star_SED = probability_set.mean(axis=0)
    u65_SED = ndc(p_star_SED, 'u65', classes, n_class)
    u80_SED = ndc(p_star_SED, 'u80', classes, n_class)
    u65_SED_predictions.append(u65_SED)
    u80_SED_predictions.append(u80_SED)

    # <<< Squared Euclidean distance <<<
    # >>> L1_distance >>>

    def L1_distance(p_star_L1, probability_set):
        return np.sum(np.abs(probability_set - p_star_L1))

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

    result = minimize(L1_distance, np.ones(n_class)/n_class, args=(probability_set,), constraints=constraints)
    
    p_star_L1 = result.x
    u65_L1 = ndc(p_star_L1, 'u65', classes, n_class)
    u80_L1 = ndc(p_star_L1, 'u80', classes, n_class)
    u65_L1_predictions.append(u65_L1)
    u80_L1_predictions.append(u80_L1)

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
    u65_KLD = ndc(p_star_KLD, 'u65', classes, n_class)
    u80_KLD = ndc(p_star_KLD, 'u80', classes, n_class)
    u65_KLD_predictions.append(u65_KLD)
    u80_KLD_predictions.append(u80_KLD)

    # <<< KL_divergence <<<

# [[SED], [L1], [KLD]]        
# [[[1],[2]], [[3],[4]], [[5],[6]]]        
all_set_predictions = [[u65_SED_predictions, u80_SED_predictions],
                        [u65_L1_predictions, u80_L1_predictions],[u65_KLD_predictions, u80_KLD_predictions]]
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
all_precise_results = np.ndarray(shape=(3,len(labels)))
all_precise_results[0] = np.load('./iukm/cifar_predictions_SED.npy')
all_precise_results[1] = np.load('./iukm/cifar_predictions_L1.npy')
all_precise_results[2] = np.load('./iukm/cifar_predictions_KLD.npy')

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
                u65_u65 += (-0.6/(len(u65_prediction)**2) + 1.6/len(u65_prediction))
                u65_u80 += (-1.2/(len(u65_prediction)**2) + 2.2/len(u65_prediction))
                
            
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

all_set_results = np.round((all_set_results/len(labels))*100, 2)
all_set_count /= len(labels)
all_precise_count /= len(labels)

u65_SED_beta_size = np.round(avg_beta_size[0][1]/avg_beta_size[0][0], 2)
u80_SED_beta_size = np.round(avg_beta_size[0][3]/avg_beta_size[0][2], 2)
u65_L1_beta_size =  np.round(avg_beta_size[1][1]/avg_beta_size[1][0], 2)
u80_L1_beta_size =  np.round(avg_beta_size[1][3]/avg_beta_size[1][2], 2)
u65_KLD_beta_size = np.round(avg_beta_size[2][1]/avg_beta_size[2][0], 2)
u80_KLD_beta_size = np.round(avg_beta_size[2][3]/avg_beta_size[2][2], 2)

u65_SED_proportion = np.round((avg_beta_size[0][0]/len(labels))*100, 2)
u80_SED_proportion = np.round((avg_beta_size[0][2]/len(labels))*100, 2)
u65_L1_proportion =  np.round((avg_beta_size[1][0]/len(labels))*100, 2)
u80_L1_proportion =  np.round((avg_beta_size[1][2]/len(labels))*100, 2)
u65_KLD_proportion = np.round((avg_beta_size[2][0]/len(labels))*100, 2)
u80_KLD_proportion = np.round((avg_beta_size[2][2]/len(labels))*100, 2)

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
corr_prec_u65_SED_corr_prec = np.round((set_cor_inc_count[0][0]/precise_cor_inc_count[0][0])*100,2)
corr_prec_u65_SED_corr_sett = np.round((set_cor_inc_count[0][1]/precise_cor_inc_count[0][0])*100,2)
inco_prec_u65_SED_corr_sett = np.round((set_cor_inc_count[0][3]/precise_cor_inc_count[0][1])*100,2)
inco_prec_u65_SED_inco_sett = np.round((set_cor_inc_count[0][4]/precise_cor_inc_count[0][1])*100,2)
inco_prec_u65_SED_inco_prec = np.round((set_cor_inc_count[0][2]/precise_cor_inc_count[0][1])*100,2)

corr_prec_u65_L1_corr_prec =  np.round((set_cor_inc_count[1][0]/precise_cor_inc_count[1][0])*100,2)
corr_prec_u65_L1_corr_sett =  np.round((set_cor_inc_count[1][1]/precise_cor_inc_count[1][0])*100,2)
inco_prec_u65_L1_corr_sett =  np.round((set_cor_inc_count[1][3]/precise_cor_inc_count[1][1])*100,2)
inco_prec_u65_L1_inco_sett =  np.round((set_cor_inc_count[1][4]/precise_cor_inc_count[1][1])*100,2)
inco_prec_u65_L1_inco_prec =  np.round((set_cor_inc_count[1][2]/precise_cor_inc_count[1][1])*100,2)

corr_prec_u65_KLD_corr_prec = np.round((set_cor_inc_count[2][0]/precise_cor_inc_count[2][0])*100,2)
corr_prec_u65_KLD_corr_sett = np.round((set_cor_inc_count[2][1]/precise_cor_inc_count[2][0])*100,2)
inco_prec_u65_KLD_corr_sett = np.round((set_cor_inc_count[2][3]/precise_cor_inc_count[2][1])*100,2)
inco_prec_u65_KLD_inco_sett = np.round((set_cor_inc_count[2][4]/precise_cor_inc_count[2][1])*100,2)
inco_prec_u65_KLD_inco_prec = np.round((set_cor_inc_count[2][2]/precise_cor_inc_count[2][1])*100,2)

corr_prec_u80_SED_corr_prec = np.round((set_cor_inc_count[0][5]/precise_cor_inc_count[0][0])*100,2)
corr_prec_u80_SED_corr_sett = np.round((set_cor_inc_count[0][6]/precise_cor_inc_count[0][0])*100,2)
inco_prec_u80_SED_corr_sett = np.round((set_cor_inc_count[0][8]/precise_cor_inc_count[0][1])*100,2)
inco_prec_u80_SED_inco_sett = np.round((set_cor_inc_count[0][9]/precise_cor_inc_count[0][1])*100,2)
inco_prec_u80_SED_inco_prec = np.round((set_cor_inc_count[0][7]/precise_cor_inc_count[0][1])*100,2)

corr_prec_u80_L1_corr_prec =  np.round((set_cor_inc_count[1][5]/precise_cor_inc_count[1][0])*100,2)
corr_prec_u80_L1_corr_sett =  np.round((set_cor_inc_count[1][6]/precise_cor_inc_count[1][0])*100,2)
inco_prec_u80_L1_corr_sett =  np.round((set_cor_inc_count[1][8]/precise_cor_inc_count[1][1])*100,2)
inco_prec_u80_L1_inco_sett =  np.round((set_cor_inc_count[1][9]/precise_cor_inc_count[1][1])*100,2)
inco_prec_u80_L1_inco_prec =  np.round((set_cor_inc_count[1][7]/precise_cor_inc_count[1][1])*100,2)

corr_prec_u80_KLD_corr_prec = np.round((set_cor_inc_count[2][5]/precise_cor_inc_count[2][0])*100,2)
corr_prec_u80_KLD_corr_sett = np.round((set_cor_inc_count[2][6]/precise_cor_inc_count[2][0])*100,2)
inco_prec_u80_KLD_corr_sett = np.round((set_cor_inc_count[2][8]/precise_cor_inc_count[2][1])*100,2)
inco_prec_u80_KLD_inco_sett = np.round((set_cor_inc_count[2][9]/precise_cor_inc_count[2][1])*100,2)
inco_prec_u80_KLD_inco_prec = np.round((set_cor_inc_count[2][7]/precise_cor_inc_count[2][1])*100,2)

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

print('corr_prec_u65_SED_corr_prec:', corr_prec_u65_SED_corr_prec)
print('corr_prec_u65_SED_corr_sett:', corr_prec_u65_SED_corr_sett)
print('inco_prec_u65_SED_corr_sett:', inco_prec_u65_SED_corr_sett)
print('inco_prec_u65_SED_inco_sett:', inco_prec_u65_SED_inco_sett)
print('inco_prec_u65_SED_inco_prec:', inco_prec_u65_SED_inco_prec)

print('corr_prec_u65_L1_corr_prec:', corr_prec_u65_L1_corr_prec)
print('corr_prec_u65_L1_corr_sett:', corr_prec_u65_L1_corr_sett)
print('inco_prec_u65_L1_corr_sett:', inco_prec_u65_L1_corr_sett)
print('inco_prec_u65_L1_inco_sett:', inco_prec_u65_L1_inco_sett)
print('inco_prec_u65_L1_inco_prec:', inco_prec_u65_L1_inco_prec)

print('corr_prec_u65_KLD_corr_prec:', corr_prec_u65_KLD_corr_prec)
print('corr_prec_u65_KLD_corr_sett:', corr_prec_u65_KLD_corr_sett)
print('inco_prec_u65_KLD_corr_sett:', inco_prec_u65_KLD_corr_sett)
print('inco_prec_u65_KLD_inco_sett:', inco_prec_u65_KLD_inco_sett)
print('inco_prec_u65_KLD_inco_prec:', inco_prec_u65_KLD_inco_prec)

print('corr_prec_u80_SED_corr_prec:', corr_prec_u80_SED_corr_prec)
print('corr_prec_u80_SED_corr_sett:', corr_prec_u80_SED_corr_sett)
print('inco_prec_u80_SED_corr_sett:', inco_prec_u80_SED_corr_sett)
print('inco_prec_u80_SED_inco_sett:', inco_prec_u80_SED_inco_sett)
print('inco_prec_u80_SED_inco_prec:', inco_prec_u80_SED_inco_prec)

print('corr_prec_u80_L1_corr_prec:', corr_prec_u80_L1_corr_prec)
print('corr_prec_u80_L1_corr_sett:', corr_prec_u80_L1_corr_sett)
print('inco_prec_u80_L1_corr_sett:', inco_prec_u80_L1_corr_sett)
print('inco_prec_u80_L1_inco_sett:', inco_prec_u80_L1_inco_sett)
print('inco_prec_u80_L1_inco_prec:', inco_prec_u80_L1_inco_prec)

print('corr_prec_u80_KLD_corr_prec:', corr_prec_u80_KLD_corr_prec)
print('corr_prec_u80_KLD_corr_sett:', corr_prec_u80_KLD_corr_sett)
print('inco_prec_u80_KLD_corr_sett:', inco_prec_u80_KLD_corr_sett)
print('inco_prec_u80_KLD_inco_sett:', inco_prec_u80_KLD_inco_sett)
print('inco_prec_u80_KLD_inco_prec:', inco_prec_u80_KLD_inco_prec)

# <<< from precise perspective <<<

# <<< SET-VALUED PREDICTIONS <<<