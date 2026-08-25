import sys
import os
import argparse
import numpy as np

from tqdm import tqdm
import pandas as pd

def cal_precise_prediction(all_precise_results, labels):
    # [SED L1 KLD]
    precise_prediction = []

    for i in range(len(all_precise_results)):
        u_alpha = 0

        for j in range(len(labels)):
            if all_precise_results[i][j] == labels[j]:
                u_alpha += 1
            else:
                u_alpha += m_metric[all_precise_results[i][j], labels[j]]

        precise_prediction.append(u_alpha)

    return precise_prediction

def find_a_range(n_class,num_out = 10):
    # Sort rewards descending by column
    m_sorted = np.sort(m_metric, axis=0)[::-1]
    a_list = []

    # iterate each cardinality k
    # k = 1 -> alpha vanishes
    for k in range(2, n_class+1):
        # For each cardinality k, find min s(k)
        s_k_min = np.min(1/np.sum(m_sorted[0:k,:], axis=0))        
        a_list.append((s_k_min * k**2 - 1)/(k - 1))
    
    a_upper = min(a_list)

    # return np.linspace(1.0, a_upper, num=10, endpoint=True), a_list
    return np.linspace(1.0, a_upper, num=num_out, endpoint=True)


def idc(p_star_sensitive, classes, n_class, alpha):
    # Here uses p_star vector multiply m_metric matrix
    # in case m_metric is anti-symetry
    # DONT DO IT HERE
    # p_star_m_metric = np.matmul(p_star, m_metric)
    # p_star_m_metric = p_star
    # Sort descending
    class_order = np.argsort(-p_star_sensitive)

    sum_tmp = 0
    eu_max = 0
    eu_tmp = 0
    k_top = 0

    # For each cardinality
    for k in range(1,n_class+1):
        # utility-discounted g
        g = (alpha / k) + ((1 - alpha) / (k**2))
        sum_tmp += p_star_sensitive[class_order[k-1]]
        eu_tmp = g * sum_tmp

        # less than or less than and equal
        if eu_tmp < eu_max:
            break
        else:
            eu_max = eu_tmp
            k_top = k
        
    return list(classes[class_order[0:k_top]])


def cal_set_value_prediction(all_p_star_SED_sensitive,
                                all_p_star_L1_sensitive,
                                all_p_star_KLD_sensitive,
                                classes, n_class, alpha):
    
    alpha_SED_predictions = []
    alpha_L1_predictions = []
    alpha_KLD_predictions = []

    for i in range(len(all_p_star_SED_sensitive)):
        # probability_set = output[:,i,:] #(100,10)

        # >>> Squared Euclidean distance >>>

        alpha_SED = idc(all_p_star_SED_sensitive[i], classes, n_class, alpha)
        alpha_SED_predictions.append(alpha_SED)

        # <<< Squared Euclidean distance <<<
        # >>> L1_distance >>>

        alpha_L1 = idc(all_p_star_L1_sensitive[i], classes, n_class, alpha)
        alpha_L1_predictions.append(alpha_L1)

        # <<< L1_distance <<<

        # >>> KL_divergence >>>

        alpha_KLD = idc(all_p_star_KLD_sensitive[i], classes, n_class, alpha)
        alpha_KLD_predictions.append(alpha_KLD)

        # <<< KL_divergence <<<
    # [[set_SED], [set_L1], [set_KLD]]
    return [alpha_SED_predictions, alpha_L1_predictions, alpha_KLD_predictions]

# >>> Remember to load the right files >>>
def main():

    args = parser.parse_args()
    ####
    k_folds = 3
    root_dir = ".."
    n_class = 10
    class_to_idx = {
        'T-shirt/top': 0,
        'Trouser': 1,
        'Pullover': 2,
        'Dress': 3,
        'Coat': 4,
        'Sandal': 5,
        'Shirt': 6,
        'Sneaker': 7,
        'Bag': 8,
        'Ankle boot': 9
    }
    #####

    data_name = "FMNIST-10-C"
    model_type = args.model_type
    train_type = args.train_type
    test_type = args.test_type
    
    save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/{model_type}/'
    print(f"model: {model_type} - train: {train_type} - test: {test_type}")

    
    classes = np.fromiter(class_to_idx.values(), dtype=int)

    alpha = find_a_range(n_class=10, num_out = 20)
    print(alpha)

    save_file = f'{save_dir}/set_output/{test_type}_set_value_prediction.txt'
    if not os.path.exists(f'{save_dir}/set_output/'):
        print(f"Path does not exist. Creating: {f'{save_dir}/set_output/'}")
        os.makedirs(f'{save_dir}set_output/')

    with open(save_file, "w") as f:
        f.write(f'list of alpha values: {alpha}')


    column = [ 'alpha', 'Precise_prediction_SED','Precise_prediction_L1D', 'Precise_prediction_KLD',
                'u65_SED_u65', 'u65_L1D_u65', 'u65_KLD_u65',
                'u65_SED_beta_size', 'u65_L1D_beta_size', 'u65_KLD_beta_size',
                'u65_SED_proportion', 'u65_L1D_proportion', 'u65_KLD_proportion',

                'rec_SED','rec_L1D','rec_KLD',
                'sin_SED','sin_L1D','sin_KLD',
                'set_SED','set_L1D','set_KLD',

                'corr_prec_u65_SED_corr_prec', 'corr_prec_u65_SED_corr_sett', 'inco_prec_u65_SED_corr_sett', 
                'inco_prec_u65_SED_inco_sett', 'inco_prec_u65_SED_inco_prec',

                'corr_prec_u65_L1D_corr_prec', 'corr_prec_u65_L1D_corr_sett', 'inco_prec_u65_L1D_corr_sett',
                'inco_prec_u65_L1D_inco_sett', 'inco_prec_u65_L1D_inco_prec',

                'corr_prec_u65_KLD_corr_prec', 'corr_prec_u65_KLD_corr_sett', 'inco_prec_u65_KLD_corr_sett',
                'inco_prec_u65_KLD_inco_sett', 'inco_prec_u65_KLD_inco_prec']
    csv_out = pd.DataFrame(columns= column)

    for alpha_i in tqdm(alpha):

        all_precise_prediction_kfold = []

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
        
        # The average recall
        rec_kfold = []
        # The proportion of correct singleton predictions
        sin_kfold = []
        # The proportion of correct set-valued predictions
        set_kfold = []
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

        #####################################################

        u65_prec_sett_kfold = []

        # Structure same as u65_prec_sett_kfold

        for fold in range(k_folds):
            # print(f'FOLD {fold}')
            # print('--------------------------------')
            
            labels = np.load(f'{save_dir}test_labels_{test_type}_fold_{fold}.npy')

            all_p_star_SED = np.load(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy')
            all_p_star_L1 =  np.load(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy')
            all_p_star_KLD = np.load(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy')

            all_precise_results = np.ndarray(shape=(3,len(labels)))

            all_p_star_SED_sensitive = np.matmul(all_p_star_SED, m_metric)
            all_precise_results[0] = np.argmax(all_p_star_SED_sensitive, axis=1)
            
            all_p_star_L1_sensitive = np.matmul(all_p_star_L1, m_metric)
            all_precise_results[1] = np.argmax(all_p_star_L1_sensitive, axis=1)

            all_p_star_KLD_sensitive = np.matmul(all_p_star_KLD, m_metric)
            all_precise_results[2] = np.argmax(all_p_star_KLD_sensitive, axis=1) 
            
            all_precise_prediction_kfold.append(cal_precise_prediction(all_precise_results.astype(int), labels))

            # [[alpha_SED], [alpha_L1], [alpha_KLD]]        
            all_set_predictions = cal_set_value_prediction(all_p_star_SED_sensitive,
                                                            all_p_star_L1_sensitive,
                                                            all_p_star_KLD_sensitive,
                                                            classes, n_class, alpha_i)

            # [[u65_u65, u65_u80, u80_u80, u80_u65], [L1], [KLD]]
            # all_set_results = np.ndarray(shape=(3,4))
            all_set_results = np.ndarray(shape=(3,4))

            ## [[SED], [L1], [KLD]]
            ## [[u65_u65_uni, u65_u65_mul, 
            ##   u80_u80_uni, u80_u80_mul], [L1], [KLD]]
            all_set_count = np.zeros([3, 4])
            # [[precise_u65_uni, precise_u65_mul,
            #   precise_u80_uni, precise_u80_mul], [L1], [KLD]]
            all_precise_count = np.zeros([3, 4])
            # [[SED], [L1], [KLD]]
            # [[u65_set_count, u65_element_count,
            #   u80_set_count, u80_element_count], [L1], [KLD]]
            avg_beta_size = np.zeros([3, 4])


            # >>> From set-valued perspective >>>
            # j=1=set_SED, j=2=set_L1, j=3=set_KLD
            for j in range(len(all_set_predictions)):
                u_alpha_total = 0
                for i in range(len(labels)):
                    u_alpha = 0
                    u_alpha_prediction = all_set_predictions[j][i]

                    # singleton prediction
                    if len(u_alpha_prediction) == 1:
                        #if u_alpha_prediction[0] == labels[i]:
                        u_alpha += m_metric[u_alpha_prediction[0], labels[i]]
                        all_set_count[j][0] += 1

                        if all_precise_results[j][i] == labels[i]:
                            all_precise_count[j][0] += 1
                    # set prediction
                    else:
                        all_set_count[j][1] += 1
                        avg_beta_size[j][0] += 1
                        avg_beta_size[j][1] += len(u_alpha_prediction)

                        for predicted_element in u_alpha_prediction:
                            u_alpha += m_metric[predicted_element, labels[i]] 

                        u_alpha = u_alpha * (alpha_i/len(u_alpha_prediction) + (1 - alpha_i)/(len(u_alpha_prediction)**2))
                        
                        if all_precise_results[j][i] == labels[i]:
                            all_precise_count[j][1] += 1
                    
                    u_alpha_total += u_alpha
                    if u_alpha < 0 or u_alpha > 1:
                        print(u_alpha)
                        sys.exit()
                
                all_set_results[j][0] = u_alpha_total


            all_set_results_kfold.append((all_set_results/len(labels))*100)
            # all_set_results = np.round((all_set_results/len(labels))*100, 2)
            all_set_count /= len(labels)
            all_precise_count /= len(labels) # recal

            u65_SED_beta_size = avg_beta_size[0][1]/avg_beta_size[0][0]
            # u80_SED_beta_size = avg_beta_size[0][3]/avg_beta_size[0][2]
            u65_L1_beta_size =  avg_beta_size[1][1]/avg_beta_size[1][0]
            # u80_L1_beta_size =  avg_beta_size[1][3]/avg_beta_size[1][2]
            u65_KLD_beta_size = avg_beta_size[2][1]/avg_beta_size[2][0]
            # u80_KLD_beta_size = avg_beta_size[2][3]/avg_beta_size[2][2]

            beta_size_kfold.append([u65_SED_beta_size,
                                    # u80_SED_beta_size,
                                    u65_L1_beta_size ,
                                    # u80_L1_beta_size ,
                                    u65_KLD_beta_size,
                                    # u80_KLD_beta_size
                                    ])

            u65_SED_proportion = (avg_beta_size[0][0]/len(labels))*100
            # u80_SED_proportion = (avg_beta_size[0][2]/len(labels))*100
            u65_L1_proportion =  (avg_beta_size[1][0]/len(labels))*100
            # u80_L1_proportion =  (avg_beta_size[1][2]/len(labels))*100
            u65_KLD_proportion = (avg_beta_size[2][0]/len(labels))*100
            # u80_KLD_proportion = (avg_beta_size[2][2]/len(labels))*100

            proportion_kfold.append([u65_SED_proportion,
                                    #  u80_SED_proportion,
                                    u65_L1_proportion ,
                                    #  u80_L1_proportion ,
                                    u65_KLD_proportion,
                                    #  u80_KLD_proportion
                                    ])



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

            # set_cor_inc_count = np.zeros([3, 10])
            set_cor_inc_count = np.zeros([3, 5])

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
                        # [[alpha_SED], [alpha_L1], [alpha_KLD]]
                        if len(all_set_predictions[j][i]) == 1:
                            set_cor_inc_count[j][0] += 1
                        else:
                            set_cor_inc_count[j][1] += 1
                                    
                    else: # precise + incorrect
                        precise_cor_inc_count[j][1] += 1
                        # u65
                        if len(all_set_predictions[j][i]) == 1:                
                            set_cor_inc_count[j][2] += 1
                        elif labels[i] not in all_set_predictions[j][i]:
                            set_cor_inc_count[j][4] += 1
                        else:
                            set_cor_inc_count[j][3] += 1
                        

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


            #new 
            # the average recall
            
            rec_SED = precise_cor_inc_count_labels[0][0] + set_cor_inc_count_labels[0][3]
            rec_L1D = precise_cor_inc_count_labels[1][0] + set_cor_inc_count_labels[1][3]
            rec_KLD = precise_cor_inc_count_labels[2][0] + set_cor_inc_count_labels[2][3]

            rec_kfold.append([rec_SED*100,rec_L1D*100,rec_KLD*100])
            # the proportion of corect singletion:

            sin_SED = set_cor_inc_count[0][0] / (set_cor_inc_count[0][0] + set_cor_inc_count[0][2]) 
            sin_L1D = set_cor_inc_count[1][0] / (set_cor_inc_count[1][0] + set_cor_inc_count[1][2]) 
            sin_KLD = set_cor_inc_count[2][0] / (set_cor_inc_count[2][0] + set_cor_inc_count[2][2]) 

            sin_kfold.append([sin_SED*100,sin_L1D*100,sin_KLD*100])
            # the proportion of corect singletion:

            set_SED = (set_cor_inc_count[0][1] + set_cor_inc_count[0][3]) / (set_cor_inc_count[0][1] + set_cor_inc_count[0][3] + set_cor_inc_count[0][4]) 
            set_L1D = (set_cor_inc_count[1][1] + set_cor_inc_count[1][3]) / (set_cor_inc_count[1][1] + set_cor_inc_count[1][3] + set_cor_inc_count[1][4])
            set_KLD = (set_cor_inc_count[2][1] + set_cor_inc_count[2][3]) / (set_cor_inc_count[2][1] + set_cor_inc_count[2][3] + set_cor_inc_count[2][4])

            set_kfold.append([set_SED*100,set_L1D*100,set_KLD*100])

        # >>> Precise prediction mean and std >>>

        all_precise_prediction_kfold = np.array(all_precise_prediction_kfold)/len(labels)*100
        precise_prediction_mean = np.round(np.mean(all_precise_prediction_kfold, axis=0),2)
        precise_prediction_std = np.round(np.std(all_precise_prediction_kfold, axis=0),2)

        # <<< Precise prediction mean and std <<<

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

        # <<< Precise and set correct and incorrect <<<

        rec_mean = np.round(np.mean(rec_kfold, axis=0), 2)
        rec_std = np.round(np.std(rec_kfold, axis=0), 2)

        ####

        sin_mean = np.round(np.mean(sin_kfold, axis=0), 2)
        sin_std = np.round(np.std(sin_kfold, axis=0), 2)

        ####

        set_mean = np.round(np.mean(set_kfold, axis=0), 2)
        set_std = np.round(np.std(set_kfold, axis=0), 2)

        #### save csv file
        temp_pandas = []
        temp_pandas.append(np.round(alpha_i, 5))
        temp_pandas.append(f"{precise_prediction_mean[0]} : {precise_prediction_std[0]}")
        temp_pandas.append(f"{precise_prediction_mean[1]} : {precise_prediction_std[1]}")
        temp_pandas.append(f"{precise_prediction_mean[2]} : {precise_prediction_std[2]}")
        # u65_SED_u65
        temp_pandas.append(f"{all_set_results_mean[0][0]} : {all_set_results_std[0][0]}")
        temp_pandas.append(f"{all_set_results_mean[1][0]} : {all_set_results_std[1][0]}")
        temp_pandas.append(f"{all_set_results_mean[2][0]} : {all_set_results_std[2][0]}")
        # u65_SED_beta_size
        temp_pandas.append(f"{beta_size_mean[0]} : {beta_size_std[0]}")
        temp_pandas.append(f"{beta_size_mean[1]} : {beta_size_std[1]}")
        temp_pandas.append(f"{beta_size_mean[2]} : {beta_size_std[2]}")
        # u65_SED_proportion
        temp_pandas.append(f"{proportion_mean[0]} : {proportion_std[0]}")
        temp_pandas.append(f"{proportion_mean[1]} : {proportion_std[1]}")
        temp_pandas.append(f"{proportion_mean[2]} : {proportion_std[2]}")
        # rec
        temp_pandas.append(f"{rec_mean[0]} : {rec_std[0]}")
        temp_pandas.append(f"{rec_mean[1]} : {rec_std[1]}")
        temp_pandas.append(f"{rec_mean[2]} : {rec_std[2]}")
        # sin
        temp_pandas.append(f"{sin_mean[0]} : {sin_std[0]}")
        temp_pandas.append(f"{sin_mean[1]} : {sin_std[1]}")
        temp_pandas.append(f"{sin_mean[2]} : {sin_std[2]}")
        # set
        temp_pandas.append(f"{set_mean[0]} : {set_std[0]}")
        temp_pandas.append(f"{set_mean[1]} : {set_std[1]}")
        temp_pandas.append(f"{set_mean[2]} : {set_std[2]}")
        # Precise and set correct and incorrec
        # # SED
        temp_pandas.append(f"{u65_prec_sett_mean[0][0]} : {u65_prec_sett_std[0][0]}")
        temp_pandas.append(f"{u65_prec_sett_mean[0][1]} : {u65_prec_sett_std[0][1]}")
        temp_pandas.append(f"{u65_prec_sett_mean[0][2]} : {u65_prec_sett_std[0][2]}")
        temp_pandas.append(f"{u65_prec_sett_mean[0][3]} : {u65_prec_sett_std[0][3]}")
        temp_pandas.append(f"{u65_prec_sett_mean[0][4]} : {u65_prec_sett_std[0][4]}")
        # # L1D
        temp_pandas.append(f"{u65_prec_sett_mean[1][0]} : {u65_prec_sett_std[1][0]}")
        temp_pandas.append(f"{u65_prec_sett_mean[1][1]} : {u65_prec_sett_std[1][1]}")
        temp_pandas.append(f"{u65_prec_sett_mean[1][2]} : {u65_prec_sett_std[1][2]}")
        temp_pandas.append(f"{u65_prec_sett_mean[1][3]} : {u65_prec_sett_std[1][3]}")
        temp_pandas.append(f"{u65_prec_sett_mean[1][4]} : {u65_prec_sett_std[1][4]}")
        # # KLD
        temp_pandas.append(f"{u65_prec_sett_mean[2][0]} : {u65_prec_sett_std[2][0]}")
        temp_pandas.append(f"{u65_prec_sett_mean[2][1]} : {u65_prec_sett_std[2][1]}")
        temp_pandas.append(f"{u65_prec_sett_mean[2][2]} : {u65_prec_sett_std[2][2]}")
        temp_pandas.append(f"{u65_prec_sett_mean[2][3]} : {u65_prec_sett_std[2][3]}")
        temp_pandas.append(f"{u65_prec_sett_mean[2][4]} : {u65_prec_sett_std[2][4]}")
        ####

        csv_out.loc[len(csv_out)] = temp_pandas

    csv_out = csv_out.T
    csv_out.to_csv(f'{save_dir}/set_output/{test_type}_set_value_prediction.csv', header=False)

    # <<< from precise perspective <<<

    # <<< SET-VALUED PREDICTIONS <<<

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                description="Optimal precise prediction (SED / L1 / KLD) for the dataset")
    parser.add_argument('--model_type', type=str, default='Bayes',
                        choices=['Bayes', 'Drop_out'],
                        help="Uncertainty model used to generate the outputs (default: Bayes)")
    parser.add_argument('--train_type', type=str, default='clean',
                        choices=['clean', 'noise'],
                        help="Training data condition of the loaded checkpoint (default: clean)")

    parser.add_argument('--test_type', type=str, default='clean',
                            choices=['clean', 'c'],
                            help="test data condition (default: clean , noise - c)")
        
    m_metric = np.array([
        [1.00, 0.25, 0.25, 0.25, 0.25, 0.00, 0.25, 0.00, 0.00, 0.00],
        [0.25, 1.00, 0.25, 0.25, 0.25, 0.00, 0.25, 0.00, 0.00, 0.00],
        [0.25, 0.25, 1.00, 0.25, 0.25, 0.00, 0.25, 0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25, 1.00, 0.25, 0.00, 0.25, 0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25, 0.25, 1.00, 0.00, 0.25, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25, 0.00, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.00, 1.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 1.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.00, 0.25, 0.00, 1.00]
    ])

    ########
    main()