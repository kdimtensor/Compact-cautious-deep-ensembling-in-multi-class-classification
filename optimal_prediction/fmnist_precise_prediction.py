import os
import numpy as np
from scipy.optimize import minimize
import argparse

# # >>> PRECISE PREDICTIONS >>>
def evaluate(output, labels, n_class, save_dir, fold, test_type):        
    all_p_star_L1 = np.zeros([len(labels), n_class])
    all_p_star_KLD = np.zeros([len(labels), n_class])
    eval_results = []

    # >>> Squared Euclidean distance >>>

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

        all_p_star_KLD[i] = p_star_KLD

        # <<< KL_divergence <<<     

    all_p_star_SED_sensitive = np.matmul(all_p_star_SED, m_metric)
    predictions_SED = np.argmax(all_p_star_SED_sensitive, axis=1)
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_SED = (predictions_SED == labels).mean() * 100
    
    all_p_star_L1_sensitive = np.matmul(all_p_star_L1, m_metric)
    predictions_L1 = np.argmax(all_p_star_L1_sensitive, axis=1)
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_L1 = (predictions_L1 == labels).mean() * 100

    all_p_star_KLD_sensitive = np.matmul(all_p_star_KLD, m_metric)
    predictions_KLD = np.argmax(all_p_star_KLD_sensitive, axis=1)    
    # This is 0/1 accuracy not reward sensitive accuracy
    # acc_KLD = (predictions_KLD == labels).mean() * 100

    # >>> Save test outputs >>>

    np.save(f'{save_dir}all_p_star_SED_{test_type}_fold_{fold}.npy', all_p_star_SED)
    np.save(f'{save_dir}predictions_SED_{test_type}_fold_{fold}.npy', predictions_SED)

    np.save(f'{save_dir}all_p_star_L1_{test_type}_fold_{fold}.npy', all_p_star_L1)
    np.save(f'{save_dir}predictions_L1_{test_type}_fold_{fold}.npy', predictions_L1)

    np.save(f'{save_dir}all_p_star_KLD_{test_type}_fold_{fold}.npy', all_p_star_KLD)
    np.save(f'{save_dir}predictions_KLD_{test_type}_fold_{fold}.npy', predictions_KLD)

    acc_reward_sensitive_SED = 0
    acc_reward_sensitive_L1D = 0
    acc_reward_sensitive_KLD = 0
    for idx in range(len(all_p_star_SED)):
        acc_reward_sensitive_SED += m_metric[predictions_SED[idx], labels[idx]]
        acc_reward_sensitive_L1D += m_metric[predictions_L1[idx], labels[idx]]
        acc_reward_sensitive_KLD += m_metric[predictions_KLD[idx], labels[idx]]
        
    eval_results.append(acc_reward_sensitive_SED)
    eval_results.append(acc_reward_sensitive_L1D)
    eval_results.append(acc_reward_sensitive_KLD)
    return eval_results

def main():

    args = parser.parse_args()
    k_folds = 3
    root_dir = ".."
    n_class = 10
    #########
    data_name = "FMNIST-10-C"
    #########

    model_type = args.model_type
    train_type = args.train_type
    
    print(train_type)
    save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/{model_type}/'
    out_dir = save_dir
    print(out_dir)

    if not os.path.exists(out_dir):
            print(f"Path does not exist. Creating: ")
            os.makedirs(out_dir)

    kfold_results_c = []
    kfold_results_clean = []

    for fold in range(k_folds):
        print(f'FOLD {fold}')
        print('--------------------------------')
            
        # output shape ([100, 10000, 10])
        output_c = np.load(f'{save_dir}tensor_output_c_fold_{fold}.npy')
        labels_c = np.load(f'{save_dir}test_labels_c_fold_{fold}.npy')
        output_clean = np.load(f'{save_dir}tensor_output_clean_fold_{fold}.npy')
        labels_clean = np.load(f'{save_dir}test_labels_clean_fold_{fold}.npy')
        
        kfold_results_c.append(evaluate(output_c, labels_c, n_class, out_dir, fold, 'c'))
        kfold_results_clean.append(evaluate(output_clean, labels_clean, n_class, out_dir, fold, 'clean'))

    print(np.round(kfold_results_c, 2))
    print('-'*8)
    print(np.round(kfold_results_clean, 2))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Optimal precise prediction (SED / L1 / KLD) for the dataset")
    parser.add_argument('--model_type', type=str, default='Bayes',
                        choices=['Bayes', 'Drop_out'],
                        help="Uncertainty model used to generate the outputs (default: Bayes)")
    parser.add_argument('--train_type', type=str, default='clean',
                        choices=['clean', 'noise'],
                        help="Training data condition of the loaded checkpoint (default: clean)")


    m_metric = np.array([
        [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
    ])

    ########################
    main()
