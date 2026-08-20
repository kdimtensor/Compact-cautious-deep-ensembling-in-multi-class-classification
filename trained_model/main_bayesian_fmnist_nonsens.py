from __future__ import print_function
import os
import sys
import json
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, Subset
import numpy as np
import scipy
from scipy.special import softmax
import bayesian_torch.models.bayesian.simple_cnn_variational as simple_cnn
from scipy.optimize import minimize
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE

from sklearn.model_selection import KFold
from fmnist_c import FMNISTC, FMNISTCLEAN

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import saved_loss

import matplotlib.pyplot as plt
import torchvision

#########
root_dir = ""
#########

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(args, model, device, train_loader, optimizer, epoch, tb_writer=None):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(data)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        nll_loss = F.nll_loss(output, target)
        #ELBO loss
        loss = nll_loss + (kl / args.batch_size)

        
        
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        loss.backward()
        optimizer.step()


    print('Train Epoch: {} \tLoss: {:.6f} \t Acc: {:.3f}'.format(
        epoch, losses.avg,top1.avg))

        # if tb_writer is not None:
        #     tb_writer.add_scalar('train/loss', loss.item(), epoch)
        #     tb_writer.flush()

    return losses.avg, top1.avg


def validate(args, model, device, test_loader, epoch, tb_writer=None):
    model.eval()
    # test_loss = 0
    # correct = 0
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output, kl = model(data)
            loss = F.nll_loss(output, target) + (kl / args.batch_size)  # sum up batch loss
            # pred = output.argmax(
            #     dim=1,
            #     keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            output = output.float()
            prec1 = accuracy(output.data, target)[0]
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
            losses.avg, top1.avg))

    # if tb_writer is not None:
    #     tb_writer.add_scalar('val/loss', test_loss, epoch)
    #     tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)
    #     tb_writer.flush()

    return losses.avg, top1.avg

def evaluate(args, model, device, test_loader, save_dir, fold, dataset_type):
    args.num_monte_carlo = 100
    print(args.num_monte_carlo)
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    predictions_L1 = []
    predictions_KLD = []
    all_p_star_L1 = []
    all_p_star_KLD = []
    eval_results = []
    model.eval()
    with torch.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            for mc_run in range(args.num_monte_carlo):                
                # output, _ = model.forward(data)
                output, _ = model(data)
                #get probabilities from log-prob
                pred_probs = torch.exp(output)
                pred_probs_mc.append(pred_probs.cpu().data.numpy())
                
        labels = target.cpu().data.numpy()
        # labels_onehot <class 'numpy.ndarray'> (10000, 10)
        labels_onehot = torch.nn.functional.one_hot(target).data.cpu().numpy()        

        # >>> Original Squared Euclidean distance >>>

        output = np.array(pred_probs_mc)
        pred_mean = np.mean(output, axis=0)        
        Y_pred = np.argmax(pred_mean, axis=1)

        acc_SED = (Y_pred == labels).mean() * 100
        eval_results.append(acc_SED)
        print('SED_acc:', acc_SED)

        np.save(f'{save_dir}tensor_output_{dataset_type}_fold_{fold}.npy', output)
        # np.save(f'{save_dir}all_p_star_SED_{dataset_type}_fold_{fold}.npy', pred_mean)
        # np.save(f'{save_dir}predictions_SED_{dataset_type}_fold_{fold}.npy', Y_pred)
        np.save(f'{save_dir}test_labels_{dataset_type}_fold_{fold}.npy', labels)        
        np.save(f'{save_dir}test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)

        # <<< Original Squared Euclidean distance <<<

        # for i in range(len(labels)):
        #     probability_set = output[:,i,:] #(50,10)

        #     # >>> Distances >>>
        #     # >>> L1_distance >>>

        #     def L1_distance(p_star_L1, probability_set):
        #         return np.sum(np.abs(probability_set - p_star_L1))

        #     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

        #     result = minimize(L1_distance, np.ones(10)/10, args=(probability_set,), constraints=constraints)
            
        #     p_star_L1 = result.x

        #     # all_p_star_L1 shape [10000, 10]
        #     all_p_star_L1.append(p_star_L1)            

        #     # <<< L1_distance <<<

        #     # >>> KL_divergence >>>

        #     def KL_divergence(p_star_KLD, probability_set):
        #         epsilon = 1e-10
        #         p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)
        #         probability_set = np.where(probability_set < epsilon, epsilon, probability_set)
        #         return np.sum(p_star_KLD * np.log(p_star_KLD/probability_set))
        #         # return np.sum(probability_set * np.log(probability_set/p_star_KLD))

        #     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        #                            {'type': 'ineq', 'fun': lambda x: x}]
            
        #     result = minimize(KL_divergence, np.ones(10)/10, args=(probability_set,), constraints=constraints)
        #     p_star_KLD = result.x
        #     epsilon = 1e-10
        #     p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)

        #     # all_p_star_KLD shape [10000, 10]
        #     all_p_star_KLD.append(p_star_KLD)

        #     # <<< KL_divergence <<<            
        
      
        # predictions_L1 = np.argmax(all_p_star_L1, axis=1)
        # predictions_KLD = np.argmax(all_p_star_KLD, axis=1)
        # acc_L1 = (predictions_L1 == labels).mean() * 100
        # acc_KLD = (predictions_KLD == labels).mean() * 100
        # eval_results.append(acc_L1)
        # eval_results.append(acc_KLD)

        # # >>> Save test outputs >>>

        # # np.save(f'{save_dir}all_p_star_L1_{dataset_type}_fold_{fold}.npy', all_p_star_L1)
        # # np.save(f'{save_dir}predictions_L1_{dataset_type}_fold_{fold}.npy', predictions_L1)
        # # np.save(f'{save_dir}all_p_star_KLD_{dataset_type}_fold_{fold}.npy', all_p_star_KLD)
        # # np.save(f'{save_dir}predictions_KLD_{dataset_type}_fold_{fold}.npy', predictions_KLD)
        
        # # <<< Save test output <<<

        # print('L1_acc:', acc_L1)
        # print('KLD_acc:', acc_KLD)

    return eval_results

kfold_results_c = []
kfold_results_clean = []

best_prec1_noise = 0
best_prec1_clean = 0

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')

    parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='CNNs',
                    # choices=model_names,
                    # help='model architecture: ' + ' | '.join(model_names) +
                    # ' (default: CNNs)'
                    )
                
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=10000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=90,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./checkpoint/bayesian')
    parser.add_argument('--mode', type=str, required=True, help='train | test')
    parser.add_argument(
        '--num_monte_carlo',
        type=int,
        default=1,
        metavar='N',
        help='number of Monte Carlo samples to be drawn for inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of Monte Carlo runs during training')
    parser.add_argument(
        '--tensorboard',
        action="store_true",
        help=
        'use tensorboard for logging and visualization of training progress')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/mnist/bayesian',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')
    
    ###########################################
    # train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
    #     '../data',
    #     train=True,
    #     download=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307, ), (0.3081, ))
    #     ])),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     **kwargs)
    
    # test_data = datasets.FashionMNIST(
    #     '../data', 
    #     train=False, 
    #     transform=transforms.Compose([
    #         transforms.ToTensor(), 
    #         transforms.Normalize((0.1307, ), (0.3081, ))
    #     ]))   
    
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=args.test_batch_size,
    #                                           shuffle=False,
    #                                           **kwargs)

    # with open('./fmnist_test_data_classes.txt', 'w') as f:
    #     f.write(json.dumps(test_data.classes))
    
    # with open('./fmnist_test_data_class_to_idx.txt', 'w') as f:
    #     f.write(json.dumps(test_data.class_to_idx))

    ######################

    k_folds = 3

    train_type = "noise"
    print(train_type)
    global args, best_prec1_noise,best_prec1_clean
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
   
    tb_writer = None

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # if args.tensorboard:

    #     logger_dir = os.path.join(args.log_dir, 'tb_logger')
    #     print("yee")
    #     if not os.path.exists(logger_dir):
    #         os.makedirs(logger_dir)

    #     tb_writer = SummaryWriter(logger_dir)


    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    model = simple_cnn.SCNN()
    # model = model.to(device)    
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    print(args.mode)

    #####################
    train_data_c = FMNISTC(
        root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        download=False)

    test_data_c = FMNISTC(
        root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        download=False)


    train_data_clean = FMNISTCLEAN(
        root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        download=False)

    test_data_clean = FMNISTCLEAN(
        root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        download=False)

    dataset_all_c = ConcatDataset([train_data_c, test_data_c])
    dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])

    kfold = KFold(n_splits=k_folds, shuffle=False)  

    ####
    print(root_dir + f'/data/FMNIST-10-C/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####

    if args.mode == 'train':

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            args.save_dir = root_dir + f'/data/FMNIST-10-C/gaussian_{train_type}_fold/Bayes/'
            os.makedirs(args.save_dir + "model", exist_ok=True)
            # save_filename = f'bayesian_scnn_fmnist_fold_{fold}.pth'
            train_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/train_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/test_ids_fold_{fold}.npy')

            # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # train_loader = torch.utils.data.DataLoader(dataset_all,
            #                                         batch_size=args.batch_size,
            #                                         num_workers=args.workers, sampler=train_subsampler)
            # val_loader = torch.utils.data.DataLoader(dataset_all, 
            #                                         batch_size=args.batch_size, num_workers=args.workers,sampler=test_subsampler)
            
            if train_type == "noise":
                print("train noise")
                train_subset = Subset(dataset_all_c, train_ids)
                

            else:
                print("train clean")
                train_subset = Subset(dataset_all_clean, train_ids)


            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.workers,shuffle = True)

            val_subset = Subset(dataset_all_c, test_ids)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.workers)

            val_subset_clean = Subset(dataset_all_clean, test_ids)
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            # Keep initial parameters, try copy.deepcopy(model)
            model_fold = copy.deepcopy(model)

            best_prec1_noise = 0
            best_prec1_clean = 0

            ####
            output_loss = saved_loss.loss_out(args.save_dir + f'bayesian_{args.arch}_fmnist_fold_{fold}.csv')
            ####
            for epoch in range(1, args.epochs + 1):
                if (epoch < args.epochs / 3):
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr)
                elif (epoch < args.epochs*2/3):
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr/10)
                else:
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr /100)

                scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
                
                train_loss, train_acc = train(args, model_fold, device, train_loader, optimizer, epoch, tb_writer)
                
                val_loss_noise, prec1_noise = validate(args, model_fold, device, val_loader, epoch, tb_writer)
                
                val_loss_clean, prec1_clean = validate(args, model_fold, device, val_loader_clean, epoch, tb_writer)
                
                output_loss.update_acc_train(train_acc)
                output_loss.update_acc_val_noise(prec1_noise)
                output_loss.update_loss_train(train_loss)
                output_loss.update_loss_val_noise(val_loss_noise)
                output_loss.update_acc_val_clean(prec1_clean)
                output_loss.update_loss_val_clean(val_loss_clean)

                scheduler.step()

                ##########

                is_best_noise = prec1_noise > best_prec1_noise
                best_prec1_noise = max(prec1_noise, best_prec1_noise)

                if is_best_noise:
                    
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model_fold.state_dict(),
                            'best_prec1': best_prec1_noise,
                        },
                        is_best_noise,
                        filename=os.path.join(
                            args.save_dir, f'model/bayesian_{args.arch}_fmnist_noise_fold_{fold}.pth'))

                ############

                ##########

                is_best_clean = prec1_clean > best_prec1_clean
                best_prec1_clean = max(prec1_clean, best_prec1_clean)

                if is_best_clean:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model_fold.state_dict(),
                            'best_prec1': best_prec1_clean,
                        },
                        is_best_clean,
                        filename=os.path.join(
                            args.save_dir, f'model/bayesian_{args.arch}_fmnist_clean_fold_{fold}.pth'))
                            
                ############
            output_loss.save_csv()

    elif args.mode == 'test':

        
        save_dir = root_dir + f'/data/FMNIST-10-C/gaussian_{train_type}_fold/Bayes/'

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            test_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=len(test_ids))
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=len(test_ids))

            checkpoint_file = f'{save_dir}/model/bayesian_CNNs_fmnist_{train_type}_fold_{fold}.pth'
            
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            # Please check the evaluate function for save path
            evaluate(args, model, device, val_loader_c, save_dir, fold, 'c')
            evaluate(args, model, device, val_loader_clean, save_dir, fold, 'clean')

        # print(kfold_results_c)
        # print('-'*11)
        # print(kfold_results_clean)        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

if __name__ == '__main__':
    main()


