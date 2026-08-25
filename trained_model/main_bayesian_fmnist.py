import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import copy
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, Subset
import numpy as np
import bayesian_torch.models.bayesian.simple_cnn_variational as simple_cnn

from data_preprocess.fmnist_c import FMNISTC, FMNISTCLEAN

# Add parent directory to path

import saved_loss

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

    return losses.avg, top1.avg


def validate(args, model, device, test_loader, epoch, tb_writer=None):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output, kl = model(data)
            loss = F.nll_loss(output, target) + (kl / args.batch_size)  # sum up batch loss
            output = output.float()
            prec1 = accuracy(output.data, target)[0]
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
            losses.avg, top1.avg))

    return losses.avg, top1.avg

def evaluate(args, model, device, test_loader, save_dir, fold, dataset_type):
    print(args.num_monte_carlo)
    pred_probs_mc = []
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
        np.save(f'{save_dir}test_labels_{dataset_type}_fold_{fold}.npy', labels)        
        np.save(f'{save_dir}test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)

    return

def main():
    # Training settings
    
    ######################
    global best_prec1_noise,best_prec1_clean
    args = parser.parse_args()
    k_folds = args.k_fold
    train_type = args.train_type
    torch.manual_seed(args.seed)
    #####
    root_dir ="../"
    print(train_type)
    args.save_dir = root_dir + f'/data/FMNIST-10-C/gaussian_{train_type}_fold/Bayes/'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = simple_cnn.SCNN()
    # model = model.to(device)    
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    ####
    print(root_dir + f'/data/FMNIST-10-C/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####

    if args.mode == 'train':
        #####################
        train_data_c = FMNISTC(
            root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            download=False)

        test_data_c = FMNISTC(
            root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
            train=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            download=False)


        train_data_clean = FMNISTCLEAN(
            root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            download=False)

        test_data_clean = FMNISTCLEAN(
            root=root_dir +'/data/FMNIST-10-C/gaussian_noise_6/',
            train=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            download=False)

        dataset_all_c = ConcatDataset([train_data_c, test_data_c])
        dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            os.makedirs(args.save_dir + "model", exist_ok=True)
            # save_filename = f'bayesian_scnn_fmnist_fold_{fold}.pth'
            train_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/train_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/test_ids_fold_{fold}.npy')

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
                
                train_loss, train_acc = train(args, model_fold, device, train_loader, optimizer, epoch)
                
                val_loss_noise, prec1_noise = validate(args, model_fold, device, val_loader, epoch)
                
                val_loss_clean, prec1_clean = validate(args, model_fold, device, val_loader_clean, epoch)
                
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

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            test_ids = np.load(root_dir + f'/data/FMNIST-10-C/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=len(test_ids))
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=len(test_ids))

            checkpoint_file = f'{args.save_dir}/model/bayesian_CNNs_fmnist_{train_type}_fold_{fold}.pth'
            
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            # Please check the evaluate function for save path
            evaluate(args, model, device, val_loader_c, args.save_dir, fold, 'c')
            evaluate(args, model, device, val_loader_clean, args.save_dir, fold, 'clean')

        # print(kfold_results_c)
        # print('-'*11)
        # print(kfold_results_clean)        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

if __name__ == '__main__':

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
                        default=42,
                        metavar='S',
                        help='random seed (default: 42)')
    
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

    parser.add_argument('--train-type',
                            dest='train_type',
                            type=str,
                            default='clean',
                            choices=['noise', 'clean'],
                            help='type of training data: noise | clean (default: clean)')
    parser.add_argument('--k-fold',
                        dest='k_fold',
                        type=int,
                        default=3,
                        metavar='N',
                        help='number of folds for cross validation (default: 3)')
    
    ###################
    main()


