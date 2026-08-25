import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import copy

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Subset
import torchvision.transforms as transforms
import bayesian_torch.models.bayesian.resnet_variational as resnet
from bayesian_torch.models.bayesian.resnet_variational import ResNet,BasicBlock
import numpy as np

from data_preprocess.leaf_c import LEAFC, LEAFCLEAN
from tqdm import tqdm
# Add parent directory to path

import saved_loss

def load_device():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    return device

def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    #dir() or vars()
    model.train()
    loss = 0

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        output, kl = model(input_var)

        cross_entropy_loss = criterion(output, target_var)
        scaled_kl = kl / args.batch_size 
        #ELBO loss
        loss = cross_entropy_loss + scaled_kl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    print('Train Epoch: {} \tLoss: {:.6f} \t Acc: {:.3f}'.format(
        epoch, losses.avg,top1.avg))
    
    return losses.avg, top1.avg


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.to(device)
            input_var = input.to(device)
            target_var = target
            output, kl = model(input_var)

            cross_entropy_loss = criterion(output, target_var)
            scaled_kl = kl / args.batch_size 
            #ELBO loss
            loss = cross_entropy_loss + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time

        print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
            losses.avg, top1.avg))

    return losses.avg, top1.avg

def evaluate(args, model, val_loader, fold, dataset_type,save_dir):
    # --batch-size=$batch_size 
    # --num_monte_carlo=$num_monte_carlo
    args.num_monte_carlo = 100
    output_list = []
    labels_list = []
    eval_results = []
    model.eval()
    with torch.no_grad():

        for data, target in tqdm(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(args.num_monte_carlo):
                output, _ = model.forward(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_list.append(output_)
            labels_list.append(target)

        output = torch.cat(output_list, dim=1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)      
        # labels_onehot <class 'numpy.ndarray'> (10000, 10)
        labels_onehot = torch.nn.functional.one_hot(labels).data.cpu().numpy()

        # >>> Original Squared Euclidean distance >>>

        pred_mean = output.mean(dim=0).data.cpu().numpy()
        Y_pred = np.argmax(pred_mean, axis=1)
        acc_SED = (Y_pred == labels.data.cpu().numpy()).mean() * 100
        eval_results.append(acc_SED)
        print('Test accuracy:', acc_SED)

        np.save(save_dir + f'tensor_output_{dataset_type}_fold_{fold}.npy', output.data.cpu().numpy())
        np.save(save_dir + f'test_labels_{dataset_type}_fold_{fold}.npy', labels.data.cpu().numpy())        
        np.save(save_dir + f'test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)

    return
def main():

    global args, best_prec1_noise,best_prec1_clean, device
    args = parser.parse_args()
    # >>> K_fold cross validation config >>>
    k_folds = args.k_fold
    train_type = args.train_type
    torch.manual_seed(args.seed)
    # <<< K_fold cross validation config <<<
    #############
    
    print(train_type)
    root_dir = ".."
    data_name = "LEAF-C"
    print(data_name)
    device = load_device()
    #############
    print(args.batch_size)

    args.save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/Bayes/model/'
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = ResNet(BasicBlock, [3, 3, 3],num_classes=39)

    model = model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    criterion = nn.CrossEntropyLoss().to(device)

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    ####
    print(root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####
    if args.mode == 'train':

        train_data = LEAFC(
            root= root_dir + f'/data/{data_name}/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            )

        train_data_clean = LEAFCLEAN(
            root=root_dir + f'/data/{data_name}/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, 4),
                transforms.ToTensor(),
                normalize,
            ]),)

        dataset_all_c = train_data

        dataset_all_clean = train_data_clean

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')
            # save_filename = f'bayesian_{args.arch}_cifar_fold_{fold}.pth'
            train_ids = np.load(root_dir + f'/data/{data_name}/k_fold_id/train_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/{data_name}/k_fold_id/test_ids_fold_{fold}.npy')
            
            if train_type == "noise":
                print("train noise")
                train_subset = Subset(dataset_all_c, train_ids)

            elif train_type == "clean":
                print("train clean")
                train_subset = Subset(dataset_all_clean, train_ids)
            else:
                print("errors")

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.workers,shuffle = True)

            val_subset = Subset(dataset_all_c, test_ids)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.workers)

            val_subset_clean = Subset(dataset_all_clean, test_ids)
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            # Keep initial parameters, try copy.deepcopy(model)
            model_fold = copy.deepcopy(model)
            model_fold = model_fold.to(device)
            # Reset this or the save_checkpoint doesnt want to work 
            best_prec1_noise = 0
            best_prec1_clean = 0
            ####
            output_loss = saved_loss.loss_out(args.save_dir + f'bayesian_{args.arch}_leaf_fold_{fold}.csv')
            ####
            for epoch in range(args.start_epoch, args.epochs):
                
                lr = args.lr
                if (epoch >= 80 and epoch < 120):
                    lr = 0.1 * args.lr
                elif (epoch >= 120 and epoch < 160):
                    lr = 0.01 * args.lr
                elif (epoch >= 160 and epoch < 180):
                    lr = 0.001 * args.lr
                elif (epoch >= 180):
                    lr = 0.0005 * args.lr

                optimizer = torch.optim.Adam(model_fold.parameters(), lr)

                # train for one epoch
                print('current lr {:.5e}'.format(lr))
                train_loss, train_acc = train(args, train_loader, model_fold, criterion, optimizer, epoch)

                val_loss_noise, prec1_noise = validate(args, val_loader, model_fold, criterion, epoch)
                
                val_loss_clean, prec1_clean = validate(args, val_loader_clean, model_fold, criterion, epoch)

                output_loss.update_acc_train(train_acc)
                output_loss.update_acc_val_noise(prec1_noise)
                output_loss.update_loss_train(train_loss)
                output_loss.update_loss_val_noise(val_loss_noise)
                output_loss.update_acc_val_clean(prec1_clean)
                output_loss.update_loss_val_clean(val_loss_clean)

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
                            args.save_dir, f'bayesian_{args.arch}_leaf_noise_fold_{fold}.pth'))

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
                            args.save_dir, f'bayesian_{args.arch}_leaf_clean_fold_{fold}.pth'))
                            
                ############
            output_loss.save_csv()


    elif args.mode == 'test':
        
        save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/Bayes/'

        train_data = LEAFC(
            root= root_dir + f'/data/{data_name}/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            )

        train_data_clean = LEAFCLEAN(
            root=root_dir + f'/data/{data_name}/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),)

        dataset_all_c = train_data
        dataset_all_clean = train_data_clean

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')
                        
            # test_ids = np.load(f'{save_dir}test_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/{data_name}/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)           
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=args.batch_size, num_workers=args.workers)
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            checkpoint_file = f'{save_dir}model/bayesian_{args.arch}_leaf_{train_type}_fold_{fold}.pth'
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            evaluate(args, model, val_loader_c, fold, 'c',save_dir)
            evaluate(args, model, val_loader_clean, fold, 'clean',save_dir)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


if __name__ == '__main__':

    model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

    print(model_names)

    parser = argparse.ArgumentParser(description='LEAF')
    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet20)')
    parser.add_argument('-j',
                        '--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=128,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.001,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')

    parser.add_argument('--print-freq',
                        '-p',
                        default=50,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 20)')

    parser.add_argument('--save-dir',
                        dest='save_dir',
                        help='The directory used to save the trained models',
                        default='/checkpoint/bayesian',
                        type=str)
    parser.add_argument('--mode', type=str, required=True, help='train | test')
    parser.add_argument(
        '--num_monte_carlo',
        type=int,
        default=20,
        metavar='N',
        help='number of Monte Carlo samples to be drawn during inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of Monte Carlo runs during training')

    parser.add_argument('--seed',
                            type=int,
                            default=42,
                            metavar='S',
                            help='random seed (default: 42)')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=32,
        # metavar='N',
        help='use tensorboard for logging and visualization of training progress')

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
    ####################
    main()
