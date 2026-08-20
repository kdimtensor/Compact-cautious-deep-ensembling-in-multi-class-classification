import argparse
import os
import shutil
import time
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.bayesian.resnet_variational as resnet
import numpy as np
from scipy.optimize import minimize
from pycalib.metrics import binary_ECE, binary_MCE, classwise_ECE, classwise_MCE, conf_ECE, conf_MCE

from sklearn.model_selection import KFold

from cifar10_c import CIFAR10C, CIFAR10CLEAN

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import saved_loss

import matplotlib.pyplot as plt
import torchvision

#########
root_dir = ""
#########

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

print(model_names)
# len_trainset = 50000
# len_testset = 10000

parser = argparse.ArgumentParser(description='CIFAR10')
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
                    default=150,
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
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 20)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default=root_dir + '/checkpoint/bayesian',
                    type=str)
parser.add_argument(
    '--save-every',
    dest='save_every',
    help='Saves checkpoints at every specified number of epochs',
    type=int,
    default=10)
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
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/cifar/bayesian',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')

best_prec1_noise = 0
best_prec1_clean = 0


def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    """

    if (str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data

    elif (str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors
        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

kfold_results_c = []
kfold_results_clean = []

def main():
    # >>> K_fold cross validation config >>>
    k_folds = 3
    # <<< K_fold cross validation config <<<

    #############
    train_type = "noise"
    print(train_type)
    #############
    global args, best_prec1_noise,best_prec1_clean
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.evaluate, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    # if args.tensorboard:
    #     logger_dir = os.path.join(args.log_dir, 'tb_logger')
    #     if not os.path.exists(logger_dir):
    #         os.makedirs(logger_dir)
    #     tb_writer = SummaryWriter(logger_dir)

    ####################
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    # train_data = CIFAR10C(
    #     root= root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
    #     train=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     download=False,
    #     )

    # test_data = CIFAR10C(
    #     root=root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
    #     train=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     download=False,
    #     )

    # train_data_clean = CIFAR10CLEAN(
    #     root=root_dir +'/data/CIFAR-10-C/gaussian_noise_6/',
    #     train=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     download=False)

    # test_data_clean = CIFAR10CLEAN(
    #     root=root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
    #     train=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     download=False)

    # dataset_all_c = ConcatDataset([train_data, test_data])
    # dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])
    # kfold = KFold(n_splits=k_folds, shuffle=False)    
    ######################
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    if args.half:
        model.half()
        criterion.half()

    if args.arch in ['resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    ####
    print(root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####
    if args.mode == 'train':
        
        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            args.save_dir = root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/Bayes/'
            os.makedirs(args.save_dir,exist_ok=True)
            os.makedirs(args.save_dir + "model",exist_ok=True)
            # save_filename = f'bayesian_{args.arch}_cifar_fold_{fold}.pth'
            train_ids = np.load(root_dir + f'/data/CIFAR-10-C/k_fold_id/train_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/CIFAR-10-C/k_fold_id/test_ids_fold_{fold}.npy')

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
                

            elif train_type == "clean":
                print("train clean")
                train_subset = Subset(dataset_all_clean, train_ids)
            else:
                print("errors")

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.workers, shuffle = True )

            val_subset = Subset(dataset_all_c, test_ids)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.workers)

            val_subset_clean = Subset(dataset_all_clean, test_ids)
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            # Keep initial parameters, try copy.deepcopy(model)
            model_fold = copy.deepcopy(model)
            # Reset this or the save_checkpoint doesnt want to work 
            best_prec1_noise = 0
            best_prec1_clean = 0
            ####
            output_loss = saved_loss.loss_out(args.save_dir + f'bayesian_{args.arch}_cifar_fold_{fold}.csv')
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
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train_loss, train_acc = train(args, train_loader, model_fold, criterion, optimizer, epoch, tb_writer)

                val_loss_noise, prec1_noise = validate(args, val_loader, model_fold, criterion, epoch,
                                tb_writer)
                
                val_loss_clean, prec1_clean = validate(args, val_loader_clean, model_fold, criterion, epoch,
                                tb_writer)

                print(train_acc)
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
                            args.save_dir, f'model/bayesian_{args.arch}_cifar_noise_fold_{fold}.pth'))

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
                            args.save_dir, f'model/bayesian_{args.arch}_cifar_clean_fold_{fold}.pth'))
                            
                ############
            output_loss.save_csv()


    elif args.mode == 'test':
        ######
        train_data = CIFAR10C(
            root= root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=False,
            )

        test_data = CIFAR10C(
            root=root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=False,
            )

        train_data_clean = CIFAR10CLEAN(
            root=root_dir +'/data/CIFAR-10-C/gaussian_noise_6/',
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=False)

        test_data_clean = CIFAR10CLEAN(
            root=root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=False)
        #########
        dataset_all_c = ConcatDataset([train_data, test_data])
        dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])
        kfold = KFold(n_splits=k_folds, shuffle=False)    
        
        save_dir = root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/Bayes/'
        print(torch.cuda.is_available())

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')
                        
            # test_ids = np.load(f'{save_dir}test_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/CIFAR-10-C/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)           
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=args.batch_size, num_workers=args.workers)
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            checkpoint_file = f'{save_dir}model/bayesian_{args.arch}_cifar_{train_type}_fold_{fold}.pth'
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            
            kfold_results_c.append(evaluate(args, model, val_loader_c, fold, 'c',save_dir))
            kfold_results_clean.append(evaluate(args, model, val_loader_clean, fold, 'clean',save_dir))
        
        print(np.round(kfold_results_c, 2))
        print('-'*8)
        print(np.round(kfold_results_clean, 2))


def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    #dir() or vars()
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        
        # from PIL import Image
        # from torchvision.transforms import ToPILImage
        # from IPython.display import display

        # def imshow(img):
        #     img = img / 2 + 0.5 # Unnormalize
        #     plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0))
        #     plt.savefig("train_clean.png")

        # array = input[2]
        # # to_pil = ToPILImage()
        # # image = to_pil(array)
        # imshow(array)

        # return 0,0

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
            target_var = target
        else:
            target = target.cpu()
            input_var = input.cpu()
            target_var = target

        if args.half:
            input_var = input_var.half()

        # compute output
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(input_var)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))
    
    return losses.avg, top1.avg


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # from PIL import Image
            # from torchvision.transforms import ToPILImage
            # from IPython.display import display

            # def imshow(img):
            #     img = img / 2 + 0.5 # Unnormalize
            #     plt.imshow(torchvision.utils.make_grid(img).permute(1, 2, 0))
            #     plt.savefig("test_noise.png")

            # array = input[2]
            # # to_pil = ToPILImage()
            # # image = to_pil(array)
            # imshow(array)

            # return 0,0

            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            # compute output
            output_ = []
            kl_ = []
            for mc_run in range(args.num_mc):
                output, kl = model(input_var)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
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
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


# Class order generated by pytorch test_loader is required
# to arrange cost sensitivity matrix, m_metric,
# with y_true on columns and y_predicted on rows.
m_metric = np.array([
    [1.0, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
    [0.4, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.6],
    [0.2, 0.2, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2],
    [0.2, 0.2, 0.4, 1.0, 0.4, 0.6, 0.4, 0.4, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.4, 1.0, 0.4, 0.4, 0.6, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.6, 0.4, 1.0, 0.4, 0.4, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 0.4, 0.2, 0.2],
    [0.2, 0.2, 0.4, 0.4, 0.6, 0.4, 0.4, 1.0, 0.2, 0.2],
    [0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 0.4],
    [0.4, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 1.0]
])


def evaluate(args, model, val_loader, fold, dataset_type,save_dir):
    # --batch-size=$batch_size 
    # --num_monte_carlo=$num_monte_carlo
    args.num_monte_carlo = 100
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    output_list = []
    labels_list = []
    predictions_L1 = []
    predictions_KLD = []
    all_p_star_L1 = []
    all_p_star_KLD = []
    eval_results = []
    model.eval()
    with torch.no_grad():
        begin = time.time()
        for data, target in val_loader:
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
        end = time.time()
        # print("inference throughput: ", len_testset / (end - begin),
        #       " images/s")
        
        # output = torch.stack(output_list)        
        # output = output.permute(1, 0, 2, 3)
        # output = output.contiguous().view(args.num_monte_carlo, len_testset,
        #                                   -1)
        # output shape torch.Size([50, 10000, 10])
        output = torch.cat(output_list, dim=1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)      
        # labels_onehot <class 'numpy.ndarray'> (10000, 10)
        labels_onehot = torch.nn.functional.one_hot(labels).data.cpu().numpy()

        # >>> Original Squared Euclidean distance >>>

        # pred_mean shape torch.Size([10000, 10])
        # all_p_star_SED = pred_mean
        pred_mean = output.mean(dim=0).data.cpu().numpy()
        # sum(m(y,y_bar)p(y|x)), numpy ndarray (10000, 10)
        all_p_star_SED_sensitive = np.matmul(pred_mean, m_metric)
        Y_pred = np.argmax(all_p_star_SED_sensitive, axis=1)
        acc_SED = (Y_pred == labels.data.cpu().numpy()).mean() * 100
        eval_results.append(acc_SED)
        print('Test accuracy:', acc_SED)

        np.save(save_dir + f'tensor_output_{dataset_type}_fold_{fold}.npy', output.data.cpu().numpy())
        # np.save('./all_p_star_SED.npy', pred_mean.data.cpu().numpy())
        # np.save('./predictions_SED.npy', Y_pred.data.cpu().numpy())
        np.save(save_dir + f'cifar_test_labels_{dataset_type}_fold_{fold}.npy', labels.data.cpu().numpy())        
        np.save(save_dir + f'cifar_test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)

        # <<< Original Squared Euclidean distance <<<

        for i in range(len(labels)):
            probability_set = output[:,i,:].data.cpu().numpy() #(50,10)

            # >>> Distances >>>
            # >>> L1_distance >>>

            def L1_distance(p_star_L1, probability_set):
                return np.sum(np.abs(probability_set - p_star_L1))

            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}]

            result = minimize(L1_distance, np.ones(10)/10, args=(probability_set,), constraints=constraints)
            
            p_star_L1 = result.x

            # all_p_star_L1 shape [10000, 10]
            all_p_star_L1.append(p_star_L1)
            # predictions_L1.append(p_star_L1.argmax())

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
            
            result = minimize(KL_divergence, np.ones(10)/10, args=(probability_set,), constraints=constraints)
            p_star_KLD = result.x
            epsilon = 1e-10
            p_star_KLD = np.where(p_star_KLD < epsilon, epsilon, p_star_KLD)

            # all_p_star_KLD shape [10000, 10]
            all_p_star_KLD.append(p_star_KLD)
            # predictions_KLD.append(p_star_KLD.argmax())

            # <<< KL_divergence <<<            
        
        all_p_star_L1_sensitive = np.matmul(all_p_star_L1, m_metric)
        predictions_L1 = np.argmax(all_p_star_L1_sensitive, axis=1)
        all_p_star_KLD_sensitive = np.matmul(all_p_star_KLD, m_metric)
        predictions_KLD = np.argmax(all_p_star_KLD_sensitive, axis=1)
        acc_L1 = (predictions_L1 == labels.data.cpu().numpy()).mean() * 100
        acc_KLD = (predictions_KLD == labels.data.cpu().numpy()).mean() * 100
        eval_results.append(acc_L1)
        eval_results.append(acc_KLD)
            
        # >>> Save test outputs >>>

        # np.save('./all_p_star_L1.npy', np.array(all_p_star_L1))
        # np.save('./predictions_L1.npy', np.array(predictions_L1))
        # np.save('./all_p_star_KLD.npy', np.array(all_p_star_KLD))
        # np.save('./predictions_KLD.npy', np.array(predictions_KLD))
        
        # <<< Save test output <<<
        # print('precise_L1_acc:', acc_L1)
        # print('precise_KLD_acc:', acc_KLD)
        
        # print('binary_ECE_SED:', binary_ECE(labels_onehot, pred_mean.data.cpu().numpy(), bins=15))
        # print('binary_ECE_L1:', binary_ECE(labels_onehot, np.array(all_p_star_L1), bins=15))
        # print('binary_ECE_KLD:', binary_ECE(labels_onehot, np.array(all_p_star_KLD), bins=15))

        # Original Test accuracy: 90.09
        # Test accuracy: 89.83
        # precise_L1_acc: 89.77000000000001
        # precise_KLD_acc: 83.39
    return eval_results


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
    main()
