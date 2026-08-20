'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset, Subset

import argparse
import os
import sys
import copy

from contrib import adf
from models.resnet import ResNet18
from models.resnet_dropout import ResNet18Dropout
from utils import progress_bar

import numpy as np
from sklearn.model_selection import KFold
from cifar10_c import CIFAR10C, CIFAR10CLEAN

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import saved_loss

from tqdm import tqdm
#########
root_dir = ""
#########

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

##################

def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""

    for m in net.modules():
#        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net   

#################

# Model flags
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--p', default=0.2, type=float, help='dropout rate')
parser.add_argument('--noise_variance', default=1e-3, type=float, 
                    help='noise variance')
parser.add_argument('--min_variance', default=1e-3, type=float, 
                    help='min variance')
# Training flags
parser.add_argument('--model_name', default='resnet18', type=str,  
                    help='model to train')
parser.add_argument('--resume', '-r', action='store_true', default=False, 
                    help='resume from checkpoint')
parser.add_argument('--show_bar', '-b', action='store_true', default=True, 
                    help='show bar or not')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--num_epochs', default=350, type=int, 
#                     help='number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, 
                    help='size of training batch')
parser.add_argument('--workers', default=8, type=int, 
                    help='number of workers')

parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default=root_dir + '/checkpoint/bayesian',
                    type=str)

parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 200)')

parser.add_argument('--num_monte_carlo', default=100, type=int, 
                    help='number of monte carlo')

parser.add_argument('--use_mcdo', '-m', action='store_true', default=False)

parser.add_argument('--mode', type=str, required=True, help='train | test')

args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# # Data
# print('==> Preparing data...')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', 
#                                         train=True, 
#                                         download=True, 
#                                         transform=transform_train)

# trainloader = torch.utils.data.DataLoader(trainset, 
#                                           batch_size=args.batch_size, 
#                                           shuffle=True, 
#                                           num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', 
#                                        train=False, 
#                                        download=True, 
#                                        transform=transform_test)

# testloader = torch.utils.data.DataLoader(testset, 
#                                          batch_size=100, 
#                                          shuffle=False, 
#                                          num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')

# train_data_c = CIFAR10C(
#     root='../data/CIFAR-10-C/i/gaussian_noise_6/',
#     train=True,
#     transform=transform_train,
#     download=False)

# test_data_c = CIFAR10C(
#     root='../data/CIFAR-10-C/i/gaussian_noise_6/',
#     train=False,
#     transform=transform_test,
#     download=False)

# train_data_clean = CIFAR10CLEAN(
#     root='../data/CIFAR-10-C/i/gaussian_noise_6/',
#     train=True,
#     transform=transform_train,
#     download=False)

# test_data_clean = CIFAR10CLEAN(
#     root='../data/CIFAR-10-C/i/gaussian_noise_6/',
#     train=False,
#     transform=transform_test,
#     download=False)

# dataset_all_c = ConcatDataset([train_data_c, test_data_c])
# dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])
# k_folds = 3
# kfold = KFold(n_splits=k_folds, shuffle=False)  

# Model
# print('==> Building model...')

def model_loader():
    model = {'resnet18': ResNet18,
             'resnet18_dropout': ResNet18Dropout
            }
    
    params = {'resnet18': [],
             'resnet18_dropout': [args.p],
             'resnet18_heteroscedastic': [args.p],
             'resnet18_adf': [args.noise_variance, args.min_variance],
             'resnet18_dropout_adf': [args.p, args.noise_variance, args.min_variance],
             }
    
    return model[args.model_name.lower()](*params[args.model_name.lower()])

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

#     model_to_load = args.model_name.lower()
#     ckpt_path = './checkpoint/ckpt_{}.pth'.format(model_to_load)
#     checkpoint = torch.load(ckpt_path)
#     print('Loaded checkpoint at location {}'.format(ckpt_path))

#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]
    
    return y_true

def keep_variance(x, min_variance):
        return x + min_variance
################

def get_labels(val_loader, o_path, fold, dataset_type):
    label_batch = []    
    for _, target in val_loader:
        label_batch.append(target)

    labels = torch.cat(label_batch, dim=0)
    labels_onehot = torch.nn.functional.one_hot(labels)

    np.save(f'{o_path}/test_labels_{dataset_type}_fold_{fold}.npy', labels)        
    np.save(f'{o_path}/test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)

def evaluate(model, val_loader, save_dir, fold, dataset_type):
    # Get and save labels from val_loader
    top1 = AverageMeter()
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
    model = set_training_mode_for_dropout(model, True)
    with torch.no_grad():

        for data, target in tqdm(val_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_mc = []
            for mc_run in range(args.num_monte_carlo):
                output = model(data)
                output_mc.append(output)
            output_ = torch.stack(output_mc)

            output_list.append(output_)
            labels_list.append(target)
            ###
            prec1 = accuracy(torch.mean(output_, dim=0).data, target)[0]
            top1.update(prec1.item(), data.size(0))
            ####
        # print("inference throughput: ", len_testset / (end - begin),
        #       " images/s")
        
        # output = torch.stack(output_list)        
        # output = output.permute(1, 0, 2, 3)
        # output = output.contiguous().view(args.num_monte_carlo, len_testset,
        #                                   -1)
        # output shape torch.Size([50, 10000, 10])
        output = torch.cat(output_list, dim=1)
        output = torch.nn.functional.softmax(output, dim=2)
        ###
        
        ####
        labels = torch.cat(labels_list)      
        # labels_onehot <class 'numpy.ndarray'> (10000, 10)
        labels_onehot = torch.nn.functional.one_hot(labels).data.cpu().numpy()

        # >>> Original Squared Euclidean distance >>>
        ##
        print(top1.avg)
        ##
        # pred_mean shape torch.Size([10000, 10])
        # all_p_star_SED = pred_mean
        # pred_mean = output.mean(dim=0).data.cpu().numpy()
        # sum(m(y,y_bar)p(y|x)), numpy ndarray (10000, 10)
        # all_p_star_SED_sensitive = np.matmul(pred_mean, m_metric)
        # Y_pred = np.argmax(all_p_star_SED_sensitive, axis=1)
        # acc_SED = (Y_pred == labels.data.cpu().numpy()).mean() * 100
        # eval_results.append(acc_SED)
        # print('Test accuracy:', acc_SED)

        np.save(save_dir + f'tensor_output_{dataset_type}_fold_{fold}.npy', output.data.cpu().numpy())
        # np.save('./all_p_star_SED.npy', pred_mean.data.cpu().numpy())
        # np.save('./predictions_SED.npy', Y_pred.data.cpu().numpy())
        np.save(save_dir + f'test_labels_{dataset_type}_fold_{fold}.npy', labels.data.cpu().numpy())        
        np.save(save_dir + f'test_labels_onehot_{dataset_type}_fold_{fold}.npy', labels_onehot)
    
    model = set_training_mode_for_dropout(model, False)

############
# Heteroscedastic loss
class SoftmaxHeteroscedasticLoss(torch.nn.Module):    
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        keep_variance_fn = lambda x: keep_variance(x, min_variance=args.min_variance)
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)
        
    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = one_hot_pred_from_label(mean, targets)
        
        precision = 1/(var + eps)
        return torch.mean(0.5*precision * (targets-mean)**2 + 0.5*torch.log(var+eps))

if args.model_name.lower().endswith('adf'):
    criterion = SoftmaxHeteroscedasticLoss()
else:
    criterion = nn.CrossEntropyLoss()

# Training
def train(epoch, model_fold, train_loader,criterion,optimizer):

    # print('\nEpoch: {} ==> lr: {}'.format(epoch, scheduler.get_last_lr()))
    model_fold.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model_fold(inputs)
        if args.model_name.lower().endswith('adf'):
            outputs_mean, outputs_var = outputs
            loss = criterion(outputs, targets)
            outputs_mean, _ = outputs
        else:
            outputs_mean = outputs
            loss = criterion(outputs_mean, targets)

        # print(loss)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()

        # _, predicted = outputs_mean.max(1)

        prec1 = accuracy(outputs_mean.data, targets)[0]

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # if args.show_bar:
        #     progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train Epoch: {} \tLoss: {:.6f} \t Acc: {:.3f}'.format(
        epoch, losses.avg,top1.avg))

    return losses.avg, top1.avg


def validate(model_fold, val_loader, criterion):
    global best_acc
    model_fold.eval()

    losses = AverageMeter()
    top1 = AverageMeter()


    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_fold(inputs) 
            if args.model_name.lower().endswith('adf'):
                outputs_mean, outputs_var = outputs
                loss = criterion(outputs, targets)
                outputs_mean, _ = outputs
            else:
                outputs_mean = outputs
                loss = criterion(outputs_mean, targets)

            test_loss += loss.item()
            # _, predicted = outputs_mean.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            output = outputs_mean.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
            losses.avg, top1.avg))

            # if args.show_bar:
            #     progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return losses.avg, top1.avg
    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('\nSaving..')
    #     state = {
    #         'model_fold': model_fold.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     # if not os.path.isdir('checkpoint'):
    #     #     os.mkdir('checkpoint')
            
    #     torch.save(state, os.path.join(save_dir, save_filename))
    #     best_acc = acc


def main():
    # >>> K_fold cross validation config >>>
    k_folds = 3
    # <<< K_fold cross validation config <<<

    #############
    train_type = "noise"
    print(train_type)
    #############
    global args, best_prec1_noise,best_prec1_clean,device

    device = load_device()
    args = parser.parse_args()

    # Check the save_dir exists or not
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # print('==> Training parameters:')
    # print('        start_epoch = {}'.format(start_epoch+1))
    # print('        best_acc    = {}'.format(best_acc))
    # print('        lr @epoch=0 = {}'.format(args.lr))
    # print('==> Starting training...')
    # model = torch.nn.DataParallel(resnet.__dict__[args.arch]())

    model = model_loader().to(device)

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

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_data = CIFAR10C(
        root= root_dir + '/data/CIFAR-10-C/gaussian_noise_6/',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
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

    dataset_all_c = ConcatDataset([train_data, test_data])
    dataset_all_clean = ConcatDataset([train_data_clean, test_data_clean])
    kfold = KFold(n_splits=k_folds, shuffle=False)    

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    # if args.half:
    #     model.half()
    #     criterion.half()

    # if args.arch in ['resnet110']:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr * 0.1

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    ####
    print(root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####
    kfold_results_c = []
    kfold_results_clean = []

    if args.mode == 'train':
        
        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            args.save_dir = root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/Drop_out/'
            os.makedirs(args.save_dir,exist_ok=True)
            os.makedirs(args.save_dir + 'model',exist_ok=True)
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

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.workers, shuffle = True)

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
            output_loss = saved_loss.loss_out(args.save_dir +f'bayesian_{args.model_name}_cifar_fold_{fold}.csv')
            ####
            for epoch in range(1, args.epochs):
                
                lr = args.lr
                if (epoch >= 80 and epoch < 120):
                    lr = 0.1 * args.lr
                elif (epoch >= 120 and epoch < 160):
                    lr = 0.01 * args.lr
                elif (epoch >= 160 and epoch < 180):
                    lr = 0.001 * args.lr
                elif (epoch >= 180):
                    lr = 0.0005 * args.lr

                optimizer = optim.SGD(model_fold.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1, last_epoch=-1)

                # train for one epoch
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
                train_loss, train_acc = train( epoch, model_fold, train_loader, criterion, optimizer)

                val_loss_noise, prec1_noise = validate(model_fold,val_loader, criterion)
                
                val_loss_clean, prec1_clean = validate( model_fold,val_loader_clean, criterion)

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
                            args.save_dir, f'model/bayesian_{args.model_name}_cifar_noise_fold_{fold}.pth'))

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
                            args.save_dir, f'model/bayesian_{args.model_name}_cifar_clean_fold_{fold}.pth'))
                            
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
        
        args.save_dir = root_dir + f'/data/CIFAR-10-C/gaussian_{train_type}_fold/Drop_out/'
        save_dir =args.save_dir
        # os.makedirs(args.save_dir,exist_ok=True)
        print(args.p)
        # param_file = f'{save_dir}param.txt'
        # checkpoint_file = f'{save_dir}bayesian_{args.arch}_cifar_fold_0.pth'
        # if torch.cuda.is_available():
        #     checkpoint = torch.load(checkpoint_file)
        # else:
        #     checkpoint = torch.load(checkpoint_file,
        #                             map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint['state_dict'])

        # with open(param_file, "a") as f:
        #     for name, parameter in model.named_parameters():
        #         if not parameter.requires_grad:
        #             continue
        #         f.write(name+'\n')
        #         f.write(str(parameter.numel())+'\n')
        #         f.write('-'*11+'\n')

        # sys.exit()


        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')
                        
            # test_ids = np.load(f'{save_dir}test_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/CIFAR-10-C/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)           
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=args.batch_size, num_workers=args.workers)
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=args.batch_size, num_workers=args.workers)

            checkpoint_file = f'{save_dir}model/bayesian_{args.model_name}_cifar_{train_type}_fold_{fold}.pth'
            
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            evaluate( model, val_loader_c,save_dir ,fold, 'c')
            evaluate( model, val_loader_clean, save_dir ,fold, 'clean')
        
        # print(np.round(kfold_results_c, 2))
        # print('-'*8)
        # print(np.round(kfold_results_clean, 2))



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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()