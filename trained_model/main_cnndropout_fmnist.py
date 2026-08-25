'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset, Subset
from torch.optim.lr_scheduler import StepLR

import argparse
import os
import sys
import copy

from models.resnet_dropout import BasicBlock, SimpleDropout

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from data_preprocess.fmnist_c import FMNISTC, FMNISTCLEAN

from tqdm import tqdm
import saved_loss
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

################

def evaluate(model, val_loader, save_dir, fold, dataset_type):
    # Get and save labels from val_loader
    ############################
    print(args.num_monte_carlo)
    pred_probs_mc = []
    eval_results = []
    model.eval()
    model = set_training_mode_for_dropout(model, True)
    with torch.no_grad():
        pred_probs_mc = []
        for data, target in tqdm(val_loader):
            # data, target = data.to(device), target.to(device)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            for mc_run in range(args.num_monte_carlo):                
                # output, _ = model.forward(data)
                output = model(data)
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
    
    model = set_training_mode_for_dropout(model, False)

# Training
def train(epoch, model_fold, train_loader,optimizer):

    # print('\nEpoch: {} ==> lr: {}'.format(epoch, scheduler.get_last_lr()))
    model_fold.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model_fold(inputs)
        outputs_mean = outputs
        loss = F.nll_loss(outputs_mean, targets)

        # print(loss)
        loss.backward()
        optimizer.step()

        prec1 = accuracy(outputs_mean.data, targets)[0]

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

    print('Train Epoch: {} \tLoss: {:.6f} \t Acc: {:.3f}'.format(
        epoch, losses.avg,top1.avg))

    return losses.avg, top1.avg


def validate(model_fold, val_loader):
    model_fold.eval()

    losses = AverageMeter()
    top1 = AverageMeter()


    test_loss = 0

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_fold(inputs) 

            outputs_mean = outputs
            loss = F.nll_loss(outputs_mean, targets)

            test_loss += loss.item()

            output = outputs_mean.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}'.format(
            losses.avg, top1.avg))

    return losses.avg, top1.avg

def main():

    global args, best_prec1_noise,best_prec1_clean,device
    args = parser.parse_args()
    # >>> K_fold cross validation config >>>
    k_folds = args.k_fold
    # <<< K_fold cross validation config <<<

    #############
    root_dir = ".."
    train_type = args.train_type
    torch.manual_seed(args.seed)
    print(train_type)
    data_name = "FMNIST-10-C"
    print(data_name)
    #############
    args.save_dir = root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/Drop_out/'
    device = load_device()

    model = SimpleDropout(BasicBlock, [2,2,2,2],num_classes = 10, p=args.p)
    model = model.to(device)

    ####
    print(root_dir + f'/data/{data_name}/gaussian_{train_type}_fold/')
    print(args.epochs)
    ####

    if args.mode == 'train':
        
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

            os.makedirs(args.save_dir,exist_ok=True)
            os.makedirs(args.save_dir + 'model',exist_ok=True)
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
            # output_loss = saved_loss.loss_out(args.save_dir +f'bayesian_{args.model_name}_cifar_fold_{fold}.csv')
            output_loss = saved_loss.loss_out(args.save_dir + f'bayesian_{args.model_name}_cifar_fold_{fold}.csv')
            ####
            for epoch in range(1, args.epochs):
                
                if (epoch < args.epochs / 3):
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr)
                elif (epoch < args.epochs*2/3):
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr/10)
                else:
                    optimizer = optim.Adadelta(model_fold.parameters(), lr=args.lr /100)

                scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

                # train for one epoch
                print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

                train_loss, train_acc = train( epoch, model_fold, train_loader, optimizer)

                val_loss_noise, prec1_noise = validate(model_fold,val_loader)
                
                val_loss_clean, prec1_clean = validate( model_fold,val_loader_clean)

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
                            args.save_dir, f'model/bayesian_{args.model_name}_noise_fold_{fold}.pth'))

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
                            args.save_dir, f'model/bayesian_{args.model_name}_clean_fold_{fold}.pth'))
                            
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
        print(args.p)

        for fold in range(k_folds):
            print(f'FOLD {fold}')
            print('--------------------------------')
                        
            # test_ids = np.load(f'{save_dir}test_ids_fold_{fold}.npy')
            test_ids = np.load(root_dir + f'/data/{data_name}/k_fold_id/test_ids_fold_{fold}.npy')
            
            val_subset_c = Subset(dataset_all_c, test_ids)           
            val_loader_c = torch.utils.data.DataLoader(val_subset_c, batch_size=len(test_ids))
            
            val_subset_clean = Subset(dataset_all_clean, test_ids)           
            val_loader_clean = torch.utils.data.DataLoader(val_subset_clean, batch_size=len(test_ids))

            checkpoint_file = f'{args.save_dir}model/bayesian_{args.model_name}_{train_type}_fold_{fold}.pth'
            
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_file)
            else:
                checkpoint = torch.load(checkpoint_file,
                                        map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            evaluate( model, val_loader_c,args.save_dir ,fold, 'c')
            evaluate( model, val_loader_clean, args.save_dir ,fold, 'clean')

        return


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

    parser = argparse.ArgumentParser(description='PyTorch fmnist Training')
    parser.add_argument('--p', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--noise_variance', default=1e-3, type=float, 
                        help='noise variance')
    parser.add_argument('--min_variance', default=1e-3, type=float, 
                        help='min variance')
    # Training flags
    parser.add_argument('--model_name', default='resnet18', type=str,  
                        help='model to train')

    parser.add_argument('--gamma',
                            type=float,
                            default=0.7,
                            metavar='M',
                            help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

    parser.add_argument('--batch_size', default=128, type=int, 
                        help='size of training batch')
    parser.add_argument('--workers', default=8, type=int, 
                        help='number of workers')

    parser.add_argument('--save-dir',
                        dest='save_dir',
                        help='The directory used to save the trained models',
                        default='/checkpoint/bayesian',
                        type=str)

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 200)')

    parser.add_argument('--num_monte_carlo', default=100, type=int, 
                        help='number of monte carlo')

    parser.add_argument('--mode', type=str, required=True, help='train | test')
    parser.add_argument('--train-type',
                                dest='train_type',
                                type=str,
                                default='clean',
                                choices=['noise', 'clean'],
                                help='type of training data: noise | clean (default: clean)')
    parser.add_argument('--seed',
                            type=int,
                            default=42,
                            metavar='S',
                            help='random seed (default: 42)')
    
    parser.add_argument('--k-fold',
                        dest='k_fold',
                        type=int,
                        default=3,
                        metavar='N',
                        help='number of folds for cross validation (default: 3)')

    ##########################
    main()