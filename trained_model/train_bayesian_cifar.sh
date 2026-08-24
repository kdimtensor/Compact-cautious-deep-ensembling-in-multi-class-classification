#!/bin/bash

model=resnet20
mode='train'
batch_size=256 #128
lr=0.001
epoch=5
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_bayesian_cifar.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --epochs=$epoch --train-type=$train_type

