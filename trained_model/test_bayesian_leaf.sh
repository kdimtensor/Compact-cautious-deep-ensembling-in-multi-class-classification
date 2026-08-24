#!/bin/bash

model=resnet20
mode='test'
batch_size=256
num_monte_carlo=100
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_bayesian_leaf.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo --train-type=$train_type
