#!/bin/bash

model=resnet18_dropout
num_monte_carlo=100
modes="test"
p=0.02
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_cnndropout_cifar.py --model_name=$model --num_monte_carlo=$num_monte_carlo --mode=$modes --p=$p --train-type=$train_type
