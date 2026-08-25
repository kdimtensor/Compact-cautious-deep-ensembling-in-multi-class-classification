#!/bin/bash

mode='test'
save_dir='./checkpoint/bayesian'
num_monte_carlo=100
train_type='clean'

CUDA_VISIBLE_DEVICES=1 python main_bayesian_fmnist.py --mode=$mode --save_dir=$save_dir --num_monte_carlo=$num_monte_carlo --train-type=$train_type
