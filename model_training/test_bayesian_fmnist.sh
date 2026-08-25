#!/bin/bash

mode='test'
num_monte_carlo=100
train_type='clean'

CUDA_VISIBLE_DEVICES=1 python main_bayesian_fmnist.py --mode=$mode --num_monte_carlo=$num_monte_carlo --train-type=$train_type
