#!/bin/bash

model=resnet18_dropout
num_monte_carlo=100
modes="test"
p=0.02
batch=128
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_cnndropout_leaf.py --model_name=$model --num_monte_carlo=$num_monte_carlo --mode=$modes --p=$p --batch_size=$batch --train-type=$train_type
