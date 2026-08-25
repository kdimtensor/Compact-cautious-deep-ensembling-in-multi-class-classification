#!/bin/bash

model=resnet18_dropout
dropout_rate=0
epochs=50
modes="train"
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_cnndropout_cifar.py --model_name=$model --p=$dropout_rate --epochs=$epochs --mode=$modes --train-type=$train_type
