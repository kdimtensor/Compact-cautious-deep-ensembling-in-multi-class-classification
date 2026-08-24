# !/bin/csh

lr=0.01
batch_size=256
epochs=5
mode='train'
train_type='clean'

CUDA_VISIBLE_DEVICES=1 python main_bayesian_fmnist.py --lr=$lr --batch-size=$batch_size --epochs=$epochs --mode=$mode --train-type=$train_type
