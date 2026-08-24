# !/bin/csh

model=resnet20
mode='train'
batch_size=256 #128
lr=0.01
epoch=5
train_type='noise'

CUDA_VISIBLE_DEVICES=0 python main_bayesian_leaf.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --epochs=$epoch --train-type=$train_type
