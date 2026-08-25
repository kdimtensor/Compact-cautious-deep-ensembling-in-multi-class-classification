# !/bin/csh

model=resnet20
mode='train'
batch_size=256 #128
lr=0.01
epoch=50
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_bayesian_leaf.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --epochs=$epoch --train-type=$train_type
