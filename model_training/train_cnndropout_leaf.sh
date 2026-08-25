# # !/bin/csh

model=resnet18_dropout
dropout_rate=0
epochs=50
modes="train"
batch=128
train_type='clean'

CUDA_VISIBLE_DEVICES=0 python main_cnndropout_leaf.py --model_name=$model --p=$dropout_rate --epochs=$epochs --mode=$modes --batch_size=$batch --train-type=$train_type
