#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -q gpu.q@ullman
#$ -o exp/log1 -j y

export HOME=/slwork/users/mkh95
export PATH=$HOME/anaconda3/envs/python36/bin/:$PATH

export LC_ALL="en_US.UTF-8"


for lr_method in rmsprop adam sgd; do
    for dim_char in 5 10; do
        for hidden_char in 20 50 100 300; do
            for layers in 1 2 3; do
dir=exp/$lr_method.char-$dim_char-$hidden_char.plstm-$layers
[ -d $dir ] && rm -rf $dir
CUDA_VISIBLE_DEVICES=${1:-0} python train.py \
    --lr_method $lr_method \
    --lr 0.0115 \
    --clip -1 \
    --dim_word 300 \
    --hidden_word 300 \
    --dim_cap 2 \
    --dim_char $dim_char \
    --hidden_char $hidden_char \
    --layers $layers \
    --peephole \
    --use_cap \
    --use_char \
    --use_pretrained \
    --char_mode LSTM \
    --word_mode LSTM \
    --dir $dir 
echo $dir finished.
            done
        done
    done
done
