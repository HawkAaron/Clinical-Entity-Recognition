#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -q gpu.q@banana
#$ -o final/log -j y

export HOME=/slwork/users/mkh95
export PATH=$HOME/anaconda3/envs/python36/bin/:$PATH
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/download/cudnn6/lib64:$LD_LIBRARY_PATH

export LC_ALL="en_US.UTF-8"

#rmsprop adam adadelta
for lr_method in rmsprop adam; do
    for dim_char in 20 100; do
        for hidden_char in 300; do
            [ $hidden_char -lt $dim_char ] && continue
            for hidden_word in 300 1024; do
            for layers in 1 2; do
dir=final/$lr_method.char-$dim_char.word-$hidden_word.plstm-$layers
[ -d $dir ] && rm -rf $dir
CUDA_VISIBLE_DEVICES=${1:-0} python train.py \
    --lr_method $lr_method \
    --lr 0.0115 \
    --clip -1 \
    --dim_word 300 \
    --hidden_word $hidden_word \
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
done
