#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -q gpu.q@ullman
#$ -o exp/log -j y

export HOME=/slwork/users/mkh95
export PATH=$HOME/anaconda3/envs/python36/bin/:$PATH

export LC_ALL="en_US.UTF-8"

models="exp/adam.cap-2.char-100-100.word-300-300.lstm-1.p-0 exp/rmsprop.char-100-100.plstm-2 exp/rmsprop.cap-5.char-100-100.word-300-300.sru-1 exp/rmsprop.cap-5.char-100-300.word-300-300.lstm-2.p-0 exp/rmsprop.cap-2.char-50-100.word-300-100.lstm-3.p-0 exp/rmsprop.cap-5.char-50-100.word-300-100.sru-2 exp/rmsprop.char-20-100.plstm-2 exp/rmsprop.cap-2.char-100-300.word-300-300.sru-2 exp/adam.cap-2.char-50-300.word-300-300.lstm-1.p-100 exp/rmsprop.cap-2.char-100-100.word-300-100.sru-3 exp/rmsprop.cap-2.char-50-300.word-300-300.sru-2 exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char100.hidden_char300.layers1.LSTM exp/rmsprop.cap-2.char-50-300.word-300-100.lstm-3.p-0 exp/adam.cap-5.char-50-100.word-300-300.lstm-2.p-0 exp/rmsprop.cap-2.char-100-300.word-300-300.lstm-2.p-0"

# rmsprop adam sgd adadelta 
for lr_method in sgd; do
dir=exp/$lr_method
[ -d $dir ] && rm -rf $dir
CUDA_VISIBLE_DEVICES=${1:-0} python train.py \
    --lr_method $lr_method \
    --lr 0.0115 \
    --clip -1 \
    --dim_word 300 \
    --hidden_word 300 \
    --dim_cap 2 \
    --dim_char 20 \
    --hidden_char 300 \
    --layers 2 \
    --peephole \
    --use_cap \
    --use_char \
    --use_pretrained \
    --char_mode LSTM \
    --word_mode LSTM \
    --dir $dir \
    --epochs 1 \
    --models $models 

# run parallel on same GPU
(CUDA_VISIBLE_DEVICES=${1:-0} python evaluate.py \
    --lr_method $lr_method \
    --use_cap \
    --use_char \
    --use_pretrained \
    --dir $dir \
    --models $models > "$dir/test.txt")
echo $dir finished.
done
