#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -q gpu.q@ullman
#$ -o exp/log -j y

export HOME=/slwork/users/mkh95
export PATH=$HOME/anaconda3/envs/python36/bin/:$PATH

export LC_ALL="en_US.UTF-8"

models="exp/rmsprop.char-20-300.plstm-2 exp/rmsprop.char-20-300.plstm-1 exp/rmsprop.char-100-100.plstm-2 exp/adam.char-20-300.plstm-1 exp/adam.char-20-100.plstm-2"
models="../tf_tag/exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char100.hidden_char300.layers1.LSTM ../tf_tag/exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char50.hidden_char300.layers2.LSTM ../tf_tag/adam-plstm/adam.cap-5.char-50-100.word-300-300.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-300.lstm-2.p-0 ../tf_tag/adam-plstm/adam.cap-5.char-50-100.word-300-300.lstm-1.p-0 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-100.word-300-100.sru-3 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-100.word-300-300.sru-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-300.word-300-300.sru-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-100-100.word-300-300.sru-1 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-50-100.word-300-100.sru-2 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-300.lstm-1.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-100.word-300-100.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-300.word-300-300.lstm-2.p-0 ../tf_tag/best/rmsprop.char-100-100.plstm-2 final/adam.char-100-100.plstm-2 final/adam.char-20-300.plstm-1 final/rmsprop.char-20-100.plstm-2 final/rmsprop.char-20-300.plstm-2"
models="../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-50-100.word-300-100.sru-3 ../tf_tag/adam-plstm/adam.cap-2.char-100-100.word-300-300.lstm-1.p-100 ../tf_tag/exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char50.hidden_char300.layers2.LSTM ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-100.word-300-300.sru-2 ../tf_tag/adam-plstm/adam.cap-5.char-50-100.word-300-300.lstm-1.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-100.word-300-100.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-300.word-300-100.lstm-1.p-0 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-50-100.word-300-300.sru-2 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-100.lstm-1.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-300.lstm-1.p-0 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-50-100.word-300-100.sru-3 ../tf_tag/adam-plstm/adam.cap-2.char-100-100.word-300-300.lstm-1.p-0 ../tf_tag/best/rmsprop.char-100-100.plstm-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-100-100.word-300-300.sru-1 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-300.word-300-300.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-50-100.word-300-100.lstm-3.p-0 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-50-100.word-300-100.sru-2 ../tf_tag/best/rmsprop.char-20-100.plstm-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-300.word-300-300.sru-2 ../tf_tag/adam-plstm/adam.cap-2.char-50-300.word-300-300.lstm-1.p-100 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-100.word-300-100.sru-3 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-50-300.word-300-300.sru-2 ../tf_tag/exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char100.hidden_char300.layers1.LSTM ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-50-300.word-300-100.lstm-3.p-0 ../tf_tag/adam-plstm/adam.cap-5.char-50-100.word-300-300.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-300.lstm-2.p-0 final/adam.char-100-100.plstm-2 final/rmsprop.char-20-300.plstm-2 final/rmsprop.char-20-100.plstm-2 final/adam.char-20-300.plstm-1"
models="../tf_tag/adam-plstm/adam.cap-2.char-100-100.word-300-300.lstm-1.p-0 ../tf_tag/best/rmsprop.char-100-100.plstm-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-100-100.word-300-300.sru-1 ../tf_tag/rmsprop-plstm/rmsprop.cap-5.char-100-300.word-300-300.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-50-100.word-300-100.lstm-3.p-0 ../tf_tag/rmsprop-lstm/rmsprop.cap-5.char-50-100.word-300-100.sru-2 ../tf_tag/best/rmsprop.char-20-100.plstm-2 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-300.word-300-300.sru-2 ../tf_tag/adam-plstm/adam.cap-2.char-50-300.word-300-300.lstm-1.p-100 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-100-100.word-300-100.sru-3 ../tf_tag/rmsprop-lstm/rmsprop.cap-2.char-50-300.word-300-300.sru-2 ../tf_tag/exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char100.hidden_char300.layers1.LSTM ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-50-300.word-300-100.lstm-3.p-0 ../tf_tag/adam-plstm/adam.cap-5.char-50-100.word-300-300.lstm-2.p-0 ../tf_tag/rmsprop-plstm/rmsprop.cap-2.char-100-300.word-300-300.lstm-2.p-0 final/rmsprop.char-20-100.plstm-2 final/adam.char-20-300.plstm-1"

# rmsprop adam sgd
for lr_method in nsgd adadelta ; do
dir=exp/$lr_method.17
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
