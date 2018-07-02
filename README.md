# Clinical Entity Recognition with Tensorflow
This repo implements a CER model using Tensorflow (chars embeddings + word embeddings + BLSTM + CRF).

State-of-the-art performance after ensemble training (F1 score between 84 and 85 on test set).

For more details, please check the [report](https://github.com/HawkAaron/Clinical-Entity-Recognition/blob/master/doc/report.pdf).

## Using pretrained models
1. Pretrained graph can be easily loaded using TensorFlow. The pretrained models are in `exp` dir.

2. Using ensemble trained model with
```bash
# pretrained ensemble of models
models="exp/adam.cap-2.char-100-100.word-300-300.lstm-1.p-0 exp/rmsprop.char-100-100.plstm-2 exp/rmsprop.cap-5.char-100-100.word-300-300.sru-1 exp/rmsprop.cap-5.char-100-300.word-300-300.lstm-2.p-0 exp/rmsprop.cap-2.char-50-100.word-300-100.lstm-3.p-0 exp/rmsprop.cap-5.char-50-100.word-300-100.sru-2 exp/rmsprop.char-20-100.plstm-2 exp/rmsprop.cap-2.char-100-300.word-300-300.sru-2 exp/adam.cap-2.char-50-300.word-300-300.lstm-1.p-100 exp/rmsprop.cap-2.char-100-100.word-300-100.sru-3 exp/rmsprop.cap-2.char-50-300.word-300-300.sru-2 exp/adam.0.0115.-1.dim_word200.hidden_word300.dim_char100.hidden_char300.layers1.LSTM exp/rmsprop.cap-2.char-50-300.word-300-100.lstm-3.p-0 exp/adam.cap-5.char-50-100.word-300-300.lstm-2.p-0 exp/rmsprop.cap-2.char-100-300.word-300-300.lstm-2.p-0"

# ensemble trained model dir
dir=exp/sgd

# evaluate
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
            --lr_method sgd \
            --use_cap \
            --use_char \
            --use_pretrained \
            --dir $dir \
            --models $models > "$dir/test.txt"

# combine prediction
python combine.py --src data/test.txt --pred "$dir/test.txt" --dst $dir/result.txt

```

## Start from scratch
1. Download the GloVe vectors with
```bash
wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip"
unzip ./data/glove.6B.zip -d data/glove.6B/
rm ./data/glove.6B.zip
```

2. Build the training data, train and evaluate the model with
```bash
python build_data.py
python train.py <params>
python evaluate.py <params>
```

## Requirements
* Python 3.6
* NumPy 1.14
* TensorFlow >= 1.4

## Ensemble training
1. Grid search hyper parameters
Change `grid.sh` to do grid search.

2. Ensemble
Choose several models with highest F1 score on dev set, add their dirs to `models` in `ensemble.sh`.


# References
[Named Entity Recognition with Tensorflow](https://github.com/guillaumegenthial/sequence_tagging)

[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf)

[Unified Neural Architecture for Drug, Disease and Clinical Entity Recognition](https://arxiv.org/pdf/1708.03447.pdf)

[Bidirectional LSTM-CRF for Clinical Concept Extraction](https://arxiv.org/pdf/1610.05858.pdf)
