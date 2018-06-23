# Clinical Entity Recognition with Tensorflow
This repo implements a CER model using Tensorflow (chars embeddings + word embeddings + BLSTM + CRF).

State-of-the-art performance after ensemble training (F1 score between 84 and 85 on test set).

For more details, please check the [report](https://github.com/HawkAaron/Clinical-Entity-Recognition/blob/master/doc/report.pdf).

## Getting started
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
