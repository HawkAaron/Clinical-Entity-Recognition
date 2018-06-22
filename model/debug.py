import os
import time
import numpy as np
import tensorflow as tf

from .data_utils import pad_sequences, minibatches, get_chunks
from .ner_model import NERModel

class Debug(NERModel):
    """Ensemble of several models"""
    def __init__(self, config):
        super(Debug, self).__init__(config)
        

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        start_time = time.time()
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            # no dropout
            fd[self.dropout] = 1.0
            # not update
            out = self.sess.run(
                    [self.loss, self.logits], feed_dict=fd)
            print(out)
            exit()

            if i % 50 == 0:
                self.logger.info('[Epoch {}] batch {} loss {:.3f}'.format(epoch+1, i, train_loss))

        metrics = self.run_evaluate(dev)
        msg = ' - '.join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info('Epoch {} (time: {:.3f} s) (lr: {:.3e}), {}'.format(
                            epoch+1, time.time() - start_time, self.config.lr, msg))
        
        return metrics['f1']

