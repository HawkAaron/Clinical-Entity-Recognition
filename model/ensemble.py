import os
import time
import random
import numpy as np
import tensorflow as tf

from .data_utils import pad_sequences, minibatches, get_chunks
from .base_model import BaseModel

def all_saveable_objects(scope=None):
    return (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) +
            tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS, scope))

class Ensemble():
    """Ensemble of several models"""
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.idx_to_tag = {idx: tag for tag, idx in
                        self.config.vocab_tags.items()}

        random.seed(1024)
        tf.set_random_seed(1024)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cfg)
        # teachers
        teas = config.ensembles
        models = [None] * len(teas)
        trans = [None] * len(teas)
        for i, t in enumerate(teas):
            scope = 'model' + str(i)
            saver = tf.train.import_meta_graph(
                    os.path.join(t, 'params.meta'),
                    clear_devices=True,
                    import_scope=scope)
            saver.restore(self.sess, os.path.join(t, 'params'))
            # get output before dense layer
            #models[i] = tf.get_default_graph().get_tensor_by_name('bi-lstm/concat:0')
            models[i] = tf.get_default_graph().get_operation_by_name(scope + '/proj/dense/BiasAdd').outputs[0]
            trans[i] = tf.get_default_graph().get_tensor_by_name(scope + '/transitions:0')
            # not train models
        self.models = models
        self.trans = tf.reduce_mean(tf.stack(trans), 0)
        
    def add_placeholders(self):
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None],
                        name="labels")
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None],
                        name="sequence_lengths")

    def get_feed_dict(self, words, labels=None, lr=None, dropout=1.0):
        ret = list(zip(*words))
        # caps
        cap_ids = ret[0]
        cap_ids, _ = pad_sequences(cap_ids)
        # chars
        char_ids = ret[1]
        char_ids, word_lengths = pad_sequences(char_ids, nlevels=2)
        # word
        word_ids = ret[2]
        word_ids, sequence_lengths = pad_sequences(word_ids)

        # feed dict
        feed = {}
        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels
        if lr is not None: feed[self.lr] = lr
        for i in range(len(self.models)):
            scope = 'model' + str(i)
            feed[scope + '/word_ids:0'] = word_ids
            feed[scope + '/sequence_lengths:0'] = sequence_lengths
            feed[scope + '/cap_ids:0'] = cap_ids
            feed[scope + '/char_ids:0'] = char_ids
            feed[scope + '/word_lengths:0'] = word_lengths
            feed[scope + '/dropout:0'] = 1.0
        feed[self.sequence_lengths] = sequence_lengths

        return feed, sequence_lengths
            
    def add_logits_op(self):
        with tf.variable_scope('fusion'):
            # average initial
            w = tf.get_variable('w', dtype=tf.float32, shape=[len(self.models)],
                    initializer=tf.zeros_initializer())
            w = tf.nn.softmax(w)
            # multiply each model outputs with a probability scalar
            out = tf.tensordot(tf.stack(self.models), w, [0, 0])
            self.logits = out
            #self.logits = tf.layers.dense(tf.nn.relu(out), self.config.ntags,
            #            kernel_initializer=tf.contrib.layers.xavier_initializer())

    def add_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        if self.config.use_crf:
            with tf.variable_scope('fusion'):
                trans = tf.Variable(self.trans, name='transitions')
                log_ll, trans_params = tf.contrib.crf.crf_log_likelihood(
                        self.logits, self.labels, self.sequence_lengths, trans)
                self.trans_params = trans_params
                self.loss = tf.reduce_mean(-log_ll)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # TODO only summary this training
        tf.summary.scalar('loss', self.loss)

    def add_train_op(self, lr_method, lr, loss, clip=0, tvars=None):
        _lr_m = lr_method.lower()

        with tf.variable_scope('train_step'):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'msgd':
                optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            elif _lr_m == 'nsgd':
                optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            
            if clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(loss, var_list=tvars))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss, var_list=tvars) 


    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # fusion scope
        tvars = tf.trainable_variables(scope='fusion') 
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip, tvars)

        # only init fusion and train_step scopes
        self.sess.run(tf.variables_initializer(tvars + all_saveable_objects('train_step')))

        # {v.op.name: v for v in tvars} + {'transitions': self.trans_params}
        self.saver = tf.train.Saver(tvars)
    
    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        start_time = time.time()
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=fd)

            if i % 50 == 0:
                self.logger.info('[Epoch {}] batch {} loss {:.3f}'.format(epoch+1, i, train_loss))

        metrics = self.run_evaluate(dev)
        msg = ' - '.join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info('Epoch {} (time: {:.3f} s) (lr: {:.3e}), {}'.format(
                            epoch+1, time.time() - start_time, self.config.lr, msg))
        
        return metrics['f1']

    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {'p': 100*p, 'r': 100*r, 'acc': 100*acc, 'f1': 100*f1}

    def predict(self, words):
        if type(words[0]) == tuple:
            words = list(zip(*words))
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def restore_session(self, dir_model):
        self.logger.info('Reloading the latest trained model...')
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        self.sess.close()

    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0
        
        for epoch in range(self.config.nepochs):
            self.logger.info('Epoch {} out of {}'.format(epoch + 1,
                        self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            
            # only one epoch
            #return ;
            if score > best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info('- new best score {}'.format(score))
            else:
                self.config.lr *= self.config.lr_decay
                self.restore_session(self.config.dir_model)
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info('- early stopping {} epochs without '\
                            'improvement'.format(nepoch_no_imprv))
                    break

    def evaluate(self, test):
        self.logger.info('Testing model over test set')
        metrics = self.run_evaluate(test)
        msg = ' - '.join(['{} {:04.2f}'.format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

