import os
import time
import tensorflow as tf

from .data_utils import pad_sequences, minibatches
from .ner_model import NERModel

class Ensemble(NERModel):
    """Ensemble of several models"""
    def __init__(self, config):
        super(Ensemble, self).__init__(config)
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
            #feed[scope + '/labels:0'] = labels
            feed[scope + '/dropout:0'] = 1.0
        feed[self.sequence_lengths] = sequence_lengths

        return feed, sequence_lengths
            
    def add_logits_op(self):
        with tf.variable_scope('fusion'):
            # average initial
            w = tf.get_variable('w', dtype=tf.float32, shape=[len(self.models)],
                    initializer=tf.zeros_initializer())
            w = tf.nn.softmax(w)
            out = tf.tensordot(tf.stack(self.models), w, [0, 0])
            self.logits = tf.layers.dense(tf.nn.relu(out), self.config.ntags,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    def add_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        if self.config.use_crf:
            log_ll, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths, self.trans)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_ll)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar('loss', self.loss)

    def add_train_op(self, lr_method, lr, loss, clip=0, g_vars=None):
        _lr_m = lr_method.lower()

        with tf.variable_scope('train_setp'):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            
            if clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(loss, var_list=g_vars))
                        #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fusion')))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss, var_list=g_vars) 
                        #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fusion'))

    def build(self):
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # fusion scope
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'fusion' in var.name]
 
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip, g_vars)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(g_vars)
                #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fusion'))

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

