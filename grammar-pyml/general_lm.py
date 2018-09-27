# tensorflow essentials
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import dynamic_rnn
# non-tensorflow essentials
import numpy as np
import datetime as dt
# for saving and restarting, generating from trained model
import argparse
import pickle
import os
# for fancy output
from multiprocessing.pool import ThreadPool
from time import sleep
import sys
# local imports
import reader
from configs import *

# Stores all the additional information needed by generator.py to restore
#   functionality from a pre-trained model.
class MetaModel(object):
    def __init__(self, word_to_id, id_to_word, configuration):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.config = configuration
        # bit of a hack--save important params in case they change in the global class
        self.n_possibilities = configuration.n_possibilities
        self.hidden_size = configuration.hidden_size
        self.num_layers = configuration.num_layers
        self.init_scale = configuration.init_scale
        self.mode = configuration.mode

# Create the main model
# Started from:
#   https://github.com/adventuresinML/adventures-in-ml-code/blob/master/lstm_tutorial.py
class Model(object):
    def __init__(self, config, vocab_size, is_training, is_generating=False):
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.mode = config.mode
        # below, only fed a single "word" per iteration when generating
        self.batch_size = 1 if is_generating else config.batch_size
        self.seq_length = 1 if is_generating else config.seq_length
        self.temperature = 1 if is_generating else config.temperature
        self.vocab_size = vocab_size

        self.X = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length])
        self.Y_true = tf.placeholder(
            tf.int32, [self.batch_size, self.seq_length])
        inputs = self._get_inputs(self.X, config)
        
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
        # set up the state storage / extraction
        # 2 states (cell, hidden) per layer, hence hardcoded 2
        self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.hidden_size])
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)])
        
        # check out ways to reorganize this, just seems like more parameters than
        # necessary...
        output, self.state = self._build_network(self.num_layers, self.hidden_size, inputs, rnn_tuple_state, is_training)

        softmax_w = tf.Variable(tf.random_uniform([self.hidden_size, self.vocab_size],
            -config.init_scale, config.init_scale))
        softmax_b = tf.Variable(tf.random_uniform([self.vocab_size],
            -config.init_scale, config.init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # temperature implementation:
        logits = tf.divide(logits, self.temperature)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits,
            [self.batch_size, self.seq_length, self.vocab_size])

        onehot_labels = tf.one_hot(self.Y_true, depth=self.vocab_size,
                                on_value=1.0, off_value=0.0,
                                axis=-1, dtype=tf.float32)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels,
            logits)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, self.vocab_size]))

        if is_training:
            self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        else:
            # use normalized probs of top k outputs to add some randomness to the output
            _, possibilities = tf.nn.top_k(self.softmax_out, k=config.n_possibilities, sorted=True)
            choice_reduce = lambda x: tf.py_func(np.random.choice,
                                                 [x, 1, True, tf.reverse(
                                                    tf.divide(
                                                        tf.cast(x, tf.float32),
                                                        tf.norm(tf.cast(x, tf.float32), ord=1)),
                                                    axis=[-1])],
                                                 tf.int32)  
            choices = tf.map_fn(choice_reduce, possibilities)
            self.predict = choices

        correct_prediction = tf.equal(self.predict, tf.reshape(self.Y_true, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def _build_network(self, num_layers, hidden_size, inputs, init_state, is_training):
        # helper function to avoid variable confusion
        # create an LSTM cell to be unrolled, add dropout wrapper if training
        def make_cell():
            cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
            if is_training and config.keep_prob < 1:
                return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            else:
                return cell

        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)],
                                           state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=init_state)
        # reshape to (batch_size * seq_length, self.hidden_size)
        output = tf.reshape(output, [-1, hidden_size])
        return output, state

    def _get_inputs(self, X, config):
        if config.mode == 'char':
            return tf.one_hot(self.X, depth=self.vocab_size,
                              on_value=1.0, off_value=0.0,
                              axis=-1, dtype=tf.float32)
        elif config.mode == 'word': # intuitively, embeddings only seem to make sense on a word-level model
            with tf.device("/cpu:0"):
            	embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.hidden_size], 
                                        -config.init_scale, config.init_scale))
            return tf.nn.embedding_lookup(embedding, self.X)
        else:
            raise ValueError('"mode" must be one of "word" or "char"')

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

# Does the work required in a single epoch
def run_epoch(session, lm, config, processed, epoch_size, prints_per_epoch=10, shuffle_iter=None):
    current_state = np.zeros((config.num_layers, 2, config.batch_size, lm.hidden_size))
    print_iter = max(1, epoch_size // prints_per_epoch) # ensure no modulo by zero
    total_prints = 0 # TODO: hack-y. do better.
    total_acc = 0
    print("epoch size:", epoch_size)
    curr_time = dt.datetime.now()
    i = 0 # TODO: maybe something better?
    # shuffle the data in blocks for better generalization
    for (X, Y) in processed.doc_slice(config.batch_size, config.seq_length, batches_to_reset=shuffle_iter):
        # below per theblog.github.io/post/character-language-model-lstm-tensorflow/
        if i % shuffle_iter == 0:
            current_state = np.zeros((config.num_layers, 2, config.batch_size, lm.hidden_size))
        if i % print_iter != 0:
            cost, _, current_state = session.run([lm.cost, lm.train_op, lm.state],
                                                feed_dict={lm.init_state: current_state, lm.X: X, lm.Y_true: Y})
        else:
            seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
            curr_time = dt.datetime.now()
            cost, _, current_state, acc = session.run([lm.cost, lm.train_op, lm.state, lm.accuracy],
                                                    feed_dict={lm.init_state: current_state, lm.X: X, lm.Y_true: Y})
            print("Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(i, cost, acc, seconds))
            total_acc += acc
            total_prints += 1
        i += 1
    # Seems like there should be a better fix for below
    return total_acc / total_prints

def train(model, model_name, processed, config=None, start_epoch=0):
    init_op = tf.global_variables_initializer()
    model_path = path_from_name(model_name)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        ### Handle training resumption if requested ###
        if start_epoch == 0:
            sess.run([init_op])
        else:
            # start w/data saved at end of former epoch
            version_path = model_path + '-{}'.format(start_epoch - 1)
            if os.path.exists(version_path + '.meta'):
                saver.restore(sess, version_path)
                with open(model_path + '.pkl', 'rb') as file:
                    config = pickle.load(file).config
            else:
                raise ValueError('Could not find a model path for epoch ' + str(start_epoch))
        
        # ensure learning rate is properly decayed on resume
        learning_rate = config.learning_rate * config.lr_decay ** max((start_epoch - 1) // config.max_epoch, 0)
        ###############################################

        ### save info for later generation ###
        meta = MetaModel(processed.word_to_id, processed.id_to_word, config)
        
        os.makedirs('../models/' + model_name, exist_ok=True) # ensure directory exists
        with open('{}.pkl'.format(model_path), 'wb') as outpath:
            pickle.dump(meta, outpath)
        ######################################

        ### do the actual training ###
        epoch_size = ((len(processed.data) // config.batch_size) - 1) // config.seq_length

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(start_epoch, config.max_max_epoch):
            model.assign_lr(sess, config.learning_rate)
            print("{s:#^80}\n".format(s=' Entering epoch {} '.format(epoch)))
            epoch_acc = run_epoch(sess, model, config, processed, epoch_size,
                                  prints_per_epoch=10,
                                  shuffle_iter=config.shuffle_iter)
            print("\n{s:#^80}\n".format(s=" Epoch accuracy {:.3f} ".format(epoch_acc))) # TODO: change nested .formats
            # save a model checkpoint
            saver.save(sess, model_path, global_step=epoch)
            if epoch % config.max_epoch == 0:
                learning_rate *= config.lr_decay
            # TODO: get validation accuracy here at each checkpoint
        coord.request_stop()
        coord.join(threads)
        ###############################
        
def path_from_name(name, version=None):
    if version is not None:
        return os.path.abspath('../models/{}/{}-{}'.format(name, name, version))
    else:
        return os.path.abspath('../models/{}/{}'.format(name, name))

def split_version(model_name):
    dash_index = model_name.rindex('-') # model name looks like [name]-[version_number]
    return (model_name[:dash_index], model_name[dash_index:])

def cycler(cycle_through):
    length = len(cycle_through)
    i = -1 # to start
    while True:
        i = (i + 1) % length
        yield cycle_through[i]

if __name__ == '__main__':
    ########## Setup ##########
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='Relative path to formatted text file')
    parser.add_argument('-r', '--resume', type=int, default=0, \
        help='The epoch at which to resume training if previously interrupted')
    parser.add_argument('-c', '--config', default='custom', \
        help='The configuration to use, one of small, medium, large, or custom')
    args = parser.parse_args()

    # below, if fed '../data/foobar.txt', get 'foobar'
    basepath = os.path.splitext(os.path.basename(args.filepath))[0]

    # load in data
    config = get_config(args.config)
    # process data in the background
    pool = ThreadPool(processes=1)
    processor = pool.apply_async(reader.DocReader, (args.filepath, config.mode))

    # make absolutely sure that there's a necessary spinning thing to look at
    #spinner = cycler(['\\', '|', '/', '-']) # original idea
    spinner = cycler(['o', '0', 'O', '0', 'o'])
    # processed = reader.DocReader(args.filepath, config.mode)
    while not processor.ready():
        sys.stdout.write('\rPr' + next(spinner) + 'cessing')
        sleep(0.1)

    processed = processor.get()
    print('\rPr' + next(spinner) + 'cessed.')

    print("Sanity check: {}...".format(processed.data[:50]))
    ########## Train ##########
    lm = Model(config, processed.vocab_size, is_training=True)
    train(lm, basepath, processed, config, start_epoch=args.resume)