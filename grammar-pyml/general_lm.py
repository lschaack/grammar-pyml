import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import dynamic_rnn

import numpy as np
import datetime as dt

import argparse
import pickle
import os

import reader

# Stores all the additional information needed by generator.py to restore
#   functionality from a pre-trained model.
class MetaModel(object):
    def __init__(self, word_to_id, id_to_word, configuration):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.config = configuration

class Config(object):
    """Adapted from medium config at:
      https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    seq_length = 35 # this is a more sensible/readable name than num_steps IMO
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    seq_length = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 2 # 1 # TODO: add support for 1-layer networks, clearly some confusion going on
    seq_length = 3 # 2
    hidden_size = 60 # 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20

def predict(x, weights, biases):
    x = np.reshape(np.array(x), [config.seq_length, 1])
    x = tf.split(x, config.seq_length, 1)

    rnn_cell = rnn.MultiRNNCell([
        rnn.BasicLSTMCell(config.hidden_size) for _ in range(config.num_layers)
    ])
    # do the actual predicting
    output, _ = dynamic_rnn(rnn_cell, x, dtype=tf.float32) # output, states
    return tf.nn.xw_plus_b(output, weights, biases)

# create the main model
# largely from (with substantial and ongoing changes):
#   https://github.com/adventuresinML/adventures-in-ml-code/blob/master/lstm_tutorial.py
class Model(object):
    def __init__(self, reader, config, is_training):
        self.is_training = is_training
        self.hidden_size = config.hidden_size
        self.vocab_size  = reader.vocab_size
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.num_steps = config.seq_length

        dropout = config.keep_prob

        # TODO: change these variable names
        self.input_data, self.targets = reader.batch_producer(self.batch_size, self.num_steps)
        #/TODO

        # create the word embeddings
        with tf.device("/cpu:0"):
            embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.hidden_size], -config.init_scale, config.init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        # TODO: hardcoded 2??? maybe for cell/hidden state
        self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.hidden_size])
        ### This next sequence seems to confirm above todo
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)], state_is_tuple=True)

        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, self.hidden_size)
        output = tf.reshape(output, [-1, self.hidden_size])

        softmax_w = tf.Variable(tf.random_uniform([self.hidden_size, self.vocab_size], -config.init_scale, config.init_scale))
        softmax_b = tf.Variable(tf.random_uniform([self.vocab_size], -config.init_scale, config.init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, self.vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def path_from_name(name):
    return os.path.abspath('../models/{}/{}'.format(name, name))

def train(model, model_name, processed, config=None, resume=False, start_epoch=0):
    init_op = tf.global_variables_initializer()
    model_path = path_from_name(model_name)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if not resume:
            sess.run([init_op])
        elif model_path is not None:
            # start w/data saved at end of former epoch
            saver.restore(sess, model_path + '-{}'.format(start_epoch - 1))
            with open(model_path + '.pkl', 'rb') as file:
                config = pickle.load(file).config
        else:
            raise ValueError('If resume is set to True, then model_path cannot be None')

        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        ### additional ###
        num_epochs = config.max_epoch # added
        # max_lr_epoch = 10 # TODO: maybe add this functionality back in?
        epoch_size = ((len(processed.data) // config.batch_size) - 1) // config.seq_length
        print_iter = 100

        meta = MetaModel(processed.word_to_id, processed.id_to_word, config)
        with open('{}.pkl'.format(model_path), 'wb') as outpath:
            pickle.dump(meta, outpath)
        ##################

        learning_rate = config.learning_rate
        learning_rate *= config.lr_decay ** start_epoch # make sure learning rate is properly decayed on resume

        for epoch in range(start_epoch, num_epochs):
            m.assign_lr(sess, learning_rate)
            # print(m.config.learning_rate.eval(), new_lr_decay)
            current_state = np.zeros((config.num_layers, 2, config.batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                            step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(sess, model_path, global_step=epoch)
            learning_rate *= config.lr_decay
        # do a final save
        saver.save(sess, model_path + '-final')
        # TODO: save meta-info with pickle: dict and reverse dict, config used, but maybe do this above since it won't change
        # close threads
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    ########## Setup ##########
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Relative path to formatted text file")
    args = parser.parse_args()

    # TODO: below will include the ../data/, so strip this. not much to log yet anyway...
    # below, if fed '../data/foobar.txt', get 'foobar'
    basepath = os.path.splitext(os.path.basename(args.filepath))[0]
    log_path = os.path.abspath('../logs/' + basepath)
    # writer = tf.summary.FileWriter(log_path)

    # load in data
    processed = reader.DocReader(args.filepath)
    config = SmallConfig()
    ########## Train ##########
    m = Model(processed, config, is_training=True)
    
    # train(m, basepath, processed, config)
    train(m, basepath, processed, resume=True, start_epoch=1)