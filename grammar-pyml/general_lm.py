import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import dynamic_rnn

import numpy as np
import datetime as dt

import argparse
import pickle
import os

import reader
from configs import *

# Stores all the additional information needed by generator.py to restore
#   functionality from a pre-trained model.
class MetaModel(object):
	def __init__(self, word_to_id, id_to_word, configuration):
		self.word_to_id = word_to_id
		self.id_to_word = id_to_word
		self.config = configuration

# Create the main model
# Started from:
#   https://github.com/adventuresinML/adventures-in-ml-code/blob/master/lstm_tutorial.py
class Model(object):
	def __init__(self, reader, config, is_training, is_generating=False):
		self.hidden_size = config.hidden_size
		self.vocab_size  = reader.vocab_size
		# below, only fed a single sentence during generation
		self.batch_size = config.batch_size if not is_generating else 1
		self.num_layers = config.num_layers
		self.num_steps = config.seq_length

		self.X, self.y_true = reader.batch_producer(self.batch_size, self.num_steps)

		# create the word embeddings
		with tf.device("/cpu:0"):
			embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.hidden_size], -config.init_scale, config.init_scale))
			inputs = tf.nn.embedding_lookup(embedding, self.X)

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		# set up the state storage / extraction
		# TODO: hardcoded 2??? maybe for cell/hidden state
		self.init_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.hidden_size])
		### This next sequence seems to confirm above todo
		state_per_layer_list = tf.unstack(self.init_state, axis=0)
		rnn_tuple_state = tuple(
			[tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
			 for idx in range(self.num_layers)]
		)
		
		# check out ways to reorganize this, just seems like more parameters than necessary...
		output, self.state = self.build_network(self.num_layers, self.hidden_size, inputs, rnn_tuple_state, is_training)

		softmax_w = tf.Variable(tf.random_uniform(
			[self.hidden_size, self.vocab_size],
			-config.init_scale, config.init_scale))
		softmax_b = tf.Variable(tf.random_uniform(
			[self.vocab_size],
			-config.init_scale, config.init_scale))
		logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
		# Reshape logits to be a 3-D tensor for sequence loss
		logits = tf.reshape(logits,
			[self.batch_size, self.num_steps, self.vocab_size])

		# Use the contrib sequence loss and average over the batches
		loss = tf.contrib.seq2seq.sequence_loss(
			logits,
			self.y_true,
			tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True)

		# Update the cost
		self.cost = tf.reduce_sum(loss)

		# get the prediction accuracy
		self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, self.vocab_size]))
		self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
		correct_prediction = tf.equal(self.predict, tf.reshape(self.y_true, [-1]))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		if not is_training:
		   return
		self.learning_rate = tf.Variable(0.0, trainable=False)

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)
		# optimizer = tf.train.AdamOptimizer(self.learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		self.train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.train.get_or_create_global_step())

		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(self.learning_rate, self.new_lr)

	def build_network(self, num_layers, hidden_size, inputs, init_state, is_training):
		# helper function to avoid variable confusion
		# create an LSTM cell to be unrolled, add dropout wrapper if training
		def make_cell():
			cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0)
			if is_training and config.keep_prob < 1:
				return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
			else:
				return cell

		cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)], state_is_tuple=True)

		output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=init_state)
		# reshape to (batch_size * seq_length, self.hidden_size)
		output = tf.reshape(output, [-1, hidden_size])
		return output, state

	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

def run_epoch(session, saver, lm, config, epoch_size, epoch_num, model_path, print_iter=100):
	current_state = np.zeros((config.num_layers, 2, config.batch_size, lm.hidden_size))
	curr_time = dt.datetime.now()
	for step in range(epoch_size):
		if step % print_iter != 0:
			cost, _, current_state = session.run([lm.cost, lm.train_op, lm.state],
												feed_dict={lm.init_state: current_state})
		else:
			seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
			curr_time = dt.datetime.now()
			cost, _, current_state, acc = session.run([lm.cost, lm.train_op, lm.state, lm.accuracy],
													feed_dict={lm.init_state: current_state})
			print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(
					epoch_num, step, cost, acc, seconds))

	# save a model checkpoint
	saver.save(session, model_path, global_step=epoch_num)

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
			version_path = model_path + '-{}'.format(start_epoch)
			if os.path.exists(version_path + '.meta'):
				saver.restore(sess, version_path)
				with open(model_path + '.pkl', 'rb') as file:
					config = pickle.load(file).config
			else:
				raise ValueError('Could not find a model path for epoch ' + str(start_epoch))

		# ensure learning rate is properly decayed on resume
		learning_rate = config.learning_rate * config.lr_decay ** max(start_epoch - config.max_epoch, 0)
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
			model.assign_lr(sess, learning_rate)
			# TODO: clearly refactor this...
			run_epoch(sess, saver, model, config, epoch_size, epoch, model_path)
			if epoch > config.max_epoch:
				learning_rate *= config.lr_decay
			# TODO: get validation accuracy here at each 
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
	processed = reader.DocReader(args.filepath)
	config = get_config(args.config)
	########## Train ##########
	lm = Model(processed, config, is_training=True)
	train(lm, basepath, processed, config, start_epoch=args.resume)