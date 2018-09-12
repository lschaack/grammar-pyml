import tensorflow as tf
import numpy as np

import general_lm
from general_lm import MetaModel
from configs import *
import reader

import argparse
import pickle
import os

def generate(model_path, meta, n_predict, input_file, wordy=False):
	sentence = "" # for scope

	processed = get_reader(input_file, meta)
	if wordy: print("Length of input data:", len(processed.data))

	m = general_lm.Model(processed, meta.config, is_training=False, is_generating=True)
	if wordy: print("finished creating model")
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# start threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		current_state = np.zeros((m.num_layers, 2, 1, m.hidden_size)) # effective batch size of 1, hence third arg

		# restore the trained model
		# TODO: something less static
		saver.restore(sess, model_path)
		if wordy: print("restored session, predicting")

		for ii in range(n_predict):
			if wordy: print('making prediction {}'.format(ii))
			pred, current_state = sess.run([m.predict, m.state],
											feed_dict={m.init_state: current_state})
			# pred_string = [processed.id_to_word[x] for x in pred[:m.num_steps]] # get the next word
			# sentence += " " + pred_string[-1]
			pred_word = meta.id_to_word[pred[-1]]
			sentence += " " + pred_word
			if wordy: print('finished making prediction {}: {}'.format(ii, pred_word))

		print("\n\t{}\n".format(sentence))
		# close threads
		coord.request_stop()
		coord.join(threads)

# TODO: change the desired_length param
def get_reader(input_file, meta):
	prefab_dicts=(meta.word_to_id, meta.id_to_word)
	if not input_file:
		cooperating = False
		min_length = meta.config.seq_length + 1
		while not cooperating:
			sentence = input('Please provide a "seed" phrase of at least {} words to begin: '.format(min_length))
			if len(sentence.split()) >= (min_length):
				cooperating = True
		return reader.DocReader(sentence, input_is_string=True, prefab_dicts=prefab_dicts)
	else:
		return reader.DocReader(os.path.abspath('../data/{}.txt'.format(input_file)), prefab_dicts=prefab_dicts)

def find_latest_version(model_name, max_version):
	for version_number in range(max_version, -1, -1):
		poss_path = general_lm.path_from_name(model_name, version=version_number)
		if os.path.exists(poss_path + ".meta"): # TODO: something less hack-y
			return poss_path
	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('modelname',\
		help="Name of the model, typically the name of the originating text file without \".txt\"")
	parser.add_argument('-i', '--input', default=None, \
		help="The relative filepath to a formatted text file serving as a \"seed\" story providing a basis for model output")
	parser.add_argument('-n', '--npredict', type=int, default=None, \
		help="The number of words to predict")
	parser.add_argument('-v', '--verbose', action='store_true',
		help="Print a really excessive number of lines informing the user what is going on at any given point")
	args = parser.parse_args()

	with open('{}.pkl'.format(general_lm.path_from_name(args.modelname)), 'rb') as inpath:
		meta = pickle.load(inpath)

	# automatically find latest version
	model_path = find_latest_version(args.modelname, meta.config.max_max_epoch-1)
	if model_path is None:
		raise ValueError("The name {} does not correspond to a trained model".format(args.modelname))

	generate(model_path, meta, args.npredict or 2 * meta.config.seq_length, input_file=args.input, wordy=args.verbose)