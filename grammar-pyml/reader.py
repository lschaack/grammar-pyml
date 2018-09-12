import collections
import os
import sys

import random
import tensorflow as tf

### Much of this class annotated and modified from:
#   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
class DocReader(object):
	def __init__(self, to_read, input_is_string=False, prefab_dicts=None):
		self.data = self._read_words(to_read) if input_is_string else self._read_file(to_read)
		
		if prefab_dicts is None:
			self.word_to_id, self.id_to_word = self._build_dicts(self.data)
		elif prefab_dicts is not None:
			self.word_to_id, self.id_to_word = prefab_dicts
		elif input_is_string:
			raise ValueError('If DocReader input is a string, it must come with a',\
							 'tuple of prefab word_to_id and id_to_word dictionaries')

		self.vocab_size = len(self.word_to_id)

	def _read_words(self, words):
		return words.replace("\n", " <eos> ").split()

	def _read_file(self, filename):
		with open(os.path.abspath(filename), "r", encoding="utf-8") as f:
			return self._read_words(f.read())
	
	def _build_dicts(self, data):
		counter = collections.Counter(data)
		# counter.items() is a list of tuples like [('word', integer_count)]
		# sort by descending frequency, then alphabetically for ties
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(len(words))))
		id_to_word = dict(zip(range(len(words)), words))

		return (word_to_id, id_to_word)

	# Generates tuples of ([context_word_ids], correct_next_word_index) intended as (X, y)
	# probably fully deprecated
	def doc_slice(self, batch_length, seq_length):
		random_pad = random.randint(0, seq_length + 1)
		# since range interval is half-open like [,)
		#   this shouldn't raise IndexError so long as batch_length < seq_length * 2
		for batch_number in range(len(self.data) // batch_length):
			offset = batch_number * batch_length + random_pad
			
			yield (
				[
					self.word_to_id[self.data[index]]
					for index in range(offset, offset + seq_length)
				], 
				self.word_to_id[self.data[offset + seq_length]]
			)

	# pretty much directly from:
	#  http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
	def batch_producer(self, batch_size, seq_length):
		raw_data = [self.word_to_id[word] for word in self.data]
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

		data_len = tf.size(raw_data)
		batch_len = data_len // batch_size
		data = tf.reshape(raw_data[0: batch_size * batch_len],
						[batch_size, batch_len])

		epoch_size = (batch_len - 1) // seq_length

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = data[:, i * seq_length:(i + 1) * seq_length]
		x.set_shape([batch_size, seq_length])
		y = data[:, i * seq_length + 1: (i + 1) * seq_length + 1]
		y.set_shape([batch_size, seq_length])

		return x, y
