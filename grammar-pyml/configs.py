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

class SmallConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 3 # 2
	seq_length = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 20

class MediumConfig(object):
	"""Adapted from medium config at:
	  https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py"""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	seq_length = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20

class LargeConfig(object):
	"""Large config."""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	seq_length = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1 / 1.15
	batch_size = 20

class CustomConfig(object):
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 15
	num_layers = 4
	seq_length = 35
	hidden_size = 650
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.30
	lr_decay = 1 / 1.15
	batch_size = 20

# Modified from the TensorFlow LSTM tutorial mentioned in the README
def get_config(name):
	if name =="custom":
		return CustomConfig()
	elif name == "small":
		return SmallConfig()
	elif name == "medium":
		return MediumConfig()
	elif name == "large":
		return LargeConfig()
	elif name == "test":
		return TestConfig()
	else:
		raise ValueError("Invalid configuration: %s", name)