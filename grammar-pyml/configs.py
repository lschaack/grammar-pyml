"""
init_scale sets the bounds of the uniform distribution used to initialize vars
learning_rate is exactly what it sounds like, used for the Adam optimizer
max_grad_norm is the value at which gradients are clipped
    note above is probably not as necessary w/high batch sizes
num_layers is just the number of hidden layers
hidden_size is just the number of nodes per layer
max_epoch is the number of iterations between applying lr_decay
    note above is different from original usage
    w/max_epoch = 20, then every 20th epoch learning_rate = learning_rate * lr_decay
max_max_epoch is the total number of epochs to run before stopping automatically
keep_prob is (1 - dropout), helps prevent overfitting
lr_decay is the amount that learning_rate is decayed every "max_epoch'th" epoch
batch_size is the number of sequences used to calculate the gradient at each step
temperature is a divisor for logits before the softmax layer
n_possibilities is the number of outputs to choose from (using norm'd probs)
mode determines whether a word- or character-level model is create
"""

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 1
    num_layers = 2
    seq_length = 3
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 2
    shuffle_iter = 1
    temperature = 1
    n_possibilities = 2
    mode = 'char'

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 2
    seq_length = 10
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    shuffle_iter = 5
    temperature = 1
    n_possibilities = 2
    mode = 'word'

class MediumConfig(object): # good for shakespeare.txt
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 10
    num_layers = 2
    seq_length = 35
    hidden_size = 650
    max_epoch = 14
    max_max_epoch = 50
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 5
    shuffle_iter = 20
    temperature = 1.2
    n_possibilities = 2
    mode = 'char'

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10
    num_layers = 2
    seq_length = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    shuffle_iter = 20
    temperature = 1
    n_possibilities = 2
    mode = 'char'

# These particular hyperparameters come from:
#  https://theblog.github.io/post/character-language-model-lstm-tensorflow/
class CustomConfig(object):
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 10
    num_layers = 3
    seq_length = 160
    hidden_size = 512
    max_epoch = 30
    max_max_epoch = 100
    keep_prob = 0.5
    lr_decay = 0.5
    batch_size = 100
    shuffle_iter = 100
    temperature = 1
    n_possibilities = 2
    mode = 'char'

# Andrej Karpathy 1mb settings (more or less, lr calculations are different)
class KarpathyConfig(object):
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 3 # 2-layer for ~1mb, 3-layer for 4+mb, same hidden size
    seq_length = 100
    hidden_size = 512
    max_epoch = 15
    max_max_epoch = 55
    keep_prob = 0.5
    lr_decay = 0.5
    batch_size = 100
    shuffle_iter = 20
    temperature = 1
    n_possibilities = 2
    mode = 'char'

# Modified from the TensorFlow LSTM tutorial mentioned in the README
def get_config(name):
    if name == "custom":
        return CustomConfig()
    elif name == "karpathy":
        return KarpathyConfig()
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