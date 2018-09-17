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
    temperature = 1
    mode = 'char'

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 3 # 2
    seq_length = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    temperature = 1
    mode = 'char'

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
    temperature = 1.2
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
    temperature = 1
    mode = 'char'

# These particular hyperparameters come from:
#  https://theblog.github.io/post/character-language-model-lstm-tensorflow/
class CustomConfig(object):
    init_scale = 0.05
    learning_rate = 0.001
    max_grad_norm = 10
    num_layers = 2
    seq_length = 160
    hidden_size = 795
    max_epoch = 14
    max_max_epoch = 40
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 100
    temperature = 10
    mode = 'char'

# Modified from the TensorFlow LSTM tutorial mentioned in the README
def get_config(name):
    if name == "custom":
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