import collections
import os
import sys

import random
import numpy as np

import formatter

### Much of this class annotated and modified from:
#   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
class DocReader(object):
    def __init__(self, to_read, unk_threshold=0, input_is_string=False, prefab_dicts=None):
        self.data = self._read_words(to_read, unk_threshold) if input_is_string else self._read_file(to_read, unk_threshold)
        
        if prefab_dicts is None:
            self.word_to_id, self.id_to_word = self._build_dicts(self.data)
        elif prefab_dicts is not None:
            self.word_to_id, self.id_to_word = prefab_dicts
        elif input_is_string:
            raise ValueError('If DocReader input is a string, it must come with a',\
                             'tuple of prefab word_to_id and id_to_word dictionaries')

        self.vocab_size = len(self.word_to_id)

    def _read_words(self, words, unk_threshold):
        formatted = formatter.get_formatted_text(words, unk_threshold)
        return formatted.replace("\n", " <eos> ").split()

    def _read_file(self, filename, unk_threshold):
        with open(os.path.abspath(filename), "r", encoding="utf-8") as f:
            return self._read_words(f.read(), unk_threshold)
    
    def _build_dicts(self, data):
        counter = collections.Counter(data)
        # counter.items() is a list of tuples like [('word', integer_count)]
        # sort by descending frequency, then alphabetically for ties
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = dict(zip(range(len(words)), words))

        return (word_to_id, id_to_word)

    # Generator for batches
    # steps_to_reset is the number of batches to run through before a new
    #   random block is selected (shuffled in chunks of steps_to_reset batches)
    def doc_slice(self, batch_size, seq_length, batches_to_reset=None):
        raw_data = [self.word_to_id[word] for word in self.data]
        data_len = len(raw_data)
        raw_data = np.array(raw_data, dtype=np.int32)

        batch_len = data_len // batch_size
        data = np.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // seq_length
        
        if batches_to_reset is None:
           batches_to_reset = epoch_size
        num_blocks = epoch_size // batches_to_reset
        offsets = list(range(num_blocks))
        np.random.shuffle(offsets)

        for offset in offsets:
            for batch_number in range(batches_to_reset):
                start_index = (offset * batches_to_reset + batch_number) * seq_length
                end_index = start_index + seq_length

                x = data[:, start_index:end_index]
                x.reshape([batch_size, seq_length])
                y = data[:, start_index + 1:end_index + 1]
                y.reshape([batch_size, seq_length])

                yield x, y
