import tensorflow as tf
import numpy as np

import general_lm
from general_lm import MetaModel
from configs import *
import reader
import formatter
from util import box

import argparse
import pickle
import os

def generate(model_path, meta, n_predict, input_file, wordy=False):
    processed = get_reader(input_file, meta)

    if wordy: print("got processed with data: {}\nLength of input data: {}".format(processed.data, len(processed.data)))

    config = meta.config
    config.n_possibilities = meta.n_possibilities
    config.hidden_size = meta.hidden_size
    config.num_layers = meta.num_layers
    config.init_scale = meta.init_scale
    config.mode = meta.mode

    lm = general_lm.Model(
        config,
        processed.vocab_size,
        is_training=False,
        is_generating=True)
    if wordy: print("finished creating model")
    saver = tf.train.Saver()

    with tf.Session() as session:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((lm.num_layers, 2, lm.batch_size, lm.hidden_size))

        # restore the trained model
        saver.restore(session, model_path)
        if wordy: print("restored session, predicting")

        sentence = ' '.join(processed.data) + '...'
        pred = None # for scope
        # set seed state
        for (X, Y) in processed.doc_slice(lm.batch_size, lm.seq_length):
            pred, current_state = session.run(
                [lm.predict, lm.state],
                feed_dict={lm.init_state: current_state, lm.X: X})
            
        # do the actual predicting
        for ii in range(n_predict):
            if wordy: print('making prediction {}'.format(ii))
            input = np.array(pred) # TODO: implicit shape? might be confusing (formerly [pred])
            pred, current_state = session.run(
                [lm.predict, lm.state],
                feed_dict={lm.init_state: current_state, lm.X: input})
            pred_word = meta.id_to_word[pred[0][0]] # formerly just one [0]
            sentence += ' ' * int(config.mode == 'word') # only add space in word mode
            sentence += pred_word.replace('_', ' ').replace('<eos>', os.linesep)
            if wordy: print('finished making prediction {}: {}'.format(ii, pred_word))

        #print("\n\t{}\n".format(sentence))
        print(box(sentence, alignment='left'))
        # close threads
        coord.request_stop()
        coord.join(threads)

# Construct input from a file if one is passed, otherwise ask the user for a seed sentence
def get_reader(input_file, meta):
    prefab_dicts=(meta.word_to_id, meta.id_to_word)
    if not input_file:
        cooperating = False
        while not cooperating:
            sentence = input('Please provide a "seed" phrase of length > 2 to begin: ')
            processed = reader.DocReader(
                sentence,
                meta.mode,
                input_is_string=True,
                prefab_dicts=prefab_dicts)
            if len(processed.data) > 1:
                cooperating = True
                return processed
    else:
        return reader.DocReader(
            os.path.abspath('../data/{}.txt'.format(input_file)),
            meta.mode,
            prefab_dicts=prefab_dicts)

def find_latest_version(model_name, max_version):
    for version_number in range(max_version, -1, -1):
        poss_path = general_lm.path_from_name(model_name, version=version_number)
        if os.path.exists(poss_path + ".meta"):
            return poss_path
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname',\
        help="Name of the model, typically the name of the originating text file without \".txt\"")
    parser.add_argument('-i', '--input', default=None, \
        help="The relative filepath to a formatted text file serving as a \"seed\" story providing a basis for model output")
    parser.add_argument('-n', '--npredict', type=int, default=100, \
        help="The number of words to predict")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Print a really excessive number of lines informing the user what is going on at any given point")
    args = parser.parse_args()

    with open('{}.pkl'.format(general_lm.path_from_name(args.modelname)), 'rb') as inpath:
        meta = pickle.load(inpath)

    model_path = find_latest_version(args.modelname, meta.config.max_max_epoch-1) # TODO: more elegant solution
    if model_path is None:
        raise ValueError("The name {} does not correspond to a trained model".format(args.modelname))

    generate(
        model_path,
        meta,
        args.npredict,
        input_file=args.input,
        wordy=args.verbose)