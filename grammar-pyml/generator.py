import tensorflow as tf
import numpy as np

import general_lm
from general_lm import MetaModel, TestConfig, SmallConfig, Config
import reader

import argparse
import pickle
import os

N_PREDICT = 20

def generate(model_name, wordy=False):
    model_path = general_lm.path_from_name(model_name)

    with open('{}.pkl'.format(model_path), 'rb') as inpath:
        meta = pickle.load(inpath)

    config = meta.config
    config.batch_size = 1 # TODO: something better...

    sentence = "" # for scope
    cooperating = False
    while not cooperating:
        sentence = input('Please provide a "seed" phrase of at least {} words to begin: '.format(2 * config.seq_length + 1))
        if len(sentence.split()) > (2 * config.seq_length):
            cooperating = True
    processed = reader.DocReader(sentence, input_is_string=True, prefab_dicts=(meta.word_to_id, meta.id_to_word))
    if wordy: print("updated processed.data with input \"{}\"".format(processed.data))

    m = general_lm.Model(processed, config, is_training=False)
    if wordy: print("finished creating model")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        # TODO: something less static
        model_path = os.path.abspath('{}-final'.format(model_path))
        saver.restore(sess, model_path)
        if wordy: print("restored session, predicting")

        for ii in range(N_PREDICT):
            if wordy: print('making prediction {}'.format(ii))
            pred, current_state = sess.run([m.predict, m.state],
                                            feed_dict={m.init_state: current_state})
            # pred_string = [processed.id_to_word[x] for x in pred[:m.num_steps]] # get the next word
            # sentence += " " + pred_string[-1]
            pred_word = processed.id_to_word[pred[-1]]
            sentence += " " + pred_word
            if wordy: print('finished making prediction {}: {}'.format(ii, pred_word))

        print("\n\t{}\n".format(sentence))
        # close threads
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname',\
        help="Name of the model, typically the name of the originating text file without \".txt\"")
    args = parser.parse_args()

    generate(args.modelname, wordy=True)