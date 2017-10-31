#!/usr/bin/env python2
# -*- coding: utf-8 -*-

######
# Execution file for the LSTM attention model
# Based on starter code from PS3-CS224n
######
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
import os
from datetime import datetime

import tensorflow as tf
import numpy as np
import cPickle as pickle

from run_text_processing import get_data, save_data_pickle

from our_util import Progbar, minibatches, pack_labels, split_data, softmax, get_performance, convertOutputs, downsample_label
# from our_model import OurModel, Config

from LSTM_attention import *

logger = logging.getLogger("hw3.q3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def run_save_data_pickle():
    save_data_pickle(outfilename = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                    embedding_type = 'twitter.27B.50d',
                    parserOption = 'nltk')
    
def run_lstm_attention(config, outputpath = '../../xp', final = False):
    config, data_dict = get_data(config,
                                filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
                                pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                                concat = True)

    y = data_dict['y']
    h_b_np = data_dict['h_b_np']
    seqlen = data_dict['seqlen']

    # Perform downsampling
    if 'downsample' in config.__dict__:
        if config.downsample == True:
            downsample_indices = downsample_label(y, label_for_ds = 3, downsample_factor = 4)
            y = y[downsample_indices]
            h_b_np = h_b_np[downsample_indices, :]
            seqlen = seqlen[downsample_indices]

    if config.max_length is not None:
        max_length = config.max_length
        if np.shape(h_b_np)[1] > max_length:
            h_b_np = h_b_np[:, 0:max_length]
        seqlen = np.minimum(seqlen, max_length)

    # Set maximum dataset size for testing purposes
    data = pack_labels(h_b_np, y, seqlen)
    if config.num_samples is not None:
        num_samples = config.num_samples
        data = data[0:num_samples - 1]

    # Split data, result is still packed
    train_data, dev_data, test_data, train_indices, dev_indices, test_indices = split_data(data, prop_train = 0.6, prop_dev = 0.2, seed = 56)

    # Compute some convenience sub-sets
    # Dev
    dev_labels = y[dev_indices]
    dev_data_np = h_b_np[dev_indices, :]
    dev_seqlen = seqlen[dev_indices]
    # Test
    test_labels = y[test_indices]
    test_data_np = h_b_np[test_indices, :]
    test_seqlen = seqlen[test_indices]

    ## Config determined at data loading:
    config.num_samples = len(train_indices)
    config.max_length = np.shape(h_b_np)[1]


    # If this is the final test:
        # Combine test and dev
        # Reassign test to dev - for compatibility with rest of the code
    if final:
        # train_dev_indices = train_indices.extend(dev_indices)
        train_dev_indices = train_indices + dev_indices
        train_data = [data[i] for i in train_dev_indices]
        dev_data_np = test_data_np
        dev_seqlen = test_seqlen
        dev_labels = test_labels
        config.num_samples = len(train_dev_indices)

    with tf.Graph().as_default():
        
        tf.set_random_seed(59)

        logger.info("Building model...",)
        start = time.time()
        model = LSTMAttention(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)
            # losses = model.fit(session, train_data)
            losses_ep, dev_performances_ep, dev_predicted_classes_ep, dev_predictions_ep = model.fit(session, train_data, dev_data_np, dev_seqlen, dev_labels) # MODIF
            # dev_predictions = model.predict_on_batch(session, dev_data_np, dev_seqlen)


            #test_predictions = model.predict_on_batch(session, test_data_np, test_seqlen)

    # outputpath = '../../xp' # MODIF
    convertOutputs(outputpath, config, losses_ep, dev_performances_ep) # MODIF
    # Compute testing predictions --> MODIF --> SHOULD BE REMOVED WHEN OK
    print('Dev Performance ', dev_performances_ep) #M
    return losses_ep, dev_predicted_classes_ep, dev_performances_ep #MODIF

if __name__ == "__main__":
    
    # print('Doing something!')
    # # run_save_data_pickle()
    # # test_model_loading_functions('')
    # # test_run_model_with_parameters('')
    # # test_save_load_data_pickle('twitter50d_h_ids_b_ids_pickle.p')
    # # losses = test_model_with_real_data_pickle('args')
    print('Execution Complete')
    # # print(losses)