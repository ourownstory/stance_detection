#!/usr/bin/env python2
# -*- coding: utf-8 -*-
######
# Initial test of LSTM model for Fake News Challenge
# Based on starter code from PS3-CS224n
# Based on Stephen's rnn_test1
######
## General libraries
import tensorflow as tf
import numpy as np
import random

## Our Own Code
# from our_model import Config
from bow_model_config import BOWModel
from run_text_processing import save_data_pickle, get_data
# from run_text_processing import get_data
## currently using: split_indices
# from our_util import Progbar, minibatches, pack_labels, split_data, split_indices, softmax, get_performance
from our_util import split_indices, softmax, get_performance, convertOutputs #M

def run_save_data_pickle(): ## Needs NLTK to be installed!
    save_data_pickle(outfilename = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                    embedding_type = 'twitter.27B.50d',
                    parserOption = 'nltk')

def run_bow(config, split = True, outputpath = '../../xp', final = False): #M



    ## Get data
    # config, y, h, b, h_len, b_len = get_BOW_data(config, reload = True, save_data = False)
    config, data_dict = get_data(config, 
            filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
            pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
            concat = False)

    ## pass data into local namespace:
    y = data_dict['y']
    h = data_dict['h_np']
    b = data_dict['b_np']
    h_len = data_dict['h_seqlen']
    b_len = data_dict['b_seqlen']
    
    # Do shortening of dataset ## affects number of samples and max_len.
    if config.num_samples  is not None:
        ## Random seed
        np.random.seed(1)
        ind = range(np.shape(h)[0])
        random.shuffle(ind)
        indices = ind[0:config.num_samples ]
        h = h[indices,:]
        b = b[indices,:]
        h_len = h_len[indices]
        b_len = b_len[indices]
        y = y[indices]

    if config.h_max_len is not None:
        h_max_len = config.h_max_len
        if np.shape(h)[1] > h_max_len:
            h = h[:, 0:h_max_len]
        h_len = np.minimum(h_len, h_max_len)

    if config.b_max_len is not None:
        b_max_len = config.b_max_len
        if np.shape(b)[1] > b_max_len:
            b = b[:, 0:b_max_len]
        b_len = np.minimum(b_len, b_max_len)

    if split:
        # Split data
        train_indices, dev_indices, test_indices = split_indices(np.shape(h)[0])
        # Divide data
        train_h = h[train_indices,:]
        train_b = b[train_indices,:]
        train_h_len = h_len[train_indices]
        train_b_len = b_len[train_indices]
        train_y = y[train_indices]

        # Development
        dev_h = h[dev_indices,:]
        dev_b = b[dev_indices,:]
        dev_h_len = h_len[dev_indices]
        dev_b_len = b_len[dev_indices]
        dev_y = y[dev_indices]

        if final:
            # Combine train and dev
            train_dev_indices = train_indices + dev_indices
            train_h = h[train_dev_indices,:]
            train_b = b[train_dev_indices,:]
            train_h_len = h_len[train_dev_indices]
            train_b_len = b_len[train_dev_indices]
            train_y = y[train_dev_indices]

            # Set dev to test
            dev_h = h[test_indices,:]
            dev_b = b[test_indices,:]
            dev_h_len = h_len[test_indices]
            dev_b_len = b_len[test_indices]
            dev_y = y[test_indices]


      
    ## Passing parameter_dict to config settings
    ## Changes to config  based on data shape
    assert(np.shape(train_h)[0] == np.shape(train_b)[0] == np.shape(train_y)[0] == np.shape(train_h_len)[0] == np.shape(train_b_len)[0])
    config.num_samples = np.shape(train_h)[0]
    config.h_max_len = np.shape(train_h)[1]
    config.b_max_len = np.shape(train_b)[1]
    
    ## Start Tensorflow!
    print('Starting TensorFlow operations')
    print 'With hidden layers: ', config.n_layers ## hidden layer?
    with tf.Graph().as_default():
        tf.set_random_seed(1)
        model = BOWModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            losses_ep, dev_performances_ep, dev_predicted_classes_ep, dev_predictions_ep = model.fit(session, train_h, train_b, train_h_len, train_b_len, train_y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y) #M

    # Write results to csv
    convertOutputs(outputpath, config, losses_ep, dev_performances_ep)

    print('Losses ', losses_ep)
    print('Dev Performance ', dev_performances_ep) #M
    return losses_ep, dev_predicted_classes_ep, dev_performances_ep #M

## for debugging
if __name__ == "__main__":
    print('Doing something!')
    losses, dev_predicted_classes, dev_performance = run_bow(num_samples = 1028)
    print('Execution Complete')
