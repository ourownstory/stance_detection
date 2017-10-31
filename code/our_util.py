#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions
"""

from __future__ import division

import sys
import time
import logging
import StringIO
import pandas as pd
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose


def split_data(data, prop_train = 0.6, prop_dev = 0.2, seed = None):
    ## Generate hold-out data
    np.random.seed(seed)
    # If data is a numpy object

    assert prop_train + prop_dev <= 1

    if (type(data).__module__ == np.__name__):
        
        num_samples = data.shape[0]
        num_train_samples = int(np.floor(num_samples * prop_train))
        num_dev_samples = int(np.floor(num_samples * prop_dev))

        indices = range(num_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[0:num_train_samples]
        dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
        test_indices = indices[num_train_samples+num_dev_samples:num_samples]

        train_data = data[indices[train_indices], :]
        dev_data = data[indices[dev_indices], :]
        test_data = data[indices[test_indices], :]
    
    elif isinstance(data, list):
        
        num_samples = len(data)
        num_train_samples = int(np.floor(num_samples * prop_train))
        num_dev_samples = int(np.floor(num_samples * prop_dev))
        
        indices = range(num_samples)
        np.random.shuffle(indices)

        # train_indices = indices[range(num_train_samples)]
        train_indices = indices[0:num_train_samples]
        dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
        test_indices = indices[num_train_samples+num_dev_samples:num_samples]

        train_data = [data[i] for i in train_indices]
        dev_data = [data[i] for i in dev_indices]
        test_data = [data[i] for i in test_indices]

    return train_data, dev_data, test_data, train_indices, dev_indices, test_indices,

def split_indices(num_samples, prop_train = 0.6, prop_dev = 0.2):
    num_train_samples = int(np.floor(num_samples * prop_train))
    num_dev_samples = int(np.floor(num_samples * prop_dev))
    indices = range(num_samples)
    np.random.shuffle(indices)
    train_indices = indices[0:num_train_samples]
    dev_indices = indices[num_train_samples:num_train_samples + num_dev_samples]
    test_indices = indices[num_train_samples + num_dev_samples:num_samples]
    return train_indices, dev_indices, test_indices

def test_data_splitting(data):
    test_data, train_data = split_data(data)
    print 'Full data' + str(len(data))
    print 'Test' + str(len(test_data))
    print 'Train' + str(len(train_data))

# Returns a list of indices that should remain in the dataset
def downsample_label(y, label_for_ds = 3, downsample_factor = 4):
    y = np.asarray(y)
    indices = np.asarray(range(len(y)))
    indices_to_sample = indices[y == label_for_ds]
    n_samples = int(np.floor(len(indices_to_sample)/downsample_factor))
    sampled_indices = np.random.choice(indices_to_sample, size = n_samples, replace = False)
    output = np.append(indices[y != label_for_ds], sampled_indices)
    return(output)

def pack_labels(data, labels, seqlen): # MODIF
    output = []
    num_rows = data.shape[0]
    assert num_rows == len(labels)
    for i in range(data.shape[0]):
        the_row = data[i, :]
        output.append((the_row, labels[i], seqlen[i]))
    return output

def softmax(x):
    """Compute the softmax function for each row of the input x.
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = x - np.amax(x, axis = 1).reshape(x.shape[0], 1)
        rowSums = np.sum(np.exp(x), axis = 1).reshape(x.shape[0], 1)
        x = np.exp(x) / rowSums
    else:
        # Vector
        x = x - np.max(x)
        theSum = np.sum(np.exp(x))
        x = np.exp(x) / theSum

    assert x.shape == orig_shape
    return x

# Compute performance metrics
def get_performance(predicted, truth, n_classes = None, outputStyle = 'dict'):
    # Predicted and observed are both integer vectors of class label

    # Cast both predicted and observed to numpy integer
    predicted = np.asarray(predicted, dtype = np.int64)
    truth = np.asarray(truth, dtype = np.int64)

    assert len(predicted) == len(truth)

    # Compute competition score:
    competition_score = scorer(predicted, truth)

    output = []
    # If n_classes is unknown, infer from the labels
    if n_classes is None:
        n_classes = len(np.unique(predicted.extend(truth)))
    
    for i in range(n_classes):

        # Get 2-way table
        tp = sum((predicted == i) & (truth == i))
        tn = sum((predicted != i) & (truth != i))
        fp = sum((predicted == i) & (truth != i))
        fn = sum((predicted != i) & (truth == i))

        print 'tp ' + str(tp)
        print 'tn ' + str(tn)
        print 'fp ' + str(fp)
        print 'fn ' + str(fn)

        # Compute performance metrics
        recall = tp / (tp + fn) # aka sensitivity
        print 'recall ' + str(recall)
        precision = tp / (tp + fp) # aka ppv
        print 'precision ' + str(precision)
        specificity = tn / (tn + fp)
        print 'specificity ' + str(specificity)
        f1 = 2 * tp / (2 * tp + fp + fn)
        print 'f1 ' + str(f1)
        accuracy = (tp + tn)/len(truth)
        
        keys = ['tp', 'tn', 'fp', 'fn', 'recall', 'precision', 'specificity', 'f1', 'accuracy', 'competition']
        values = [tp, tn , fp, fn, recall, precision, specificity, f1, accuracy, competition_score]
        output.append(dict(zip(keys, values)))

    return output

# Computes competition score
def scorer(pred, truth):
    # Maximum possible score
    max_score = 0.25 * sum(truth == 3) + 1 * sum(truth != 3)
    # Computing achieved sore
    # Score from unrelated correct
    unrelated_score = 0.25 * sum((truth == 3) & (pred == truth))
    # Score from related correct, but specific class incorrect
    related_score1 = 0.25 * sum((truth != 3) & (pred != truth) & (pred != 3))
    # Score from getting related correct, specific class correct
    related_score2 = 0.75 * sum((truth != 3) & (pred == truth))

    final_score = (unrelated_score + related_score1 + related_score2) / max_score
    return final_score

def convertOutputs(outputpath, config, losses_ep, dev_performances_ep): #MODIF

    '''
    Inputs are lists of length n_epochs
        - losses_ep: list. losses_ep[i][j] --> loss after batch j
        - dev_performances_ep: dictionnary
        - dev_predicted_classes_ep: np.array
        - dev_predictions_ep: np.array
    '''        
         
    # Define parameter keys
    parameter_keys = dir(config)
    params_remove = ['__doc__', '__module__','pretrained_embeddings']
    parameter_keys = [param for param in parameter_keys if param not in params_remove]
    print('parameter_keys', parameter_keys)

    n_epochs = getattr(config,'n_epochs')
    
    # Define column names
    common_keys = parameter_keys + ['epoch'] # Common keys to all csv files
    performance_keys = (dev_performances_ep[0][0]).keys() # [0] for epoch / [0] for 1st class
                       # Keys specific to performance output
    
    # Initialization        
    performances_pds = []

    for i in range(n_epochs):   
        # Performance csv
        performance_pd = pd.DataFrame(index = range(4), columns = common_keys + ['class'] + performance_keys)
        performance_pd['class'] = range(4)
        for j, outp in enumerate(dev_performances_ep[i]):
            for key in outp.keys():
                performance_pd.loc[j, key] = outp[key]
        performance_pd['epoch'] = i
        performance_pd['train_loss'] = 1.0 * sum(losses_ep[i]) / len(losses_ep[i])
        performances_pds.append(performance_pd)
    # Append all dataframes
    performance_pd_global = pd.concat(performances_pds, axis = 0)
                
    # Loss dataframe
    losses_pd_global = pd.DataFrame(columns = common_keys + ['loss'])
    losses_ep = np.array(losses_ep)
    losses_pd_global['epoch'] = range(1, n_epochs+1)
    losses_pd_global['loss'] = np.mean(losses_ep, axis = 1)
    
    # Adding parameter columns
    output_pds = [performance_pd_global, losses_pd_global]
    for par_name in parameter_keys:
        for output_pd in output_pds:
            output_pd[par_name] = getattr(config,par_name)
    
    # --- Writing to csv ---
    performance_pd_global.to_csv(outputpath+'/perf_'+ str(time.time()).replace('.','') + '.csv',index = False)
    losses_pd_global.to_csv(outputpath+'/losses_'+ str(time.time()).replace('.','') + '.csv', index = False)


# BACK-UP FUNCTION
def convertOutputs0(outputpath, config, losses_ep, dev_performances_ep): #MODIF

    '''
    Inputs are lists of length n_epochs
        - losses_ep: list. losses_ep[i][j] --> loss after batch j
        - dev_performances_ep: dictionnary
        - dev_predicted_classes_ep: np.array
        - dev_predictions_ep: np.array
    '''        
         
    # Define parameter dict
    parameter_dict = config.__dict__
    parameter_dict.pop('pretrained_embeddings', None) # Removing embedding matrix
    # Added line to handle list-valued parameter
    # if 'extra_hidden_size' in parameter_dict & parameter_dict['extra_hidden_size'] is not None:
    #     parameter_dict['extra_hidden_size'] = str(parameter_dict['extra_hidden_size'])
    parameter_keys = parameter_dict.keys()
    print('parameter_keys', parameter_keys)
    n_epochs = parameter_dict['n_epochs']
    
    # Define column names
    common_keys = parameter_keys + ['epoch'] # Common keys to all csv files
    performance_keys = (dev_performances_ep[0][0]).keys() # [0] for epoch / [0] for 1st class
                       # Keys specific to performance output
    
    # Initialization        
    performances_pds = []

    for i in range(n_epochs):   
        # Performance csv
        performance_pd = pd.DataFrame(index = range(4), columns = common_keys + ['class'] + performance_keys)
        performance_pd['class'] = range(4)
        for j, outp in enumerate(dev_performances_ep[i]):
            for key in outp.keys():
                performance_pd.loc[j, key] = outp[key]
        performance_pd['epoch'] = i
        performances_pds.append(performance_pd)
    # Append all dataframes
    performance_pd_global = pd.concat(performances_pds, axis = 0)
                
    # Loss dataframe
    losses_pd_global = pd.DataFrame(columns = common_keys + ['loss'])
    losses_ep = np.array(losses_ep)
    losses_pd_global['epoch'] = range(1, n_epochs+1)
    losses_pd_global['loss'] = np.mean(losses_ep, axis = 1)
    
    # Adding parameter columns
    output_pds = [performance_pd_global, losses_pd_global]
    for par_name in parameter_keys:
        for output_pd in output_pds:
            output_pd[par_name] = parameter_dict[par_name]
    
    # --- Writing to csv ---
    performance_pd_global.to_csv(outputpath+'/perf_'+ str(time.time()).replace('.','') + '.csv',index = False)
    losses_pd_global.to_csv(outputpath+'/losses_'+ str(time.time()).replace('.','') + '.csv', index = False)

# Ferdinand
def get_minibatches(data, minibatch_size, shuffle=True):
    
    '''
    MODIF
    Assuming we have a list [examples, labels, seqlen] of np.array
    '''

    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        
        if list_data:
            examples_minibatch = minibatch(data[0], minibatch_indices) # np.array of shape (batch_size, max_length_global)
            labels_minibatch = minibatch(data[1], minibatch_indices)
            seqlen_minibatch = minibatch(data[2], minibatch_indices)
            
            # Truncating sentences to the max_length of the minibatch --> NOT HERE, placeholders have fixed side
            #max_len_minibatch = max(seqlen_minibatch)
            #examples_minibatch = examples_minibatch[:,:max_len_minibatch]
            
            yield [examples_minibatch, labels_minibatch, seqlen_minibatch]
        
        else: # no truncating if data not in the 'packed' list format [examples, labels, seqlen]
            yield minibatch(data, minibatch_indices)


## Derived from Stanford CS 224n started code provided for assignment 3.
def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)
