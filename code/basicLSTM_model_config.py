#!/usr/bin/env python2
# -*- coding: utf-8 -*-

######
# Model class for Baseline_LSTM
# Based on starter code from PS3-CS224n
######
from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np

from our_util import Progbar, minibatches, get_performance, softmax
from our_model_config import OurModel

logger = logging.getLogger("hw3.q3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class BaselineLSTM(OurModel):

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        MODIF: OVERWRITING
        """
        self.inputs_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.max_length), name = "x")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name = "y")
        self.seqlen_placeholder = tf.placeholder(tf.int64, shape=(None), name = "seqlen")
        self.dropout_placeholder = tf.placeholder(tf.float64, name = 'dropout')
    
    def create_feed_dict(self, inputs_batch, seqlen_batch, labels_batch = None, dropout = 1.0):
        """Creates the feed_dict for the model.
        MODIF: OVERWRITING
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        feed_dict[self.seqlen_placeholder] = seqlen_batch
        return feed_dict

    def add_prediction_op(self):
        """
        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        if self.config.n_layers <= 1:
            print('layers = ', self.config.n_layers)
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout_placeholder)
            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)
            U = tf.get_variable(name = 'U', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b = tf.get_variable(name = 'b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)

            x = self.add_embedding(option = self.config.trainable_embeddings)
            rnnOutput = tf.nn.dynamic_rnn(cell, inputs = x, dtype = tf.float64, sequence_length = self.seqlen_placeholder) #MODIF
            finalState = rnnOutput[1][1] # batch_size, cell.state_size
            preds = tf.matmul(finalState, U) + b # batch_size, n_classes
        # print('Predict op: preds', preds)
        elif self.config.n_layers > 1: # MODIF
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = self.dropout_placeholder)
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.n_layers)
            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)
            U = tf.get_variable(name = 'U', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b = tf.get_variable(name = 'b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            x = self.add_embedding(option = self.config.trainable_embeddings)
            rnnOutput = tf.nn.dynamic_rnn(stacked_lstm, inputs = x, dtype = tf.float64, sequence_length = self.seqlen_placeholder) #MODIF
            print('layers = ', self.config.n_layers)
            finalState = rnnOutput[1][self.config.n_layers - 1][1] # batch_size, cell.state_size
            preds = tf.matmul(finalState, U) + b # batch_size, n_classes
        return preds
    
    def add_embedding(self, option = 'Constant'):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors"

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        # option = config.trainable_embeddings
        if option == 'Variable':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.inputs_placeholder)
        elif option == 'Constant':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.inputs_placeholder)
        embeddings = tf.reshape(embeddings_temp, shape = (-1, self.config.max_length, self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def train_on_batch(self, sess, inputs_batch, labels_batch, seqlen_batch):
        """
        MODIF
        Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features) # CHECK: np.ndarray??
            labels_batch: np.ndarray of shape (n_samples, n_classes)
            labels_batch: np.array of shape (n_samples)
        Returns:
            loss: loss over the batch (a scalar)
        """
        labels_batch = np.reshape(labels_batch, (-1, 1))
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, seqlen_batch = seqlen_batch, dropout = self.config.dropout) # MODIF
        print(inputs_batch.shape)
        print(len(labels_batch))
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, seqlen_batch):
        """Make predictions for the provided batch of data 

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch, seqlen_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
            # grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])
        return losses

    def fit(self, sess, train, dev_data_np, dev_seqlen, dev_labels): # MODIF # CAREFUL DEV/dev
        '''
            Returns LISTS:
                - losses_epochs
                - dev_performances_epochs
                - dev_predictions_epochs
                - dev_predicted_classes_epochs
        '''
        losses_epochs = [] #M
        dev_performances_epochs = [] # MODIF
        dev_predictions_epochs = [] #M
        dev_predicted_classes_epochs = [] #M
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss = self.run_epoch(sess, train)

            # Computing predictions # MODIF
            dev_predictions = self.predict_on_batch(sess, dev_data_np, dev_seqlen) #OUCH

            # Computing development performance #MODIF
            dev_predictions = softmax(np.array(dev_predictions))
            dev_predicted_classes = np.argmax(dev_predictions, axis = 1)
            dev_performance = get_performance(dev_predicted_classes, dev_labels, n_classes = 4)

            # Adding to global outputs #MODIF
            dev_predictions_epochs.append(dev_predictions)
            dev_predicted_classes_epochs.append(dev_predicted_classes)
            dev_performances_epochs.append(dev_performance) 
            losses_epochs.append(loss)

        return losses_epochs, dev_performances_epochs, dev_predicted_classes_epochs, dev_predictions_epochs

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.seqlen_placeholder = None
        self.dropout_placeholder = None
        self.build()