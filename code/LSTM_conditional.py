######
# basic BOW model with architecture extendable to more complex LSTM models which use both headings and bodies separately.
######
import tensorflow as tf
import numpy as np
import random

from our_model_config import OurModel
from our_util import Progbar, minibatches, get_performance, softmax

class LSTMCondModel(OurModel):

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.headings_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.h_max_len), name = "headings")
        self.bodies_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name = "bodies")
        self.headings_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name = "headings_lengths")
        self.bodies_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name = "bodies_lengths")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name = "labels")
        self.dropout_placeholder = tf.placeholder(tf.float64, name = 'dropout')

    def create_feed_dict(self, headings_batch, bodies_batch, headings_lengths_batch, bodies_lengths_batch, labels_batch=None, dropout = 1.0):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.headings_placeholder: headings_batch,
            self.bodies_placeholder: bodies_batch,
            self.headings_lengths_placeholder: headings_lengths_batch,
            self.bodies_lengths_placeholder: bodies_lengths_batch
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_embedding(self, option = 'Constant'):
        """Adds an embedding layer that maps from input tokens (integers) to vectors for both the headings and bodies:

        Returns:
            embeddings_headings: tf.Tensor of shape (None, h_max_len, embed_size)
            embeddings_bodies: tf.Tensor of shape (None, b_max_len, embed_size)
        """
        if option == 'Constant':
            embeddings_headings_temp = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.headings_placeholder)
            embeddings_bodies_temp   = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.bodies_placeholder)
        elif option == 'Variable':
            embeddings_headings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.headings_placeholder)
            embeddings_bodies_temp   = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.bodies_placeholder)
        embeddings_headings = tf.reshape(embeddings_headings_temp, shape = (-1, self.config.h_max_len, self.config.embed_size))
        embeddings_bodies = tf.reshape(embeddings_bodies_temp, shape = (-1, self.config.b_max_len, self.config.embed_size))
        return embeddings_headings, embeddings_bodies

    def add_prediction_op(self):

        with tf.variable_scope('head'):

            # LSTM that handles the headers
            cell_h = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_h = tf.nn.rnn_cell.DropoutWrapper(cell_h, output_keep_prob = self.dropout_placeholder)
            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)

            # x = self.inputs_placeholder
            x_header, x_body = self.add_embedding(option = self.config.trainable_embeddings)
            # print('Predict op: x', x)
            rnnOutput_h = tf.nn.dynamic_rnn(cell_h, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF
            Y = tf.slice(rnnOutput_h[0], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])

        with tf.variable_scope('body'):
            # LSTM that handles the bodies
            cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_b = tf.nn.rnn_cell.DropoutWrapper(cell_b, output_keep_prob = self.dropout_placeholder)

            U_b = tf.get_variable(name = 'U_b', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b_b = tf.get_variable(name = 'b_b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)

            rnnOutput_b = tf.nn.dynamic_rnn(cell_b, inputs = x_body, dtype = tf.float64, initial_state = rnnOutput_h[1], sequence_length = self.bodies_lengths_placeholder)
            h_N = rnnOutput_b[1][1] # batch_size, cell.state_size

        ## ATTENTION!
        W_y = tf.get_variable(name = 'Wy', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_h = tf.get_variable(name = 'Wh', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        w = tf.get_variable(name = 'w', shape = (self.config.hidden_size, 1), initializer = theInitializer, dtype = tf.float64)
        W_p = tf.get_variable(name = 'Wo', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_x = tf.get_variable(name = 'Wx', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)

        M_1 = tf.reshape(tf.matmul(tf.reshape(Y, shape = (-1, self.config.hidden_size)), W_y), shape = (-1, self.config.attention_length, self.config.hidden_size))
        M_2 = tf.expand_dims(tf.matmul(h_N, W_h), axis = 1)
        M = tf.tanh(M_1 + M_2)
        alpha = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(M, shape = (-1, self.config.hidden_size)), w)), shape = (-1, self.config.attention_length))

        r = tf.squeeze(tf.batch_matmul(tf.transpose(tf.expand_dims(alpha, 2), perm = [0, 2, 1]), Y))
        h_star = tf.tanh(tf.matmul(r, W_p) + tf.matmul(h_N, W_x))

        # Compute predictions
        preds = tf.matmul(h_star, U_b) + b_b # batch_size, n_classes
        return preds

    def add_prediction_op(self):

        with tf.variable_scope('head'):

            # LSTM that handles the headers
            cell_h = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_h = tf.nn.rnn_cell.DropoutWrapper(cell_h, output_keep_prob = self.dropout_placeholder)
            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)

            x_header, x_body = self.add_embedding(option = self.config.trainable_embeddings)

            if self.config.n_layers <= 1:
                rnnOutput_h = tf.nn.dynamic_rnn(cell_h, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF
            elif self.config.n_layers > 1:
                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell_h] * self.config.n_layers)
                rnnOutput_h = tf.nn.dynamic_rnn(stacked_lstm, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF
            Y = tf.slice(rnnOutput_h[0], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])

        with tf.variable_scope('body'):
            # LSTM that handles the bodies
            cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_b = tf.nn.rnn_cell.DropoutWrapper(cell_b, output_keep_prob = self.dropout_placeholder)

            U_b = tf.get_variable(name = 'U_b', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b_b = tf.get_variable(name = 'b_b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)

            if self.config.n_layers <= 1:
                rnnOutput_b = tf.nn.dynamic_rnn(cell_b, inputs = x_body, dtype = tf.float64, initial_state = rnnOutput_h[1], sequence_length = self.bodies_lengths_placeholder)
                h_N = rnnOutput_b[1][1] # batch_size, cell.state_size
            elif self.config.n_layers > 1:
                print('header rnn, ', len(rnnOutput_h[1]))
                stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell_b] * self.config.n_layers)
                rnnOutput_b = tf.nn.dynamic_rnn(stacked_lstm, inputs = x_body, dtype = tf.float64, initial_state = rnnOutput_h[1], sequence_length = self.bodies_lengths_placeholder)
                h_N = rnnOutput_b[1][self.config.n_layers - 1][1]

        ## ATTENTION!
        W_y = tf.get_variable(name = 'Wy', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_h = tf.get_variable(name = 'Wh', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        w = tf.get_variable(name = 'w', shape = (self.config.hidden_size, 1), initializer = theInitializer, dtype = tf.float64)
        W_p = tf.get_variable(name = 'Wo', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_x = tf.get_variable(name = 'Wx', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)

        M_1 = tf.reshape(tf.matmul(tf.reshape(Y, shape = (-1, self.config.hidden_size)), W_y), shape = (-1, self.config.attention_length, self.config.hidden_size))
        M_2 = tf.expand_dims(tf.matmul(h_N, W_h), axis = 1)
        M = tf.tanh(M_1 + M_2)
        alpha = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(M, shape = (-1, self.config.hidden_size)), w)), shape = (-1, self.config.attention_length))

        r = tf.squeeze(tf.batch_matmul(tf.transpose(tf.expand_dims(alpha, 2), perm = [0, 2, 1]), Y))
        h_star = tf.tanh(tf.matmul(r, W_p) + tf.matmul(h_N, W_x))

        # Compute predictions
        preds = tf.matmul(h_star, U_b) + b_b # batch_size, n_classes
        return preds


    def train_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            headings_batch: np.ndarray of shape (n_samples, n_features)
            bodies_batch: np.ndarray of shape (n_samples, n_features)
            headings_lengths_batch: np.ndarray of shape (n_samples, 1)
            bodies_lengths_batch: np.ndarray of shape (n_samples, 1)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch, y_batch, dropout = self.config.dropout)
        # print('feed', feed)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        ## for debugging / testing
        if (np.isnan(loss)):
            print('headings', h_batch)
            print('bodies', b_batch)
            print('nh_len', h_len_batch)
            print('b_len', b_len_batch)
            print('labels', y_batch)
            assert(False)
        return loss

    def predict_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            headings_batch: np.ndarray of shape (n_samples, n_features)
            bodies_batch: np.ndarray of shape (n_samples, n_features)
            headings_lengths_batch: np.ndarray of shape (n_samples, 1)
            bodies_lengths_batch: np.ndarray of shape (n_samples, 1)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, h_np, b_np, h_len, b_len, y):
        # prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        # shuffle
        ind = range(self.config.num_samples)
        random.shuffle(ind)
        # sizes
        batch_start = 0
        batch_end = 0       
        N = self.config.batch_size
        num_batches = self.config.num_samples / N
        # run batches
        for i in range(num_batches):
            batch_start = (i*N)
            batch_end = (i+1)*N
            indices = ind[batch_start:batch_end]
            h_batch = h_np[indices,:]
            b_batch = b_np[indices,:]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]
            y_batch = y[indices]
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
            losses.append(loss)
            if (i % (1 + num_batches/10)) == 0:
                print('batch: ', i, ', loss: ', loss)
        # run last smaller batch
        if (batch_end < self.config.num_samples):
            indices = ind[batch_end:]
            h_batch = h_np[indices,:]
            b_batch = b_np[indices,:]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]
            y_batch = y[indices]
            # loss
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
            losses.append(loss)
        return losses

    def fit(self, sess, h_np, b_np, h_len, b_len, y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y): #M
        #losses = []
        losses_epochs = [] #M
        dev_performances_epochs = [] # M
        dev_predictions_epochs = [] #M
        dev_predicted_classes_epochs = [] #M

        for epoch in range(self.config.n_epochs):
            print('-------new epoch---------')
            loss = self.run_epoch(sess, h_np, b_np, h_len, b_len, y)

            # Computing predictions #MODIF
            dev_predictions = self.predict_on_batch(sess, dev_h, dev_b, dev_h_len, dev_b_len)

            # Computing development performance #MODIF
            dev_predictions = softmax(np.array(dev_predictions))
            dev_predicted_classes = np.argmax(dev_predictions, axis = 1)
            dev_performance = get_performance(dev_predicted_classes, dev_y, n_classes = 4)

            # Adding to global outputs #MODIF
            dev_predictions_epochs.append(dev_predictions)
            dev_predicted_classes_epochs.append(dev_predicted_classes)
            dev_performances_epochs.append(dev_performance) 
            losses_epochs.append(loss)
            
            print('EPOCH: ', epoch, ', LOSS: ', np.mean(loss))

        return losses_epochs, dev_performances_epochs, dev_predicted_classes_epochs, dev_predictions_epochs

    def __init__(self, config):
        self.config = config
        self.headings_placeholder = None
        self.bodies_placeholder = None
        self.headings_lengths_placeholder = None
        self.bodies_lengths_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.build()