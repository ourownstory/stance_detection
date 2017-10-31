######
# basic BOW model with architecture extendable to more complex LSTM models which use both headings and bodies separately.
######
import tensorflow as tf
import numpy as np
import random

from our_model_config import OurModel
from our_util import Progbar, minibatches, get_performance, softmax

class BOWModel(OurModel):

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.headings_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.h_max_len), name="headings")
        self.bodies_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name="bodies")
        self.headings_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name="headings_lengths")
        self.bodies_lengths_placeholder = tf.placeholder(tf.float64, shape=(None), name="bodies_lengths")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name="labels")

    def create_feed_dict(self, headings_batch, bodies_batch, headings_lengths_batch, bodies_lengths_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.headings_placeholder: headings_batch,
            self.bodies_placeholder: bodies_batch,
            self.headings_lengths_placeholder: headings_lengths_batch,
            self.bodies_lengths_placeholder: bodies_lengths_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self, option = 'Constant'):
        """Adds an embedding layer that maps from input tokens (integers) to vectors for both the headings and bodies:

        Returns:
            embeddings_headings: tf.Tensor of shape (None, h_max_len, embed_size)
            embeddings_bodies: tf.Tensor of shape (None, b_max_len, embed_size)
        """
        #
        # embeddings_headings_temp = tf.nn.embedding_lookup(params = tf.Constant(self.config.pretrained_embeddings), ids = self.headings_placeholder)
        # embeddings_bodies_temp = tf.nn.embedding_lookup(params = tf.Constant(self.config.pretrained_embeddings), ids = self.bodies_placeholder)
        embeddings_headings_temp = tf.nn.embedding_lookup(params = self.config.pretrained_embeddings, ids = self.headings_placeholder)
        embeddings_bodies_temp   = tf.nn.embedding_lookup(params = self.config.pretrained_embeddings, ids = self.bodies_placeholder)
        embeddings_headings = tf.reshape(embeddings_headings_temp, shape = (-1, self.config.h_max_len, self.config.embed_size))
        embeddings_bodies = tf.reshape(embeddings_bodies_temp, shape = (-1, self.config.b_max_len, self.config.embed_size))
        return embeddings_headings, embeddings_bodies

    def add_bow_input(self):
        headings, bodies = self.add_embedding(option = self.config.trainable_embeddings)
        headings_bag = tf.divide(tf.reduce_sum(headings, axis=1), tf.reshape(self.headings_lengths_placeholder, shape = (-1, 1)))
        bodies_bag   = tf.divide(tf.reduce_sum(bodies, axis=1),   tf.reshape(self.bodies_lengths_placeholder, shape = (-1, 1)))
        x = tf.concat_v2(values=[headings_bag, bodies_bag], axis=1)
        return x

    def add_prediction_op(self):
        """Runs an rnn on the input using TensorFlows's
        @tf.nn.dynamic_rnn function, and returns the final state as a prediction.

        Returns:
            logits: tf.Tensor of shape (batch_size, n_classes)
        """
        hidden_size_2 = np.floor(self.config.hidden_next**2 * self.config.hidden_size)
        hidden_size_3 = np.floor(self.config.hidden_next**3 * self.config.hidden_size)
        hidden_size_4 = np.floor(self.config.hidden_next**4 * self.config.hidden_size)
        hidden_size_5 = np.floor(self.config.hidden_next**5 * self.config.hidden_size)

        x = self.add_bow_input()
        theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)
        if not self.config.n_layers:
            W = tf.get_variable(name = 'W', shape = (2*self.config.embed_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c = tf.get_variable(name = 'c', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(x, W) + c # batch_size, n_classes
        elif self.config.n_layers == 1:
            U0 = tf.get_variable(name = 'U0', shape = (2*self.config.embed_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            c0 = tf.get_variable(name = 'c0', shape = (self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0) # batch_size, hidden_size
            U1 = tf.get_variable(name = 'U1', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c1 = tf.get_variable(name = 'c1', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(h1, U1) + c1 # batch_size, n_classes
        elif self.config.n_layers == 2:
            U0 = tf.get_variable(name = 'U0', shape = (2*self.config.embed_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            c0 = tf.get_variable(name = 'c0', shape = (self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0) # batch_size, hidden_size
            U1 = tf.get_variable(name = 'U1', shape = (self.config.hidden_size, hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            c1 = tf.get_variable(name = 'c1', shape = (hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1) # batch_size, hidden_size_2
            U2 = tf.get_variable(name = 'U2', shape = (hidden_size_2, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c2 = tf.get_variable(name = 'c2', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(h2, U2) + c2 # batch_size, n_classes
        elif self.config.n_layers == 3:
            U0 = tf.get_variable(name = 'U0', shape = (2*self.config.embed_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            c0 = tf.get_variable(name = 'c0', shape = (self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0) # batch_size, hidden_size
            U1 = tf.get_variable(name = 'U1', shape = (self.config.hidden_size, hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            c1 = tf.get_variable(name = 'c1', shape = (hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1) # batch_size, hidden_size_2
            U2 = tf.get_variable(name = 'U2', shape = (hidden_size_2, hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            c2 = tf.get_variable(name = 'c2', shape = (hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2) # batch_size, hidden_size_3
            U3 = tf.get_variable(name = 'U3', shape = (hidden_size_3, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c3 = tf.get_variable(name = 'c3', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(h3, U3) + c3 # batch_size, n_classes
        elif self.config.n_layers == 4:
            U0 = tf.get_variable(name = 'U0', shape = (2*self.config.embed_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            c0 = tf.get_variable(name = 'c0', shape = (self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0) # batch_size, hidden_size
            U1 = tf.get_variable(name = 'U1', shape = (self.config.hidden_size, hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            c1 = tf.get_variable(name = 'c1', shape = (hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1) # batch_size, hidden_size_2
            U2 = tf.get_variable(name = 'U2', shape = (hidden_size_2, hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            c2 = tf.get_variable(name = 'c2', shape = (hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2) # batch_size, hidden_size_3
            U3 = tf.get_variable(name = 'U3', shape = (hidden_size_3, hidden_size_4), initializer = theInitializer, dtype = tf.float64)
            c3 = tf.get_variable(name = 'c3', shape = (hidden_size_4), initializer = theInitializer, dtype = tf.float64)
            h4 = tf.nn.relu(tf.matmul(h3, U3) + c3) # batch_size, hidden_size_4
            U4 = tf.get_variable(name = 'U4', shape = (hidden_size_4, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c4 = tf.get_variable(name = 'c4', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(h4, U4) + c4 # batch_size, n_classes
        elif self.config.n_layers == 5:
            U0 = tf.get_variable(name = 'U0', shape = (2*self.config.embed_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            c0 = tf.get_variable(name = 'c0', shape = (self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
            h1 = tf.nn.relu(tf.matmul(x, U0) + c0) # batch_size, hidden_size
            U1 = tf.get_variable(name = 'U1', shape = (self.config.hidden_size, hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            c1 = tf.get_variable(name = 'c1', shape = (hidden_size_2), initializer = theInitializer, dtype = tf.float64)
            h2 = tf.nn.relu(tf.matmul(h1, U1) + c1) # batch_size, hidden_size_2
            U2 = tf.get_variable(name = 'U2', shape = (hidden_size_2, hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            c2 = tf.get_variable(name = 'c2', shape = (hidden_size_3), initializer = theInitializer, dtype = tf.float64)
            h3 = tf.nn.relu(tf.matmul(h2, U2) + c2) # batch_size, hidden_size_3
            U3 = tf.get_variable(name = 'U3', shape = (hidden_size_3, hidden_size_4), initializer = theInitializer, dtype = tf.float64)
            c3 = tf.get_variable(name = 'c3', shape = (hidden_size_4), initializer = theInitializer, dtype = tf.float64)
            h4 = tf.nn.relu(tf.matmul(h3, U3) + c3) # batch_size, hidden_size_4
            U4 = tf.get_variable(name = 'U4', shape = (hidden_size_4, hidden_size_5), initializer = theInitializer, dtype = tf.float64)
            c4 = tf.get_variable(name = 'c4', shape = (hidden_size_5), initializer = theInitializer, dtype = tf.float64)
            h5 = tf.nn.relu(tf.matmul(h4, U4) + c4) # batch_size, hidden_size_5
            U5 = tf.get_variable(name = 'U5', shape = (hidden_size_5, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            c5 = tf.get_variable(name = 'c5', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            pred = tf.matmul(h5, U5) + c5 # batch_size, n_classes
        return pred

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
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
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
            # loss
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch)
            losses.append(loss)
            # prog.update(i + 1, [("train loss", loss)])
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
            print('batch: ', i, ', loss: ', loss)
            # print('-------last batch---------')
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
        self.build()