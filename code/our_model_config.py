import tensorflow as tf

class OurModel(object):
    """Abstracts a Tensorflow graph for use on final project.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.inputs_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.max_length), name="x")
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name="y")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self, option = 'Constant'):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        if option == 'Variable':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.inputs_placeholder)
        elif option == 'Constant':
            embeddings_temp = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.inputs_placeholder)
        embeddings = tf.reshape(embeddings_temp, shape = (-1, self.config.max_length, self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds ops to compute the loss function.

        Args:
            pred: A tensor of shape (batch_size, 1) containing the last
            state of the neural network.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = tf.reshape(self.labels_placeholder, (-1, )) # Check whether this is necessary
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred, labels = y))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """
        # Check if Adam has adaptive learning rate
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            losses.append(loss)
            prog.update(i + 1, [("train loss", loss)])
        return losses

    def fit(self, sess, train):
        losses = []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss = self.run_epoch(sess, train)
            losses.append(loss)
        return losses

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)