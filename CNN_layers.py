import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import struct

# The CNN neural network can be implemented in Tensorflows layers API as follow. The data is first loaded and prepared in the same way as before


def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = ((images / 255.) - .5) * 2

    return images, labels

# the data is split into two sets; the training set contains 50,000 while the test set contains the remaining 10,000


X_data, y_data = load_mnist('', kind='train')

X_test, y_test = load_mnist('', kind='t10k')

# a function is then created that generatres minibatches


def batch_generator(X, y, batch_size=64, shuffle=False, random_seed=None):

    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])

# the data then needs to be normalized for performance purposes, which is accomplished by mean centering the data and then dividing by the standard deviation


X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = (X_valid - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val


# the tensorflow CNN can then be generated

class ConvNN():

    def __init__(self, batchsize=64, epochs=20, learning_rate=1e-4, dropout_rate=0.5, shuffle=True, random_seed=None):

        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph()

        with g.as_default():
            tf.set_random_seed(random_seed)

            self.build()

            self.init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=g)

    def build(self):

        tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='tf_x')
        tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_y')

        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')

        tf_x_image = tf.reshape(
            tf_x, shape=[-1, 28, 28, 1], name='input_x_2dimages')

        tf_y_onehot = tf.one_hot(
            indices=tf_y, depth=10, dtype=tf.float32, name='input_y_onehot')

        # the first convolution layer is then built

        h1 = tf.layers.conv2d(tf_x_image, kernel_size=(
            5, 5), filters=32, activation=tf.nn.relu)

        # the tf.layers module provides several prebuilt classes which can be used to create layers in neural networks. the above creates the convolusion kernel.

        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2))

        h2 = tf.layers.conv2d(h1_pool, kernel_size=(
            5, 5), filters=64, activation=tf.nn.relu)

        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(2, 2), strides=(2, 2))

        # the first fully connected layer is then created

        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])

        h3 = tf.layers.dense(h2_pool_flat, 1024, activation=tf.nn.relu)

        # the dropout method is then applied. There is one important thing to note here. In the low level API, the tf.nn.dropout() functions was used. This has an argument called keep_proba, that indicates the probability of keeping units. Here, however, the tf.layers.dropout() function was used. Conversely, the 'rate' argument here is the drop probability i.e. (1 - keep_proba). Note also that the low level API version takes the keep_proba parameter as a placeholder that could be changed when switching between training sets. The below function does not do this; instead, the proability is locked in. However, it takes a boolean argument which can be used to switch between training and testing sets

        h3_drop = tf.layers.dropout(
            h3, rate=self.dropout_rate, training=is_train)

        h4 = tf.layers.dense(h3_drop, 10, activation=None)

        # the probabilities are then evaluated. Note that the tf.nn.softmax returns the probabilities. The argmax, on the other hand returns the index with the highest values. The cast function simply casts the results into integers

        predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'), 'labels': tf.cast(
            tf.argmax(h4, axis=1), tf.int32, name='labels')}

        # the cost function is then evalauted

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')

        # the optimization routine is then developed. Again, the Adamoptimizer is used

        optim = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optim.minimize(cross_entropy_loss, name='train_op')

        correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

# several conveniece functions are then defined

    def save(self, epoch, path='./tf.layers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        print('Saving Mode in {}'.format(path))

        self.saver(self.sess, os.path.join(path, 'model.cpkt'), global_step_epoch)

    def load(self, epoch, path):

        self.saver.restore(self.sess, os.path.join(path, 'model.cpkt-{}'.format(epoch)))

    def train(self, training_set, validation_set=None, initialize=True):

        if initialize:
            self.sess.run(self.init_op)

        self.training_cost = []

        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1, self.epochs + 1):

            batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)

            avg_loss = 0

            for i, (batch_x, batch_y) in enumerate(batch_gen):

                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'is_train:0': True}

                loss, _ = self.sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)

                avg_loss += loss

            print('Epoch: {} Avg. Loss: {:.2f}'.format(epoch, avg_loss), end=' ')

            if validation_set is not None:

                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'is_train:0': False}

                valid_acc = self.sess.run('accuracy:0', feed_dict=feed)

                print('Validation Accuracy: {:.2f}'.format(valid_acc))
            else:
                print()

    def predict(self, X_test, return_proba=False):

        feed = {'tf_x:0': X_test, 'is_train:0': False}

        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)




cnn = ConvNN(random_seed=123)

cnn.train(training_set=(X_train_centered, y_train), validation_set=(X_valid_centered, y_valid), initialize=True)

cnn.save(epochs=20)

del cnn

cnn2 = ConvNN(random_seed=123)
cnn2.load(epoch=20, path='./tflayers-model')

print(cnn.predict(X_test_centered[:10, :]))

preds = cnn2.predict(X_centered)

print('Test Accuracy: {:.2f}'.format(100*np.sum(y_test == preds) / len(y_test)))
