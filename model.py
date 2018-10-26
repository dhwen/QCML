import tensorflow as tf
import numpy as np

class ConvNet:
    def __init__(self, im_h, im_w):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="Input")
            self.input = tf.reshape(self.x, [-1, im_h, im_w, 1])
            self.bIsTrain = tf.placeholder(dtype=tf.bool, shape=(), name="bIsTrain")
            self.build_model()
            self.build_backprop()
            self.build_eval()

    def build_model(self):
        conv1 = self.ConvStack(inputs=self.input, nChannels=32, id=1)
        conv2 = self.ConvStack(inputs=conv1, nChannels=32, id=2)
        conv3 = self.ConvStack(inputs=conv2, nChannels=32, id=3)
        conv4 = self.ConvStack(inputs=conv3, nChannels=64, id=4)
        conv_flat = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
        dense1 = tf.layers.dense(inputs=conv_flat, units=64, name="Dense1")
        dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=self.bIsTrain, name="Dropout")
        dense2 = tf.layers.dense(inputs=dropout1, units=10, name="Dense2")
        self.output = tf.nn.softmax(logits=dense2, name="Output")

    def build_backprop(self):
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Label")
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.output)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.opt = self.optimizer.minimize(loss=self.loss)

    def build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def ConvStack(self, inputs, nChannels, id, conv_filter_dim=[3,3], padding_type="VALID"):
        with tf.variable_scope("ConvStack"+str(id)):
            conv = tf.layers.conv2d(inputs=inputs, filters=nChannels, kernel_size=conv_filter_dim, padding=padding_type, name="Conv")
            bn = tf.layers.batch_normalization(inputs=conv, name="BN")
            relu = tf.nn.relu(features=bn, name="Relu")
            output = tf.layers.dropout(inputs=relu, rate=0.1, training=self.bIsTrain, name="Dropout")
        return output