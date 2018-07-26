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
        self.conv1 = self.ConvStack(inputs=self.input, nChannels=32, conv_filter_dim=[3,3], pool_filter_dim=[5,5], bIsTrain=self.bIsTrain, id=1)
        self.conv2 = self.ConvStack(inputs=self.conv1, nChannels=32, conv_filter_dim=[3,3], pool_filter_dim=[5,5], bIsTrain=self.bIsTrain, id=2)
        self.conv3 = self.ConvStack(inputs=self.conv1, nChannels=32, conv_filter_dim=[3,3], pool_filter_dim=[5,5],bIsTrain=self.bIsTrain, id=3)
        self.conv3_flat = tf.reshape(self.conv3, [-1, np.prod(self.conv3.get_shape().as_list()[1:])])
        self.dense = tf.layers.dense(inputs=self.conv3_flat, units=10, name="Dense1")
        self.output = tf.nn.softmax(logits=self.dense, name="Softmax")

    def build_backprop(self):
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Label")
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.dense)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.opt = self.optimizer.minimize(loss=self.loss)

    def build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def ConvStack(self, inputs, nChannels, conv_filter_dim, pool_filter_dim, bIsTrain, id):
        with tf.variable_scope("ConvStack"+str(id)):
            conv = tf.layers.conv2d(inputs=inputs, filters=nChannels, kernel_size=conv_filter_dim, name="Conv")
            pooling = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_filter_dim, strides=[1, 1], name="Pooling")
            relu = tf.nn.relu(features=pooling, name="Relu")
            bn = tf.layers.batch_normalization(inputs=relu, name="BN")
            output = tf.layers.dropout(inputs=bn, rate=0.8, training=bIsTrain, name="Dropout")
        return output