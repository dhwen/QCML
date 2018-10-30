import tensorflow as tf
import numpy as np
import modelfactory

class Model:
	def __init__(self, type):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.input = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="Input")
			self.bIsTrain = tf.placeholder(dtype=tf.bool, shape=(), name="bIsTrain")
			self.build_model(type)
			self.build_backprop()
			self.build_eval()

	def build_model(self, type):
		if type == 'cnn':
			input = tf.reshape(self.input, [-1, 28, 28, 1])
			self.output = modelfactory.build_cnn_model(input)
		elif type == 'dnn':
			self.output = modelfactory.build_dnn_model(self.input)
		else:
			print("model type %s not supported", type)

	def build_backprop(self):
		self.label = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Label")
		self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=self.output)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
		self.opt = self.optimizer.minimize(loss=self.loss)

	def build_eval(self):
		correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))