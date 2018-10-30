import tensorflow as tf
import numpy as np

def build_cnn_model(input):
	conv1 = ConvStack(inputs=input, nChannels=32, id=1)
	conv2 = ConvStack(inputs=conv1, nChannels=16, id=2)
	conv3 = ConvStack(inputs=conv2, nChannels=16, id=4)
	conv_flat = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
	dense1 = tf.layers.dense(inputs=conv_flat, units=32, name="Dense1")
	bIsTrain = tf.get_default_graph().get_tensor_by_name("bIsTrain:0")
	dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=bIsTrain, name="Dropout")
	dense2 = tf.layers.dense(inputs=dropout1, units=10, name="Dense2")
	return tf.nn.softmax(logits=dense2, name="Output")
	
def build_dnn_model(input):
	dense1 = DenseStack(inputs=input, nUnits=256, id=1)
	dense2 = DenseStack(inputs=dense1, nUnits=64, id=2)
	dense3 = tf.layers.dense(inputs=dense2, units=10, name="Dense1")
	return tf.nn.softmax(logits=dense3, name="Output")

def ConvStack(inputs, nChannels, id, conv_filter_dim=[3,3], padding_type="VALID"):
	with tf.variable_scope("ConvStack"+str(id)):
		conv = tf.layers.conv2d(inputs=inputs, filters=nChannels, kernel_size=conv_filter_dim, padding=padding_type, name="Conv")
		bn = tf.layers.batch_normalization(inputs=conv, name="BN")
		relu = tf.nn.relu(features=bn, name="Relu")
		bIsTrain = tf.get_default_graph().get_tensor_by_name("bIsTrain:0")
		output = tf.layers.dropout(inputs=relu, rate=0.2, training=bIsTrain, name="Dropout")
	return output
	
def DenseStack(inputs, nUnits, id):
	with tf.variable_scope("DenseStack"+str(id)):
		dense = tf.layers.dense(inputs=inputs, units=nUnits, name="Dense")
		bn = tf.layers.batch_normalization(inputs=dense, name="BN")
		relu = tf.nn.relu(features=bn, name="Relu")
		bIsTrain = tf.get_default_graph().get_tensor_by_name("bIsTrain:0")
		output = tf.layers.dropout(inputs=relu, rate=0.5, training=bIsTrain, name="Dropout")
	return output