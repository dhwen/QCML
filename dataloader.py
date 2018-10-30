from tensorflow.examples.tutorials.mnist import input_data


def loadMNIST():
	return input_data.read_data_sets("MNIST_data/", one_hot=True)