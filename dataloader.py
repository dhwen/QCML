from tensorflow.examples.tutorials.mnist import input_data

class DataLoader:

    def __init__(self):
        pass

    def loadMNIST(self):
        return input_data.read_data_sets("MNIST_data/", one_hot=True)