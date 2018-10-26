import tensorflow as tf
import numpy as np
import os
from model import ConvNet
from dataloader import DataLoader

ckpt_path = 'model_ckpt/'
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)

loader = DataLoader()
mnist = loader.loadMNIST()

net = ConvNet(28, 28)


with tf.Session(graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #if os.path.isfile(ckpt_path + 'ConvNet.ckpt.meta'):
    #    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    #    print("Restored ckpt")

    num_train_epochs = np.power(10, 3)
    batch_size = 128

    for i in range(num_train_epochs):

        batch = mnist.train.next_batch(batch_size)

        sess.run(net.opt, feed_dict={net.x: batch[0], net.label: batch[1], net.bIsTrain: True})

        if (i + 1) % 10 == 0:
            loss = sess.run(net.loss, feed_dict={net.x: batch[0], net.label: batch[1], net.bIsTrain: False})
            print('Epoch %d, training loss is %g' % (i + 1, loss))

        if (i + 1) % 100 == 0:
            train_accuracy = sess.run(net.accuracy, feed_dict={net.x: batch[0], net.label: batch[1], net.bIsTrain: False})
            test_accuracy = sess.run(net.accuracy, feed_dict={net.x: mnist.test.images, net.label: mnist.test.labels, net.bIsTrain: False})
            print('step %d, training accuracy %g' % (i + 1, train_accuracy))
            print('step %d, test accuracy %g' % (i + 1, test_accuracy))

        if (i + 1) % 500 == 0:
            saver.save(sess, ckpt_path + "ConvNet.ckpt")