import tensorflow as tf
import numpy as np
import os
import dataloader
import matplotlib.pyplot as plt
from model import Model

ckpt_path = 'model_ckpt/'
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)

#//Configurables
model_type = 'dnn'
num_train_epochs = 20
batch_size = 128

mnist = dataloader.loadMNIST()
net = Model(model_type)

with tf.Session(graph=net.graph) as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	#if os.path.isfile(ckpt_path + 'ConvNet.ckpt.meta'):
	#    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
	#    print("Restored ckpt")

	train_accuracies = []
	test_accuracies = []

	for epoch in range(num_train_epochs):
	
		num_processed_samples = 0
		
		while(num_processed_samples < 60000):
			num_processed_samples += batch_size
			data, labels = mnist.train.next_batch(batch_size)
			sess.run(net.opt, feed_dict={net.input: data, net.label: labels, net.bIsTrain: True})

		data, labels = mnist.train.next_batch(60000)
		
		train_accuracy = 0
		for i in range(60):
			train_accuracy += 1/60*sess.run(net.accuracy, feed_dict={net.input: data[1000*i:1000*(i+1)], net.label: labels[1000*i:1000*(i+1)], net.bIsTrain: False})
		train_accuracies.append(train_accuracy)
		
		test_accuracy = 0
		for i in range(10):
			test_accuracy += 1/10*sess.run(net.accuracy, feed_dict={net.input: mnist.test.images[1000*i:1000*(i+1)], net.label: mnist.test.labels[1000*i:1000*(i+1)], net.bIsTrain: False})
		test_accuracies.append(test_accuracy)
		
		print('epoch %d, train accuracy %g, test accuracy %g' % (epoch + 1, test_accuracy, train_accuracy))
		saver.save(sess, ckpt_path + "model.ckpt")


	plt.plot(np.arange(1, num_train_epochs + 1, 1), train_accuracies, label='train accuracy')
	plt.plot(np.arange(1, num_train_epochs + 1, 1), test_accuracies, label='test accuracy')
	plt.xlabel('epoch')
	plt.xticks(np.arange(1, num_train_epochs + 1, 1))
	plt.ylabel('accuracy')
	plt.title("learning curve for" + model_type)
	plt.legend()
	plt.show()