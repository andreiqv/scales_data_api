#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 


"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
#import tensorflow_hub as hub
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from layers import *

HIDDEN_NUM_DEFAULT = 8

#


# add a final layer (or a few layers)

def network01(input_tensor, input_size, output_size):
	""" Without an activation function.
	"""

	if False:
		droupout_rate = 0.2
		input_tensor = tf.layers.dropout(inputs=input_tensor, rate=droupout_rate)
		print('Dropout with rate {} is used'.format(droupout_rate))

	x = fullyConnectedLayer(
		input_tensor, input_size=input_size, num_neurons=output_size, func=None)

	# x = tf.layers.batch_normalization(x, training=True)  # doesn't work well
		
	return x

# add a final layer (or a few layers)

def network02_CNN(input_tensor, input_size, output_size):
	""" Without an activation function.
	"""

	x = tf.reshape(input_tensor, [-1,48,32,1])
	input_mult = 1
	print('x.shape = ', x.shape)
	x = convPoolLayer(x, kernel=(4,4), pool_size=2, num_in=1, num_out=4*input_mult, 
		func=tf.nn.relu, name='1')  # 24 x 16 x 4 * input_mult
	print('x.shape = ', x.shape)
	input_size = input_mult * input_size
	x = tf.reshape(x, [-1, input_size])
	print('x.shape = ', x.shape)
	print('input_size = ', input_size)

	if False:
		droupout_rate = 0.2
		input_tensor = tf.layers.dropout(inputs=input_tensor, rate=droupout_rate)
		print('Dropout with rate {} is used'.format(droupout_rate))

	x = fullyConnectedLayer(x, input_size=input_size, num_neurons=output_size, 
							func=None)

	# x = tf.layers.batch_normalization(x, training=True)  # doesn't work well
		
	return x	


# Single layer networks with activation functions

def network1(input_tensor, input_size, output_size):

	f1 = fullyConnectedLayer(
		input_tensor, input_size=input_size, num_neurons=output_size, 
		func=tf.nn.sigmoid, name='_out') # func=tf.nn.relu
	
	return f1


def network2(input_tensor, input_size, output_size, hidden_num=HIDDEN_NUM_DEFAULT):

	f1 = fullyConnectedLayer(
		input_tensor, input_size=input_size, num_neurons=hidden_num, 
		func=tf.nn.relu, name='F1') # func=tf.nn.relu
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=hidden_num, num_neurons=output_size, 
		func=tf.nn.sigmoid, name='_out')

	return f2


#---------------------
# Neural network as a class:

class SingleLayerNeuralNetwork:	

	def __init__(self, input_size, num_neurons, func=None, name=''):

		self.W = weight_variable([input_size, num_neurons], name='W_single_layer_nn')
		self.b = bias_variable([num_neurons], name='b_single_layer_nn')

		self.name = name
		self.func = func
		self.checkpoint = "./save_model/single_layer_nn.ckpt"
		
	def module(self, x):

		h = tf.matmul(x, self.W) + self.b

		if self.func:
			h = self.func(h, name='sigmoid_out')

		return h

	def save(self, sess):

		saver = tf.train.Saver()
		saver.save(sess, self.checkpoint)
		return True

	def restore(self, sess):

		saver = tf.train.Saver({'W_single_layer_nn': self.W, 'b_single_layer_nn': self.b})
		saver.restore(sess, self.checkpoint)

#==============================
# other models:

def perceptron(x, shape, output_size):

	input_size = shape[0] * shape[1] * shape[2]
	p_flat = tf.reshape(x, [-1, input_size])

	f1 = fullyConnectedLayer(p_flat, input_size=input_size, num_neurons=output_size, 
		func=tf.nn.sigmoid, name='F1')
	
	return f1



def conv_network_224(x_image):

	# input 224 x 224 x 3
	num_color = 3

	p1 = convPoolLayer(x_image, kernel=(5,5), pool_size=2, num_in=num_color, num_out=6, 
		func=tf.nn.relu, name='1') # 112 x 112
	p2 = convPoolLayer(p1, kernel=(5,5), pool_size=2, num_in=6, num_out=6, 
		func=tf.nn.relu, name='2')  # 56 x 56
	p3 = convPoolLayer(p2, kernel=(4,4), pool_size=2, num_in=6, num_out=6, 
		func=tf.nn.relu, name='3')   # 28 x 28
	p4 = convPoolLayer(p3, kernel=(3,3), pool_size=2, num_in=6, num_out=6, 
		func=tf.nn.relu, name='4')   # 14 x 14
	p5 = convPoolLayer(p4, kernel=(3,3), pool_size=2, num_in=6, num_out=12, 
		func=tf.nn.relu, name='5')   # 7 x 7

	# fully-connected layers
	fullconn_input_size = 7*7*12 #= 588
	p_flat = tf.reshape(p5, [-1, fullconn_input_size])

	return p_flat



def conv_network_1(x_image):	

	num_color = 1
	# conv layers
	p1 = convPoolLayer(x_image, kernel=(5,5), pool_size=3, num_in=num_color, num_out=16, 
		func=tf.nn.relu, name='1') # 180 x 180
	p2 = convPoolLayer(p1, kernel=(5,5), pool_size=3, num_in=16, num_out=16, 
		func=tf.nn.relu, name='2')  # 60 x 60 
	p3 = convPoolLayer(p2, kernel=(4,4), pool_size=3, num_in=16, num_out=32, 
		func=tf.nn.relu, name='3')   # 20 x 20 
	p4 = convPoolLayer(p3, kernel=(3,3), pool_size=2, num_in=32, num_out=32, 
		func=tf.nn.relu, name='4')   # 10 x 10 
	p5 = convPoolLayer(p4, kernel=(3,3), pool_size=2, num_in=32, num_out=64, 
		func=tf.nn.relu, name='5')   # 5 x 5

	# fully-connected layers
	fullconn_input_size = 5*5*64
	p_flat = tf.reshape(p5, [-1, fullconn_input_size])

	f1 = fullyConnectedLayer(p_flat, input_size=fullconn_input_size, num_neurons=1024, 
		func=tf.nn.relu, name='F1')

	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=256, 
		func=tf.nn.relu, name='F2')
	
	drop2 = tf.layers.dropout(inputs=f2, rate=0.4)	
	f3 = fullyConnectedLayer(drop2, input_size=256, num_neurons=1, 
		func=None, name='F3')	 # it doesn't work with sigmoid or relu

	return f3

#--------

#--------

def resnet_50_1(x_image):

	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1")
	bottleneck_tensor_size = 2048
	
	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]

	f1 = fullyConnectedLayer(
		bottleneck_tensor, input_size=bottleneck_tensor_size, num_neurons=1, 
		func=tf.sigmoid, name='F1')
	
	return f1


def resnet_50_2(x_image):

	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1")
	bottleneck_tensor_size = 2048
	
	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]

	f1 = fullyConnectedLayer(
		bottleneck_tensor, input_size=bottleneck_tensor_size, num_neurons=1024, 
		func=tf.nn.relu, name='F1')
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=1, 
		func=tf.sigmoid, name='F2')

	return f2



#-------------------------------------


def inception_resnet_1(x_image):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]
	bottleneck_tensor_size = 1536

	f1 = fullyConnectedLayer(
		bottleneck_tensor, input_size=bottleneck_tensor_size, num_neurons=1024, 
		func=tf.sigmoid, name='F1')
	
	return f1

	

def inception_resnet_2(x_image):
	# iter 2160: train_loss=0.1142, valid_loss=0.1190 (min=0.0926 (109.52 grad.))

	"""
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1")		
	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")	
	assert height, width == hub.get_expected_image_size(module)
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	"""

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
	#height, width, color =  299, 299, 3
	bottleneck_tensor_size = 1536

	bottleneck_tensor = module(x_image)  # Features with shape [batch_size, num_features]
	
	print('bottleneck_tensor:', bottleneck_tensor)


	"""
	bottleneck_input = tf.placeholder_with_default(  # A placeholder op that passes through input when its output is not fed.
		bottleneck_tensor,
		shape=[None, bottleneck_tensor_size],
		name='BottleneckInputPlaceholder')
	print('bottleneck_input:', bottleneck_input)
	"""

	FCL_input = bottleneck_tensor

	f1 = fullyConnectedLayer(
		FCL_input, input_size=bottleneck_tensor_size, num_neurons=1024, 
		func=tf.nn.relu, name='F1')
	
	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=1, 
		func=tf.sigmoid, name='F2')

	return f2



