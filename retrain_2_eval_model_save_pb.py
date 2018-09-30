#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
01.10.2018
In this version we add to hub-model only single layer (as an object of class SingleLayerNeuralNetwork
and we save all variables
but restore only for this layer.
"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import os
import sys
import argparse
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

import tensorflow as tf
#import tensorflow_hub as hub

#from rotate_images import *
from layers import *
import networks

HIDDEN_NUM = 8
CHECKPOINT_NAME = 'my_test_model'
output_node_names = ['sigmoid_out']

#NUM_CLASSES = 412

from model import *



def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--restore', dest='restore', action='store_true')
	parser.add_argument('-i', '--in_file', default="dump.gz", type=str,\
		help='input dir')
	parser.add_argument('-k', '--k', default=1, type=int,\
		help='number of network')
	parser.add_argument('-hn', '--hidden_num', default=8, type=int,\
		help='number of neurons in hiden layer')

	return parser




if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	data_file = arguments.in_file

	f = gzip.open(data_file, 'rb')
	data = pickle.load(f)
	map_label_id = data['label_id']
	NUM_CLASSES = len(map_label_id)
	print('NUM_CLASSES =', NUM_CLASSES)

	#-------------------

	# Create a new graph
	graph = tf.Graph() # no necessiry

	with graph.as_default():

		# 1. Construct a graph representing the model.

		shape = SHAPE
		height, width, color =  shape
		#x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
		x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')		
		resized_input_tensor = tf.reshape(x, [-1, height, width, 3])

		if use_hub:
			module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1", 
				trainable=False)	
	
		bottleneck_tensor = module(resized_input_tensor)

		#bottleneck_input = tf.placeholder_with_default(
		#	bottleneck_tensor_stop, shape=[None, bottleneck_tensor_size], name='BottleneckInputPlaceholder') # Placeholder for input.
		#output = last_layers(bottleneck_input, bottleneck_tensor_size, NUM_CLASSES)

		single_layer_nn = networks.SingleLayerNeuralNetwork(
			input_size=bottleneck_tensor_size, 
			num_neurons=NUM_CLASSES,
			func=tf.nn.sigmoid,
			name='_out')
		
		logits = single_layer_nn.module(bottleneck_tensor)

		#tf.contrib.quantize.create_training_graph()
		#tf.contrib.quantize.create_eval_graph()

		y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='Placeholder-y')   # Placeholder for labels.

		# 2. Add nodes that represent the optimization algorithm.
		# for regression:
		#loss = tf.reduce_mean(tf.square(output - y))
		#optimizer= tf.train.AdagradOptimizer(0.005)
		#train_op = optimizer.minimize(loss)
			
		# for classification:
		#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
		#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1

		#acc_top5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=y, k=5))
		"""
		indices_1 = tf.nn.top_k(logits, k=5)
		indices_2 = tf.nn.top_k(y, k=5)
		correct = tf.equal(indices_1, indices_2)
		acc_top5 = tf.reduce_mean(tf.cast(correct, 'float'))
		"""

		#acc_top5 = tf.nn.in_top_k(logits, tf.argmax(y,1), 5)
		#acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)

		#output_angles_valid = []

		# 3. Execute the graph on batches of input data.
		with tf.Session() as sess:  # Connect to the TF runtime.
			init = tf.global_variables_initializer()
			sess.run(init)	# Randomly initialize weights.

			if arguments.restore:				
				single_layer_nn.restore(sess)
				if False:
					tf.train.Saver().restore(sess, './save_model/{0}'.format(CHECKPOINT_NAME))

			#print('is_train=', is_train.eval())

			# Save model
			
			#single_layer_nn.save(sess)

			if False:
				saver = tf.train.Saver()		
				saver.save(sess, './save_model/{0}'.format(CHECKPOINT_NAME))  

			# SAVE GRAPH TO PB
			graph = sess.graph			
			#op = is_train.assign(False)
			#sess.run(op)
			tf.graph_util.remove_training_nodes(graph.as_graph_def())
			# tf.contrib.quantize.create_eval_graph(graph)
			# tf.contrib.quantize.create_training_graph()
			output_graph_def = tf.graph_util.convert_variables_to_constants(
				sess, graph.as_graph_def(), output_node_names)
			dir_for_model = '.'
			tf.train.write_graph(output_graph_def, dir_for_model,
				'saved_model_full_2.pb', as_text=False)	


