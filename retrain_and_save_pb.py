#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
25.09.2018
Обучение 1-2 полносвязных слоя на данных из bottleneck.

Add 1 or 2 layers to hub.model and retrain it.
Save whole graph to pb-file.
But problem is that the graph contains placeholderWithDefault 
which doesn't is not supported by tensorrt.

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
PB_FILE_NAME = 'saved_model.pb'
OUTPUT_NODE = 'softmax'

#NUM_CLASSES = 412

from model import *
from timer import timer

"""
if os.path.exists('.notebook'):
	bottleneck_tensor_size =  588
	BATCH_SIZE = 2
	DISPLAY_INTERVAL, NUM_ITERS = 1, 500
else:
	bottleneck_tensor_size =  1536
	#bottleneck_tensor_size =  1001
	BATCH_SIZE = 10
	DISPLAY_INTERVAL, NUM_ITERS = 100, 20*1000*1000
"""
#------------------------




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

	parser.add_argument('-ni', '--num_iters', default=None, type=int,\
		help='Set number of iterations')

	parser.add_argument('-dud', '--dont_use_dump', dest='dont_use_dump', 
		action='store_true', help='Create bottleneck data from origin dataset directory')

	return parser




if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	data_file = arguments.in_file

	if arguments.k == 1:	
		last_layers = networks.network01
		#output_node_names = ['output']
		#last_layers = networks.network1
		#output_node_names = ['sigmoid_out']
	elif arguments.k == 2:	
		last_layers = networks.network2
		#output_node_names = ['sigmoid_out']
	else:
		raise Exception('Bad argument arguments.k')

	if arguments.hidden_num > 0:
		HIDDEN_NUM = arguments.hidden_num

	if arguments.num_iters:	
		num_iters = arguments.num_iters
	else:
		num_iters = NUM_ITERS	# from settings file

	#data_1 = load_data(in_dir, img_size=(540,540))
	#data = split_data(data1, ratio=(6,1,3))

	print('selected network =', arguments.k)
	if arguments.k > 1:
		print('HIDDEN_NUM =', HIDDEN_NUM)


	# Use ready dump file with bottlenecks or bottleneck data from origin dataset directory
	LOAD_DUMP_FILE = not arguments.dont_use_dump
	if LOAD_DUMP_FILE:
		print('data_file =', data_file)
		f = gzip.open(data_file, 'rb')
		data = pickle.load(f)
	else:
		from make_bottleneck_subdir_data_api import make_bottleneck_data
		from model import SHAPE, DATASET_DIR
		data = make_bottleneck_data(src_dir=DATASET_DIR, shape=SHAPE, ratio=[9,1,0])

	train = data['train']
	valid = data['valid']
	test  = data['test']
	#train_data = train['embedding']
	#valid_data = valid['embedding']
	#test_data = test['embedding']
	train_data = train['images']
	valid_data = valid['images']
	test_data = test['images']
	train_labels = train['labels']
	valid_labels = valid['labels']
	test_labels = test['labels']
	train['size'] = len(train['labels'])
	valid['size'] = len(valid['labels'])
	test['size'] = len(test['labels'])

	print('train size:', len(train['labels']))
	print('valid size:', len(valid['labels']))
	print('test size:', len(test['labels']))
	print('Data was loaded.')
	#print('Example of data:', train_data[0])
	print('size of vector:', len(train_data[0]))
	#print('Example of label:',train_labels[0])
	print('size of label:', len(train_labels[0]))
	#sys.exit()

	#train_data = [np.transpose(t) for t in train_data]
	#valid_data = [np.transpose(t) for t in valid_data]
	#test_images = [np.transpose(t) for t in test_images]
	num_train_batches = train['size'] // BATCH_SIZE
	num_valid_batches = valid['size'] // BATCH_SIZE
	num_test_batches = test['size'] // BATCH_SIZE
	print('num_train_batches:', num_train_batches)
	print('num_valid_batches:', num_valid_batches)
	print('num_test_batches:', num_test_batches)

	SAMPLE_SIZE = train['size']
	min_valid_acc = 0

	map_label_id = data['label_id']
	NUM_CLASSES = len(map_label_id)
	print('NUM_CLASSES =', NUM_CLASSES)

	#-------------------

	# Create a new graph
	graph = tf.Graph() # no necessiry

	with graph.as_default():

		# 1. Construct a graph representing the model.

		#is_train = tf.Variable(True)

		shape = SHAPE
		height, width, color =  shape
		#x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
		x = tf.placeholder(tf.float32, [None, height, width, 3], name='input')
		resized_input_tensor = tf.reshape(x, [-1, height, width, 3])

		if use_hub:
			print('*************')
			module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1", 
				trainable=False)
			print(module._graph)
	
		bottleneck_tensor = module(resized_input_tensor)

		"""
		def f_true():
			return tf.stop_gradient(bottleneck_tensor)
		def f_false():
			return bottleneck_tensor
		bottleneck_tensor_stop = tf.cond(is_train, f_true, f_false)
		"""

		"""
		bottleneck_tensor_stop = tf.cond(is_train,
                     lambda: tf.stop_gradient(bottleneck_tensor),
                     lambda: bottleneck_tensor)
		"""
		bottleneck_tensor_stop = tf.stop_gradient(bottleneck_tensor)

		#x = tf.placeholder(tf.float32, [None, 1, bottleneck_tensor_size], name='Placeholder-x') # Placeholder for input.
		bottleneck_input = tf.placeholder_with_default(
			bottleneck_tensor_stop, shape=[None, bottleneck_tensor_size], name='BottleneckInputPlaceholder') # Placeholder for input.
		#bottleneck_input = tf.reshape(bottleneck_input, [-1, bottleneck_tensor_size])
		logits = last_layers(bottleneck_input, bottleneck_tensor_size, NUM_CLASSES)
		print('logits =', logits)

		output = tf.nn.softmax(logits, name=OUTPUT_NODE)

		#output = tf.identity(output, name='logits')
		#logits = output		

		#tf.contrib.quantize.create_training_graph()

		y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='Placeholder-y')   # Placeholder for labels.

		# 2. Add nodes that represent the optimization algorithm.
		# for regression:
		#loss = tf.reduce_mean(tf.square(output - y))
		#optimizer= tf.train.AdagradOptimizer(0.005)
		#train_op = optimizer.minimize(loss)
			
		# for classification:
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
		correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1

		#acc_top5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=y, k=5))
		"""
		indices_1 = tf.nn.top_k(logits, k=5)
		indices_2 = tf.nn.top_k(y, k=5)
		correct = tf.equal(indices_1, indices_2)
		acc_top5 = tf.reduce_mean(tf.cast(correct, 'float'))
		"""

		acc_top5 = tf.nn.in_top_k(logits, tf.argmax(y,1), 5)
		acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)

		#output_angles_valid = []

		# 3. Execute the graph on batches of input data.
		with tf.Session() as sess:  # Connect to the TF runtime.
			init = tf.global_variables_initializer()
			sess.run(init)	# Randomly initialize weights.

			if arguments.restore:
				tf.train.Saver().restore(sess, './saved_model/{0}'.format(CHECKPOINT_NAME))

			#print('is_train=', is_train.eval())

			timer('TRAINING')

			for iteration in range(num_iters):			  # Train iteratively for NUM_iterationS.		 

				if iteration % DISPLAY_INTERVAL == 0:

					# set limit on the number of batches for check of train process
					is_last_iterations = (iteration >= num_iters - DISPLAY_INTERVAL)
					if is_last_iterations:
						print('This is the last iterations.')
					limit = 10000 if is_last_iterations else 100
					num_train_batches_limit = min(num_train_batches, limit)
					num_valid_batches_limit = min(num_valid_batches, limit)

					train_acc = np.mean( [accuracy.eval( \
						feed_dict={bottleneck_input:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_train_batches_limit)])
					valid_acc = np.mean([ accuracy.eval( \
						feed_dict={bottleneck_input:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_valid_batches_limit)])

					# valid top5,6
					valid_acc_top5 = np.mean([ acc_top5.eval( \
						feed_dict={bottleneck_input:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_valid_batches_limit)])					
					valid_acc_top6 = np.mean([ acc_top6.eval( \
						feed_dict={bottleneck_input:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
						y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
						for i in range(0,num_valid_batches_limit)])		
					if valid_acc > min_valid_acc:
						min_valid_acc = valid_acc

					try:
						epoch = iteration//(num_train_batches // BATCH_SIZE * BATCH_SIZE)
					except:
						epoch = 0

					print('epoch {0:2} (i={1:06}): train={2:0.4f}, valid={3:0.4f} (max={4:0.4f}) [top5={5:0.4f}, top6={6:0.4f}]'.\
						format(epoch, iteration, train_acc, valid_acc, min_valid_acc, valid_acc_top5, valid_acc_top6))

					if epoch % 200 == 0:	
						saver = tf.train.Saver()		
						saver.save(sess, './saved_model/{0}'.format(CHECKPOINT_NAME))
				
				a1 = iteration*BATCH_SIZE % train['size']
				a2 = (iteration + 1)*BATCH_SIZE % train['size']
				x_data = train['images'][a1:a2]
				y_data = train['labels'][a1:a2]
				if len(x_data) <= 0: continue
				#sess.run(train_op, {x: x_data, y: y_data})  # Perform one training iteration.		
				sess.run(train_op, {bottleneck_input: x_data, y: y_data})  # Perform one training iteration.
				

			# Save the comp. graph
			if False:
				print('Save the comp. graph')
				x_data, y_data =  valid['images'], valid['labels'] 
				#mnist.train.next_batch(BATCH_SIZE)		
				writer = tf.summary.FileWriter("output", sess.graph)
				#print(sess.run(train_op, {x: x_data, y: y_data}))
				writer.close()  

			# Test of model
			"""
			HERE SOME ERROR ON GPU OCCURS
			num_test_batches = test['size'] // BATCH_SIZE
			test_loss = np.mean([ loss.eval( \
				feed_dict={x:test['images'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]}) \
				for i in range(num_test_batches) ])
			print('Test of model')
			print('test_loss={0:0.4f}'.format(test_loss))
			"""

			"""
			print('Test model')
			test_loss = loss.eval(feed_dict={x:test['images'][0:BATCH_SIZE]})
			print('test_loss={0:0.4f}'.format(test_loss))				
			"""

			# Saver			
			saver = tf.train.Saver()		
			saver.save(sess, './saved_model/{0}'.format(CHECKPOINT_NAME))  
	

			# SAVE GRAPH TO PB
			graph = sess.graph			
			#op = is_train.assign(False)
			#sess.run(op)
			tf.graph_util.remove_training_nodes(graph.as_graph_def())
			# tf.contrib.quantize.create_eval_graph(graph)
			# tf.contrib.quantize.create_training_graph()
			output_node_names = [OUTPUT_NODE]
			output_graph_def = tf.graph_util.convert_variables_to_constants(
				sess, graph.as_graph_def(), output_node_names)
			dir_for_model = '.'
			tf.train.write_graph(output_graph_def, dir_for_model,
				PB_FILE_NAME, as_text=False)	

			timer()

			# it doesn't work. I don't know why.
			#graph_file_name = 'saved_model_gf.pb'			
			#with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
			#	f.write(output_graph_def.SerilizeToString())

			#tf.train.write_graph(graph, dir_for_model,
			#	'saved_model.pb', as_text=False)
