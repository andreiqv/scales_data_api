#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Сквозное обучение всей нейронной сети на основе хабовской модели.
Т.е. изображения подаются на вход сети (без вычисления боттлнека).
https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
"""

import os
import os.path
import sys
import argparse
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip
import random
from random import randint
import math
import numpy as np

import tensorflow as tf
#from tensorflow.contrib.data import Dataset, Iterator
Dataset = tf.data.Dataset
Iterator = tf.data.Iterator

import networks
import split_data
import distort
import model
from model import *
#last_layers = networks.network1  # with sigmoid
last_layers = networks.network01  # no sigmoid

#DO_MIX = False
DO_MIX = False
NUM_CLASSES = 0
CHECKPOINT_NAME = 'full_model'

np.set_printoptions(precision=4, suppress=True)

#---------------------------------

def save_labels_to_file(map_label_id):

	with open('labels.txt', 'wt') as f:
		for label in range(len(map_label_id)):
			class_id = map_label_id[label]
			f.write('{0}\n'.format(class_id))

#------------------------------------------------

def make_filenames_list_from_subdir(src_dir, shape, ratio):
	"""
	Use names of subdirs as a id.
	And then calculate class_index from id.
	"""

	class_id_set = set()
	#bottleneck_data = dict()
	feature_vectors, labels, filenames = [], [], []

	image_size = (shape[0], shape[1])
	listdir = os.listdir(src_dir)
	
	# 1) findout number of classes
	for class_id in listdir:
		
		subdir = src_dir + '/' + class_id
		if not os.path.isdir(subdir): continue

		if len(os.listdir(subdir)) == 0: 
			continue
		else: 
			try:
				class_id_int = int(class_id)
				class_id_set.add(class_id_int)
			except:
				continue	
			
	# 2) maps class_id to class_index			
	id_list = list(class_id_set)
	id_list.sort()
	print('Number of classes in the sample: {0}'.format(len(id_list)))
	print('Min class id: {0}'.format(min(id_list)))
	print('Max class id: {0}'.format(max(id_list)))
	map_id_label = {class_id : index for index, class_id in enumerate(id_list)}
	map_label_id = {index : class_id for index, class_id in enumerate(id_list)}
	maps = {'id_label' : map_id_label, 'label_id' : map_label_id}
	num_classes = len(map_id_label)	
	NUM_CLASSES = num_classes

	for class_id in class_id_set:

		subdir = src_dir + '/' + str(class_id)
		print(subdir)
		files = os.listdir(subdir)
		num_files = len(files)
		
		for index_file, filename in enumerate(files):

			base = os.path.splitext(filename)[0]
			ext = os.path.splitext(filename)[1]
			if not ext in {'.jpg', ".png"} : continue

			# ????
			#if base.split('_')[-1] != '0p': continue # use only _0p.jpg files

			class_index = map_id_label[class_id]
			
			label = class_index
			#label = [0]*num_classes
			#label[class_index] = 1

			file_path = subdir + '/' + filename
				
			#im = Image.open(file_path)		
			#im = im.resize(image_size, Image.ANTIALIAS)
			#arr = np.array(im, dtype=np.float32) / 256				
			#feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })			
			#feature_vectors.append(feature_vector)
			
			feature_vectors.append(0) # not used
			filenames.append(file_path) # filename or file_path
			labels.append(label)

			#im.close()
			print("dir={0}, class={1}: {2}/{3}: {4}".format(class_id, class_index, index_file, num_files, filename))
	
	print('----')
	print('Number of classes: {0}'.format(num_classes))	
	print('Number of feature vectors: {0}'.format(len(feature_vectors)))	

	data = {'images':feature_vectors, 'labels': labels, 'filenames':filenames}

	# mix data	
	if DO_MIX:
		print('start mix data')
		zip3 = list(zip(data['images'], data['labels'], data['filenames']))
		random.shuffle(zip3)
		print('mix ok')
		data['images']    = [x[0] for x in zip3]
		data['labels']    = [x[1] for x in zip3]
		data['filenames'] = [x[2] for x in zip3]

	print('Split data')
	data = split_data.split_data_v3(data, ratio=ratio)


	assert type(data['train']['labels'][0]) is int
	assert type(data['train']['filenames'][0]) is str
	#print(data['train']['labels'])
	#print(data['train']['filenames'])
	print('TRAIN')
	for i in range(len(data['train']['labels'])):
		print('{0} - {1}'.format(data['train']['labels'][i], data['train']['filenames'][i]))
	print('VALID')
	for i in range(len(data['valid']['labels'])):
		print('{0} - {1}'.format(data['valid']['labels'][i], data['valid']['filenames'][i]))

	data['id_label'] = map_id_label
	data['label_id'] = map_label_id
	data['num_classes'] = num_classes

	return data


"""
def save_bottleneck_to_txt_file(bottleneck_data):	

	with open('bottleneck_data.txt', 'wt') as f:
		for key in bottleneck_data:
			if key not in {'train','valid','test'}: continue
			f.write('\n PART {0}:\n'.format(key))			
			for i in range(len(bottleneck_data[key]['labels'])):
				f.write('{0}: {1}\n'.format(\
					bottleneck_data[key]['labels'][i], np.mean(bottleneck_data[key]['images'][i])))
"""


def input_parser(image_path, label, num_classes):
	# convert the label to one-hot encoding
	#NUM_CLASSES = 11
	#input_height, input_width = SHAPE[0], SHAPE[1]

	one_hot = tf.one_hot(label, num_classes)
	#one_hot = tf.constant(np.array([1,2,3,4,5,6,7,8,9,10,11]))

	# read the img from file
	image_string = tf.read_file(image_path)
	image_decoded = tf.image.decode_jpeg(image_string)
	image_resized = tf.image.resize_images(image_decoded, [SHAPE[1], SHAPE[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
	image = tf.cast(image_resized, tf.float32) / tf.constant(255.0)

	"""
	decoded_image = tf.image.decode_image(image_file, channels=3)
	decoded_image_as_float = tf.image.convert_image_dtype(
		decoded_image, tf.float32)
	
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)

	resize_shape = tf.stack([input_height, input_width])
	resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
	resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)

	"""
	return image, one_hot


def make_tf_dataset(filenames_data):

	print('Train labels:', filenames_data['train']['labels'])
	print('Valid labels:', filenames_data['valid']['labels'])


	# 
	#print(filenames_data['train']['filenames'])
	train_filenames = tf.constant(filenames_data['train']['filenames'])
	train_labels = tf.constant(filenames_data['train']['labels'])
	valid_filenames = tf.constant(filenames_data['valid']['filenames'])
	valid_labels = tf.constant(filenames_data['valid']['labels'])
	test_filenames = tf.constant(filenames_data['test']['filenames'])
	test_labels = tf.constant(filenames_data['test']['labels'])
	#print(train_labels)

	# create TensorFlow Dataset objects
	train_data = Dataset.from_tensor_slices((train_filenames, train_labels))
	valid_data = Dataset.from_tensor_slices((valid_filenames, valid_labels))
	test_data  = Dataset.from_tensor_slices((test_filenames, test_labels))
	print(train_data)
	print(valid_data)
	print(test_data)

	# load images and labels:
	num_classes = filenames_data['num_classes']
	input_parser_two_arg = lambda x,y : input_parser(x, y, num_classes)
	train_data = train_data.map(input_parser_two_arg)
	valid_data = valid_data.map(input_parser_two_arg)
	test_data  = test_data.map(input_parser_two_arg)

	#dataset = dataset.batch(batch_size)
	do_augmentation = True
	if do_augmentation: 
		train_data = distort.augment_dataset(train_data, mult=2)

	batch_size = 16
	train_data = train_data.batch(batch_size)
	valid_data = valid_data.batch(batch_size)
	test_data  = test_data.batch(batch_size)

	dataset = {'train':train_data, 'valid':valid_data, 'test':test_data}	

	return 	dataset




def train_and_save_model(dataset, shape):

	train_data = dataset['train']
	valid_data = dataset['valid']
	test_data  = dataset['test']
	print(train_data)
	print(valid_data)
	#sys.exit(0)


	# create TensorFlow Iterator object
	iterator = Iterator.from_structure(train_data.output_types,
	                                   train_data.output_shapes)
	
	next_element = iterator.get_next() #features, labels = iterator.get_next()

	# create two initialization ops to switch between the datasets
	train_init_op = iterator.make_initializer(train_data)
	valid_init_op = iterator.make_initializer(valid_data)

	# 3) Calculate bottleneck in TF
	height, width, color =  shape
	#x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='input')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
		
	if use_hub:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1", 
			trainable=True)
			#trainable=False)
		print(module._graph)
	else:
		module = model.module

	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]	

	print('bottleneck_tensor:', bottleneck_tensor)

	#NUM_CLASSES = 112
	NUM_CLASSES = 6

	print('NUM_CLASSES =', NUM_CLASSES)
	logits = last_layers(bottleneck_tensor, bottleneck_tensor_size, NUM_CLASSES)

	y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='output')   # Placeholder for labels.

	# for train for classification:
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # top-1

	acc_top5 = tf.nn.in_top_k(logits, tf.argmax(y,1), 5)
	acc_top6 = tf.nn.in_top_k(logits, tf.argmax(y,1), 6)

	#bottleneck_data = dict()
	#bottleneck_data['train'] = {'images':[], 'labels':[]}
	#bottleneck_data['valid'] = {'images':[], 'labels':[]}
	#bottleneck_data['test'] =  {'images':[], 'labels':[]}

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		NUM_EPOCH = 20

		for epoch in range(NUM_EPOCH):
			print('\nEPOCH {0}'.format(epoch))			

			# initialize the iterator on the training data
			sess.run(train_init_op) # switch to train dataset
			i = 0
			# get each element of the training dataset until the end is reached
			while True:
				i += 1
				try:
					batch = sess.run(next_element)

					# Perform one training iteration.
					sess.run(train_op, {x: batch[0], y: batch[1]}) 
					train_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
					#print('train batch', i)
					print('epoch={0} i={1} train_acc={2:.4f}'.format(epoch, i, train_acc))

					"""
					feature_vectors = bottleneck_tensor.eval(feed_dict={ x : batch[0] })
					images = list(map(list, feature_vectors))
					labels = list(map(list, batch[1]))
					bottleneck_data['train']['images'] += images
					bottleneck_data['train']['labels'] += labels
					#print(labels)
					"""

				except tf.errors.OutOfRangeError:
					print("End of training dataset.")
					break
				
			# initialize the iterator on the validation data
			sess.run(valid_init_op)
			# get each element of the validation dataset until the end is reached
			i = 0
			while True:
				i += 1
				try:
					print('valid batch', i)
					batch = sess.run(next_element)
					valid_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
					print('epoch={0} i={1} valid_acc={2:.4f}'.format(epoch, i, valid_acc))
					#print('i={0} valid_acc={1:.4f}'.format(i, valid_acc))

				except tf.errors.OutOfRangeError:
					print("End of validation dataset.")
					break

		saver = tf.train.Saver()		
		saver.save(sess, './save_model/{0}'.format(CHECKPOINT_NAME))  

		# SAVE GRAPH TO PB
		graph = sess.graph			
		tf.graph_util.remove_training_nodes(graph.as_graph_def())
		# tf.contrib.quantize.create_eval_graph(graph)
		# tf.contrib.quantize.create_training_graph()
		output_node_names = ['output']
		output_graph_def = tf.graph_util.convert_variables_to_constants(
			sess, graph.as_graph_def(), output_node_names)
		# save graph:		
		dir_for_model = '.'
		tf.train.write_graph(output_graph_def, dir_for_model,
			'saved_model_full.pb', as_text=False)	

	return True	

	"""
	# 3) Calculate bottleneck in TF
	height, width, color =  shape
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)			

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.

		feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })			

		feature_vectors.append(feature_vector)
		labels.append(label)
		filenames.append(filename) # or file_path
	"""
#------------------------------------------------




def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--src_dir', default=None, type=str,\
		help='input dir')
	parser.add_argument('-o', '--dst_file', default=None, type=str,\
		help='output file')
	parser.add_argument('-m', '--mix', dest='mix', action='store_true')

	parser.add_argument('-n', '--num', default=100, type=int,\
		help='num_angles for a single picture')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	#NUM_ANGLES 	 = arguments.num
	#DO_MIX 		 = arguments.mix
	#print('NUM_ANGLES =', 	NUM_ANGLES)
	#print('DO_MIX =',		DO_MIX)

	if not arguments.src_dir:
		src_dir = data_dir
	if not arguments.dst_file:		
		dst_file = 'dump.gz'

	filenames_data = make_filenames_list_from_subdir(
		src_dir=src_dir, shape=SHAPE, ratio=[9,1,1])

	dataset = make_tf_dataset(filenames_data)

	train_and_save_model(dataset, shape=SHAPE)
