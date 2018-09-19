#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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

import network
import split_data
import distort

DO_MIX = True # mix dataset before division into train/valid/test subsets
NUM_CLASSES = 0

np.set_printoptions(precision=4, suppress=True)

from model import module, SHAPE, data_dir


#---------------------------------

def data_analysis(dir_path):

	class_id_set = set()
	#num_classes = 412
	files = os.listdir(dir_path)
	random.shuffle(files)
	num_files = len(files)	

	for index_file, filename in enumerate(files):

		#print(filename)
		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.jpg', ".png"} : continue			

		try:
			class_id = int(base.split('_')[-1])
		except:
			continue
	
		class_id_set.add(class_id)

	return class_id_set 	


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

	save_labels_to_file(map_label_id)

	for class_id in class_id_set:

		subdir = src_dir + '/' + str(class_id)
		print(subdir)
		files = os.listdir(subdir)
		num_files = len(files)
		
		for index_file, filename in enumerate(files):

			base = os.path.splitext(filename)[0]
			ext = os.path.splitext(filename)[1]
			if not ext in {'.jpg', ".png"} : continue

			if base.split('_')[-1] != '0p': continue # use only _0p.jpg files

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

	print('Split data 22')
	#data = split_data.split_data(data, ratio=ratio)
	data = split_data.split_data_2(data, ratio=ratio, num_labels=num_classes)
	sys.exit(0)

	data['id_label'] = map_id_label
	data['label_id'] = map_label_id
	data['num_classes'] = num_classes

	return data


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
	#image = tf.cast(image_resized, tf.float32) / tf.constant(255.0)
	image = tf.cast(image_resized, tf.float32)

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
	train_images = tf.constant(filenames_data['train']['filenames'])
	train_labels = tf.constant(filenames_data['train']['labels'])
	valid_images = tf.constant(filenames_data['valid']['filenames'])
	valid_labels = tf.constant(filenames_data['valid']['labels'])
	test_images = tf.constant(filenames_data['test']['filenames'])
	test_labels = tf.constant(filenames_data['test']['labels'])
	#print(train_labels)

	# create TensorFlow Dataset objects
	train_data = Dataset.from_tensor_slices((train_images, train_labels))
	valid_data = Dataset.from_tensor_slices((valid_images, valid_labels))
	test_data  = Dataset.from_tensor_slices((test_images, test_labels))
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

	# AUGMENTATION (Distrot only train dataset)
	do_augmentation = True
	if do_augmentation: 
		train_data = distort.augment_dataset_2(train_data)
		#train_data = distort.augment_dataset(train_data)

	batch_size = 16
	train_data = train_data.batch(batch_size)
	valid_data = valid_data.batch(batch_size)
	test_data  = test_data.batch(batch_size)

	dataset = {'train':train_data, 'valid':valid_data, 'test':test_data}	

	return 	dataset


def make_bottleneck_with_tf(dataset, shape):

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
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)

	bottleneck_data = dict()
	bottleneck_data['train'] = {'images':[], 'labels':[]}
	bottleneck_data['valid'] = {'images':[], 'labels':[]}
	bottleneck_data['test'] =  {'images':[], 'labels':[]}

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		# initialize the iterator on the training data
		sess.run(train_init_op) # switch to train dataset
		i = 0
		# get each element of the training dataset until the end is reached
		while True:
			i += 1
			try:
				print('train batch', i)
				batch = sess.run(next_element)				
				#img, label = elem
				feature_vectors = bottleneck_tensor.eval(feed_dict={ x : batch[0] })
				#label = elem[1]
				images = list(map(list, feature_vectors))
				labels = list(map(list, batch[1]))
				bottleneck_data['train']['images'] += images
				bottleneck_data['train']['labels'] += labels
				#print(labels)
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
				feature_vectors = bottleneck_tensor.eval(feed_dict={ x : batch[0] })
				images = list(map(list, feature_vectors))
				labels = list(map(list, batch[1]))
				bottleneck_data['valid']['images'] += images
				bottleneck_data['valid']['labels'] += labels								
				#print(labels)
			except tf.errors.OutOfRangeError:
				print("End of validation dataset.")
				break

	return bottleneck_data			

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


def save_data_dump(data, dst_file):
	
	# save the data on a disk
	dump = pickle.dumps(data)
	print('dump done')
	f = gzip.open(dst_file, 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()



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
	NUM_ANGLES 	 = arguments.num
	#DO_MIX 		 = arguments.mix
	print('NUM_ANGLES =', 	NUM_ANGLES)
	#print('DO_MIX =',		DO_MIX)

	if not arguments.src_dir:
		src_dir = data_dir
	if not arguments.dst_file:		
		dst_file = 'dump.gz'

	filenames_data = make_filenames_list_from_subdir(
		src_dir=src_dir, shape=SHAPE, ratio=[1,1,1])

	dataset = make_tf_dataset(filenames_data)

	bottleneck_data = make_bottleneck_with_tf(dataset, shape=SHAPE)

	bottleneck_data['id_label'] = filenames_data['id_label']
	bottleneck_data['label_id'] = filenames_data['label_id']
	bottleneck_data['num_classes'] = filenames_data['num_classes']

	print('Train size:', len(bottleneck_data['train']['images']))
	print('Valid size:', len(bottleneck_data['valid']['images']))
	print('Test size:', len(bottleneck_data['test']['images']))

	save_data_dump(bottleneck_data, dst_file=dst_file)
