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

import networks
import split_data
import distort

DO_MIX = True
NUM_CLASSES = 0

MIN_NUM_IMAGES = 5
DO_RESIZE = True

if os.path.exists('.notebook'):
	#data_dir = '../data'
	data_dir = '../separated'
	module = networks.conv_network_224
	SHAPE = 224, 224, 3
else:
	#data_dir = '/home/chichivica/Data/Datasets/Scales/data'
	#data_dir = '/home/chichivica/Data/Datasets/Scales/separated'
	data_dir = '/home/chichivica/Data/Datasets/Scales/separated_cropped_mult'
	import tensorflow_hub as hub

	model_number = 3
	type_model = 'feature_vector'
	#type_model = 'classification'
	
	if model_number == 1:		
		module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/{0}/1".format(type_model))
		SHAPE = 224, 224, 3
	elif model_number == 2:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	elif model_number == 3:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	else:
		raise Exception('Bad n_model')
		# https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1

np.set_printoptions(precision=4, suppress=True)


#---------------------------------



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



#------------------------------------------------

def make_filenames_list_from_subdir(src_dir):
	"""
	Use names of subdirs as a id.
	And then calculate class_index from id.
	"""

	filenames = []
	
	files = os.listdir(src_dir)
	
	for index_file, filename in enumerate(files):

		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.jpg', ".png"} : continue

		#if base.split('_')[-1] != '0p': continue # use only _0p.jpg files

		file_path = src_dir + '/' + filename
				
		#feature_vectors.append(0) # not used
		filenames.append(file_path) # filename or file_path
	
	return filenames


def input_parser(image_path):
	# convert the label to one-hot encoding

	# read the img from file
	image_string = tf.read_file(image_path)
	image_decoded = tf.image.decode_jpeg(image_string)

	#do_resize = False
	if DO_RESIZE:
		image_resized = tf.image.resize_images(image_decoded, [SHAPE[1], SHAPE[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
	else:
		image_resized = image_decoded

	image = tf.cast(image_resized, tf.float32) #/ tf.constant(255.0)

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
	return image


def make_tf_dataset(filenames, mult):

	train_images = tf.constant(filenames)
	# create TensorFlow Dataset objects
	dataset = Dataset.from_tensor_slices(train_images)
	#valid_data = Dataset.from_tensor_slices((valid_images, valid_labels))

	dataset = dataset.map(input_parser)
	#valid_data = valid_data.map(input_parser_two_arg)
	#dataset = dataset.batch(batch_size)

	if True: # Distrot train dataset
		dataset = distort.augment_dataset_no_labels(dataset, mult=mult)

	#batch_size = 16
	#dataset = dataset.batch(batch_size)

	return 	dataset


def save_dataset(dataset, shape, subdir_name):

	#dataset = dataset['train']
	#valid_data = dataset['valid']
	#test_data  = dataset['test']
	print(dataset)
	#print(valid_data)
	#sys.exit(0)

	dataset = dataset.batch(16)

	# create TensorFlow Iterator object
	iterator = Iterator.from_structure(dataset.output_types,
	                                   dataset.output_shapes)
	
	next_element = iterator.get_next() #features, labels = iterator.get_next()

	# create two initialization ops to switch between the datasets
	train_init_op = iterator.make_initializer(dataset)
	#valid_init_op = iterator.make_initializer(valid_data)

	# 3) Calculate bottleneck in TF
	height, width, color =  shape
	x = tf.placeholder(tf.float32, [height, width, 3], name='Placeholder-x')

	#x = x * tf.constant(255.0)

	#resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	
	# num_features = 2048, height x width = 224 x 224 pixels
	#assert height, width == hub.get_expected_image_size(module)	
	#bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	#print('bottleneck_tensor:', bottleneck_tensor)

	#bottleneck_data = dict()
	#bottleneck_data['train'] = {'images':[], 'labels':[]}
	#bottleneck_data['valid'] = {'images':[], 'labels':[]}
	#bottleneck_data['test'] =  {'images':[], 'labels':[]}	

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		# initialize the iterator on the training data
		sess.run(train_init_op) # switch to train dataset
		i = 0
		# get each element of the training dataset until the end is reached
		while True:
			i += 1
			try:				
				batch = sess.run(next_element)
				#jpeg_image = tf.image.encode_jpeg(item)
				
				#op = tf.image.encode_jpeg(item, format='rgb', quality=100)

				for j, item in enumerate(batch):
					op = tf.image.encode_jpeg(item)

					#data_np = sess.run(op, feed_dict={ x: item })
					#print(data_np)
					image = op.eval()
					#print(image)

					print('i = {0}, j = {1}'.format(i,j))
					fname = tf.constant('{0}/augment_{1}_{2}.jpg'.format(subdir_name,i,j))
					wr = tf.write_file(fname, image)
					sess.run(wr)

				#with open('01.jpg', 'wb') as fp:
				#	fp.write(image)

				#tf.write_file(fname, data_np * tf.constant(255.0))
				#print("tf.write_file('1.jpg', jpeg_image)")

			except tf.errors.OutOfRangeError:
				print("End of training dataset.")
				break
			

	return 0			

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


def dataset_preprocessing(data_dir):

	subdir_list = os.listdir(data_dir)

	for item in subdir_list:

		subdir = data_dir + '/' + item

		print('Subdir {0}'.format(subdir))		
		if not os.path.isdir(subdir): continue
		
		files = os.listdir(subdir)
		num = len(files)
		if num < 1: continue

		min_num = MIN_NUM_IMAGES		
		if num < min_num:
			mult = math.floor(min_num / num)
		else:
			mult = 1
		print('num={}, mult={}'.format(num, mult))

		if mult > 1:
			filenames = make_filenames_list_from_subdir(src_dir=subdir)
			dataset = make_tf_dataset(filenames, mult=mult)
			save_dataset(dataset, shape=SHAPE, subdir_name=subdir)


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	NUM_ANGLES 	 = arguments.num
	#DO_MIX 		 = arguments.mix
	print('NUM_ANGLES =', 	NUM_ANGLES)
	#print('DO_MIX =',		DO_MIX)

	if not arguments.src_dir:
		dataset_dir = data_dir
	if not arguments.dst_file:		
		dst_file = 'dump.gz'

	dataset_preprocessing(dataset_dir)
		

	"""
	bottleneck_data['id_label'] = filenames_data['id_label']
	bottleneck_data['label_id'] = filenames_data['label_id']
	bottleneck_data['num_classes'] = filenames_data['num_classes']

	print('Train size:', len(bottleneck_data['train']['images']))
	print('Valid size:', len(bottleneck_data['valid']['images']))
	print('Test size:', len(bottleneck_data['test']['images']))

	save_data_dump(bottleneck_data, dst_file=dst_file)
	"""
