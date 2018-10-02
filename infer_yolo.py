#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Using TF for inference, and TensorRT for compress a graph.

import sys
import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import timer

import tensorflow.contrib.tensorrt as trt


nn_model = 'yolo'

if nn_model == 'hub':
	FROZEN_FPATH = '/home/andrei/Data/Datasets/Scales/pb/output_graph.pb'
	ENGINE_FPATH = '/home/andrei/Data/Datasets/Scales/pb/hub_model_engine.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'final_result'
	INPUT_SIZE = [3, 299, 299]
	sinput_output_placeholders = ['Placeholder:0', 'final_result:0']

elif nn_model == 'my':
	#FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
	#ENGINE_FPATH = '/root/tmp/engine.plan'
	FROZEN_FPATH = 'saved_model_full_2.pb'
	ENGINE_FPATH = 'saved_model_full_2.plan'
	INPUT_NODE = 'Placeholder-x'
	OUTPUT_NODE = 'sigmoid_out'
	INPUT_SIZE = [3, 299, 299]
	input_output_placeholders = ['Placeholder-x:0', 'sigmoid_out:0']

elif nn_model == 'yolo':

	FROZEN_FPATH = '/root/tmp/yolov3-coco-nchw.pb'
	#ENGINE_FPATH = 'saved_model_full_2.plan'
	INPUT_NODE = 'Placeholder'
	OUTPUT_NODE = 'concat_scaled_boxes'
	INPUT_SIZE = [3, 608, 608]
	input_output_placeholders = ['Placeholder:0', 'concat_scaled_boxes:0']

else:
	raise Exception('nn_model is wrong')



def get_image_as_array(image_file):

	# Read the image & get statstics
	image = Image.open(image_file)
	#img.show()
	#shape = [299, 299]
	shape = tuple(INPUT_SIZE[1:])
	#image = tf.image.resize_images(img, shape, method=tf.image.ResizeMethod.BICUBIC)
	image = image.resize(shape, Image.ANTIALIAS)
	image_arr = np.array(image, dtype=np.float32) / 256.0

	return image_arr


def get_labels(labels_file):	

	with open(labels_file) as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
		print(labels)
	#sys.exit(0)
	return labels


def get_frozen_graph(pb_file):

	# We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
	with gfile.FastGFile(pb_file,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		#sess.graph.as_default() #new line	
	return graph_def



def compress_graph_with_trt(graph_def, precision_mode):

	output_node = input_output_placeholders[1]

	if precision_mode==0: 
		return graph_def

	trt_graph = trt.create_inference_graph(
		graph_def,
		[output_node],
		max_batch_size=1,
		max_workspace_size_bytes=2<<20,
		precision_mode=precision_mode)

	return trt_graph


def inference_with_graph(graph_def, image, labels):

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Import a graph_def into the current default Graph
			print("import graph")	
			input_, predictions =  tf.import_graph_def(graph_def, name='', 
				return_elements=input_output_placeholders)
			
			timer.timer('predictions.eval')

			time_res = []
			for i in range(10):
				p_val = predictions.eval(feed_dict={input_: [image]})
				index = np.argmax(p_val)
				label = labels[index]
				dt = timer.timer('{0}: label={1}'.format(i, label))
				time_res.append(0)
				#print('index={0}, label={1}'.format(index, label))

			print('mean time = {0}'.format(np.mean(time_res)))

			return index


def createParser ():
	"""ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', default="images/1.jpg", type=str,\
		help='input')
	parser.add_argument('-pb', '--pb', default="saved_model_full.pb", type=str,\
		help='input')
	parser.add_argument('-o', '--output', default="logs/1/", type=str,\
		help='output')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])		
	pb_file = arguments.pb

	image = get_image_as_array(arguments.input)
	labels = get_labels('labels.txt')
	graph_def = get_frozen_graph(pb_file)

	modes = ['FP32', 'FP16', 0]
	#precision_mode = modes[2]

	#pb_file_name = 'saved_model.pb' # output_graph.pb

	for mode in modes*2:
		print('\nMODE: {0}'.format(mode))
		graph_def = compress_graph_with_trt(graph_def, mode)
		inference_with_graph(graph_def, image, labels)
		