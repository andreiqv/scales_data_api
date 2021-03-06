#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import networks

use_hub = not os.path.exists('.dont_use_hub')
USE_HUB = use_hub

DO_BALANCING = False


if use_hub:
	#data_dir = '/home/andrei/Data/Datasets/Scales/data'
	#data_dir = '/home/andrei/Data/Datasets/Scales/separated_cropped'
	data_dir = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018'
	data_dir = data_dir.rstrip('/')
	DATASET_DIR = data_dir	
	
	import tensorflow_hub as hub
	model_number = 3
	type_model = 'feature_vector'
	#type_model = 'classification'
	
	if model_number == 0:		
		module = networks.conv_network_224
		bottleneck_tensor_size =  1536
		SHAPE = 224, 224, 3	
	elif model_number == 1:		
		module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/{0}/1".format(type_model))
		bottleneck_tensor_size =  1536
		SHAPE = 224, 224, 3
	elif model_number == 2:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/{0}/1".format(type_model))
		bottleneck_tensor_size =  1536
		SHAPE = 299, 299, 3
	elif model_number == 3:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model))
		bottleneck_tensor_size =  1536
		SHAPE = 299, 299, 3
		
		#module = lambda : hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model))
	else:
		raise Exception('Bad n_model')
		# https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1
	
	#bottleneck_tensor_size =  1001
	BATCH_SIZE = 32
	#DISPLAY_INTERVAL, NUM_ITERS = 1000, 1000*1500
	LEARNING_RATE = 0.01
	DISPLAY_INTERVAL = 1000 
	NUM_ITERS = 1000*500


#------------------------------------------
else:  # for local testing without tf.hub
	#data_dir = '../data'
	data_dir = '../separated'
	DATASET_DIR = data_dir

	module = networks.conv_network_224
	bottleneck_tensor_size =  588 #1536
	SHAPE = 224, 224, 3
	
	BATCH_SIZE = 4
	LEARNING_RATE = 0.01
	DISPLAY_INTERVAL, NUM_ITERS = 10, 100

	