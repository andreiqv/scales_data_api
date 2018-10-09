#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import networks

use_hub = os.path.exists('.use_hub')

if not use_hub:  # for local testing
	#data_dir = '../data'
	data_dir = '../separated'

	module = networks.conv_network_224
	SHAPE = 224, 224, 3

	bottleneck_tensor_size =  588 #1536
	BATCH_SIZE = 4
	DISPLAY_INTERVAL, NUM_ITERS = 10, 100

else:
	#data_dir = '/home/andrei/Data/Datasets/Scales/data'
	#data_dir = '/home/andrei/Data/Datasets/Scales/separated_cropped'
	data_dir = '/home/andrei/Data/Datasets/Scales/classsifier_dataset_081018'

	data_dir = data_dir.rstrip('/')
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

	bottleneck_tensor_size =  1536
	#bottleneck_tensor_size =  1001
	BATCH_SIZE = 32
	#DISPLAY_INTERVAL, NUM_ITERS = 1000, 1000*1500
	DISPLAY_INTERVAL, NUM_ITERS = 1000, 1000*2500