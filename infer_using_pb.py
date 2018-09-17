import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile
from PIL import Image
import sys


def inference(image_file):

	# Read the image & get statstics
	image = Image.open(image_file)
	#img.show()
	width, height = image.size
	print(width)
	print(height)
	shape = [299, 299]
	#image = tf.image.resize_images(img, shape, method=tf.image.ResizeMethod.BICUBIC)

	image = image.resize(shape, Image.ANTIALIAS)
	image_arr = np.array(image, dtype=np.float32) / 256

	#Plot the image
	#image.show()

	with open('labels.txt') as f:
		labels = f.readlines()
		labels = [x.strip() for x in labels]
		print(labels)
	#sys.exit(0)

	with tf.Graph().as_default() as graph:

		with tf.Session() as sess:

			# Load the graph in graph_def
			print("load graph")

			# We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
			with gfile.FastGFile("output_graph.pb",'rb') as f:

				#Set default graph as current graph
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				#sess.graph.as_default() #new line

				# Import a graph_def into the current default Graph
				input_, predictions =  tf.import_graph_def(graph_def, name='', 
					return_elements=['Placeholder:0', 'final_result:0'])

				p_val = predictions.eval(feed_dict={input_: [image_arr]})
				index = np.argmax(p_val)
				label = labels[index]

				print(p_val)
				print(np.max(p_val))
				print('label: ', label)

				return label

if __name__ == '__main__':

	label = inference('img120.jpg')