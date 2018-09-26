import tensorflow as tf
import uff
import tensorrt
from tensorrt.parsers import uffparser

FROZEN_FPATH = '/root/tmp/saved_model_inception_resnet.pb'
ENGINE_FPATH = '/root/tmp/engine.plan'
INPUT_NODE = 
OUTPUT_NODE = 
INPUT_SIZE = [3, 299, 299]
MAX_BATCH_SIZE = 1
MAX_WORKSPACE = 1 << 20