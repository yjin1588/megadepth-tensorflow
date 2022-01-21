import tensorflow as tf
from tensorflow.python.keras import backend as K

import math, os
import cv2

from dataset.data_generator import Dataset

def train(args):
    sess = tf.Session()
    K.set_session(sess)

    with tf.name_scope('input'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name='image')
        m = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name='mask')
        n = tf.placeholder(dtype=tf.string, shape=(None,), name='image_name')
        y = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1), name='gt')
        is_ord = tf.placeholder(dtype=tf.float32, shape=(None,), name='is_ordinal')

    if args.dataset == 'md':
        dataset_train_landscape = Dataset(args, 'train', 'land', should_shuffle=True, should_repeat=False)
        dataset_train_portrait = Dataset(args, 'train', 'port', should_shuffle=True, should_repeat=False)
        
    