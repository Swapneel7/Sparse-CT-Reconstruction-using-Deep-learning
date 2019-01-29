import tensorflow as tf
import numpy as np

# # fixed parameters
ROW = 1500
COLUMN = 1024


def transform_to_2D_tf(x):
    return tf.reshape(x, [-1, ROW*COLUMN])

def transform_to_3D_tf(x):
    return tf.reshape(x, [-1, ROW, COLUMN])

def transform_to_4D_tf(x):
    return tf.reshape(x, [-1, ROW, COLUMN, 1])

def transform_to_2D_np(x):
    return np.reshape(x, [-1, ROW*COLUMN])

def transform_to_3D_np(x):
    return np.reshape(x, [-1, ROW, COLUMN])

def transform_to_4D_np(x):
    return np.reshape(x, [-1, ROW, COLUMN, 1])



class Flag(object):
    def __init__(self):
        pass