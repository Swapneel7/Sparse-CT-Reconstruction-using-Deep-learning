import tensorflow as tf
import numpy as np

# # fixed parameters
ROW = 64
COLUMN = 64


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


def calcRMSE(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))




class Flag(object):
    def __init__(self):
        pass