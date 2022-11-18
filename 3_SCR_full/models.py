import math
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def Encoder(x, N, M, data_format):
    with tf.variable_scope("E", reuse=tf.AUTO_REUSE) as vs:
        # Encoder
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        x = tf.contrib.layers.gdn(x, inverse=False)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        x = tf.contrib.layers.gdn(x, inverse=False)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        x = tf.contrib.layers.gdn(x, inverse=False)
        # =======================================================================================================
        filter_num = M
        filter_size = 5
        stride = 2
        out = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def Decoder(x,input_channel, N, M, data_format):
    with tf.variable_scope("D", reuse=tf.AUTO_REUSE) as vs:
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution
        x = tf.contrib.layers.gdn(x, inverse=True)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution
        x = tf.contrib.layers.gdn(x, inverse=True)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution
        x = tf.contrib.layers.gdn(x, inverse=True)
        # =======================================================================================================
        filter_num = input_channel
        filter_size = 5
        stride = 2
        out = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def Hyper_Encoder(x, N, data_format):
    with tf.variable_scope("HE", reuse=tf.AUTO_REUSE) as vs:
        x = tf.abs(x)
        # Encoder
        # =======================================================================================================
        filter_num = N
        filter_size = 3
        stride = 1
        x = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        x = tf.nn.relu(x)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        x = tf.nn.relu(x)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        out = slim.conv2d(x, filter_num, filter_size, stride, activation_fn=None)
        # =======================================================================================================
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def Hyper_Decoder(x, N, M, data_format):
    with tf.variable_scope("HD", reuse=tf.AUTO_REUSE) as vs1:
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution
        x = tf.nn.relu(x)
        # =======================================================================================================
        filter_num = N
        filter_size = 5
        stride = 2
        x = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                         activation_fn=None)  # transposed convolution
        penultimate = x = tf.nn.relu(x)
        # =======================================================================================================
        filter_num = M
        filter_size = 3
        stride = 1
        out = slim.convolution2d_transpose(x, filter_num, filter_size, stride, normalizer_fn=None,
                                           activation_fn=None)  # transposed convolution
        # =======================================================================================================
        sigma = tf.nn.relu(out)
    with tf.variable_scope("MASK") as vs2:
        filter_num = M
        filter_size = 1
        stride = 1
        out2 = slim.convolution2d_transpose(penultimate, filter_num, filter_size, stride, normalizer_fn=None,
                                            activation_fn=None)  # transposed convolution
    variables = tf.contrib.framework.get_variables(vs1)
    variables_mask = tf.contrib.framework.get_variables(vs2)
    importance_map = tf.clip_by_value(out2 + 0.5, 0, 1)
    return sigma, importance_map, variables, variables_mask

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    return shape
