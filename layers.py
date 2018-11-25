# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:45:50 2018

@author: liweipeng
"""

import tensorflow as tf

elu = tf.nn.elu  # elu activation
relu = tf.nn.relu  # relu activation
sigmoid = tf.nn.sigmoid  # sigmoid activation


def swish(x, beta=1):
    '''
    swish activation
    :param x: input tensor
    :param beta: factor
    :return: tensor
    '''
    return x * sigmoid(beta * x)


def lrelu(x, leak=0.2):
    '''
    lrelu activation
    :param x: input tensor
    :param leak: factor
    :return: tensor
    '''
    return tf.maximum(x, leak * x)


def group_norm(x, G=64, eps=1e-5):
    '''
    Group Normalization
    :param x: input tensor
    :param G: the number of group
    :param eps: a small float number to avoid dividing by 0
    :return: instance_normalized tensor
    '''
    with tf.variable_scope("group_norm"):
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [1, 1, 1, C],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [N, H, W, C]) * gamma + beta
        return x


def instance_norm(x):
    '''
    Instance Normalization
    :param x: input tensor
    :return: instance_normalized tensor
    '''
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
        return out


def conv2d(inputconv,
           o_d=64,
           f_h=7,
           f_w=7,
           s_h=1,
           s_w=1,
           padding="VALID",
           name="conv2d",
           do_norm=True,
           do_relu=True,
           relufactor=0
           ):
    '''
    convolution layer
    :param inputconv: input tensor
    :param o_d: output channels
    :param f_h: height of filter
    :param f_w: width of filter
    :param s_h: height of strides
    :param s_w: width of strides
    :param padding: method of padding
    :param name: operation name
    :param do_norm: whether instance_normalize
    :param do_relu: whether ReLU
    :param relufactor: factor of lrelu
    :return: tensor
    '''
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(
            inputconv,
            o_d,
            [f_w, f_h],
            [s_w, s_h],
            padding,
            activation_fn=None,
            weights_initializer=tf.glorot_normal_initializer(),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm: conv = instance_norm(conv)
        # if do_relu: conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        # if do_relu: conv = elu(conv)
        if do_relu: conv = swish(conv)
        return conv


def deconv2d(inputconv,
             o_d=64,
             f_h=7,
             f_w=7,
             s_h=1,
             s_w=1,
             padding="VALID",
             name="deconv2d",
             do_norm=True,
             do_relu=True,
             relufactor=0
             ):
    '''
    transpose convolution layer
    :param inputconv: input tensor
    :param o_d: output channels
    :param f_h: height of filter
    :param f_w: width of filter
    :param s_h: height of strides
    :param s_w: width of strides
    :param padding: method of padding
    :param name: operation name
    :param do_norm: whether instance_normalize
    :param do_relu: whether ReLU
    :param relufactor: factor of lrelu
    :return: tensor
    '''
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d_transpose(
            inputconv,
            o_d,
            [f_h, f_w],
            [s_h, s_w],
            padding,
            activation_fn=None,
            weights_initializer=tf.glorot_normal_initializer(),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if do_norm: conv = instance_norm(conv)
        # if do_relu: conv = relu(conv) if relufactor == 0 else lrelu(conv, relufactor)
        # if do_relu: conv = elu(conv)
        if do_relu: conv = swish(conv)
        return conv
