"""
This file contains utility functions for creating a convolution neural network.
"""

import tensorflow as tf

from neural_util import conv2d_mirror_padding, conv2d_transpose_mirror_padding

WEIGHTS_INIT_STDEV = .1
def conv_layer(net, num_filters, filter_size, strides, relu=True, mirror_padding = True, one_hot_style_vector = None, name ='', reuse = False):
    with tf.variable_scope('conv_layer' + name, reuse=reuse):
        weights_init = conv_init_vars(net, num_filters, filter_size, name=name, reuse = reuse)
        strides_shape = [1, strides, strides, 1]
        if mirror_padding:
            net = conv2d_mirror_padding(net, weights_init, None, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = instance_norm(net, name=name, one_hot_style_vector = one_hot_style_vector, reuse = reuse)
        if relu:
            # net = tf.nn.relu(net)
            net = tf.nn.elu(net)

        return net

def conv_tranpose_layer(net, num_filters, filter_size, strides, mirror_padding = True, one_hot_style_vector = None, name ='', reuse = False):
    with tf.variable_scope('conv_tranpose_layer' + name, reuse=reuse):
        weights_init = conv_init_vars(net, num_filters, filter_size, transpose=True, name=name, reuse = reuse)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.pack(new_shape)
        strides_shape = [1,strides,strides,1]

        if mirror_padding:
            net = conv2d_transpose_mirror_padding(net, weights_init, None, tf_shape, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = instance_norm(net, name=name, one_hot_style_vector = one_hot_style_vector, reuse = reuse)
        return tf.nn.elu(net)  # tf.nn.relu(net)

def residual_block(net, filter_size=3, mirror_padding = True, name ='', one_hot_style_vector = None, reuse = False):
    tmp = conv_layer(net, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_first', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    return net + conv_layer(tmp, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_second', relu=False, one_hot_style_vector = one_hot_style_vector, reuse = reuse)

def instance_norm(net, name ='', one_hot_style_vector = None, reuse = False):
    with tf.variable_scope('instance_norm' + name, reuse=reuse):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        if one_hot_style_vector is None:
            var_shape = [channels]
        else:
            num_styles = one_hot_style_vector.get_shape().as_list()[1]
            var_shape = [num_styles, channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        # Try applying an abs on the sigma_sq. in theory it should always be positive but in practice due to inaccuracy in float calculation, it may be negative when the actual sigma is very small, which causes the output to be NaN sometimes.
        sigma_sq = tf.abs(sigma_sq)
        shift_init = tf.zeros(var_shape)
        shift = tf.get_variable('shift', initializer= shift_init)
        scale_init = tf.ones(var_shape)
        scale = tf.get_variable('scale', initializer= scale_init)
        if one_hot_style_vector is not None:
            shift = tf.matmul(one_hot_style_vector, shift)
            scale = tf.matmul(one_hot_style_vector, scale)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

def conv_init_vars(net, out_channels, filter_size, transpose=False, name ='', reuse = False):
    with tf.variable_scope('conv_init_vars' + name, reuse=reuse):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
        weights_initializer = tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
        weights_init = tf.get_variable('weights_init', dtype=tf.float32, initializer=weights_initializer)
        return weights_init

def fully_connected(net, out_channels, name ='', reuse = False):
    with tf.variable_scope('fully_connected_' + name, reuse=reuse):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]

        weights_shape = [rows*cols*in_channels, out_channels]
        weights_initializer = tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
        weights_init = tf.get_variable('weights_init', dtype=tf.float32, initializer=weights_initializer)

        bias_shape = [out_channels]
        bias_initializer = tf.truncated_normal(bias_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
        bias_init = tf.get_variable('bias_init', dtype=tf.float32, initializer=bias_initializer)

        fc1 = tf.reshape(net, [-1, rows*cols*in_channels])
        fc1 = tf.add(tf.matmul(fc1, weights_init), bias_init)

        fc1 = tf.nn.elu(fc1)

        return fc1