# The code skeleton mainly comes from https://github.com/anishathalye/neural-style.
# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
import numpy as np
import scipy.io
import scipy.ndimage
import tensorflow as tf



def net(data_path, input_image, stride_multiplier = 1):
    # The stride multiplier feature is an attempt to make the style/texture features look larger. It is not fully
    # developed yet so please keep it at 1.
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias, stride_multiplier=stride_multiplier)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, stride_multiplier=stride_multiplier)
        net[name] = current

    assert len(net) == len(layers)
    return net, mean_pixel

# Given the path to vgg net, read the data and compute the mean pixel.
def read_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    return data, mean_pixel

# Given the data from scipy.io.loadmat(data_path), generate the net directly
def pre_read_net(data, input_image, stride_multiplier = 1):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias, stride_multiplier=stride_multiplier)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, stride_multiplier=stride_multiplier)
        net[name] = current

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias, stride_multiplier = 1):
    # The stride multiplier feature is an attempt to make the style/texture features look larger. It is not fully
    # developed yet so please keep it at 1.
    if stride_multiplier == 1:
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                padding='SAME')
    else:
        weights_repeated = np.repeat(np.repeat(weights, 2, axis=0), stride_multiplier, axis=1) / float(stride_multiplier ** 2)
        assert weights_repeated.shape[1] == weights.shape[1] * stride_multiplier and weights_repeated.shape[0] == weights.shape[0] * stride_multiplier
        conv = tf.nn.conv2d(input, tf.constant(weights_repeated), strides=(1, 1, 1, 1),
                padding='SAME')

    # conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1 * stride_multiplier, 1 * stride_multiplier, 1),
    #         padding='SAME')

    # weights_repeated = weights_shape_multiply(weights, stride_multiplier) / float(stride_multiplier ** 2)
    # assert weights_repeated.shape[1] == weights.shape[1] * stride_multiplier and weights_repeated.shape[0] == weights.shape[0] * stride_multiplier
    # conv = tf.nn.conv2d(input, tf.constant(weights_repeated), strides=(1, 1, 1, 1),
    #         padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, stride_multiplier = 1):
    # The stride multiplier feature is an attempt to make the style/texture features look larger. It is not fully
    # developed yet so please keep it at 1.
    # Multiply the ksize and strides by the stride multiplier will have an effect similar to making the features look
    # a little bit smaller.
    # return tf.nn.max_pool(input, ksize=(1, 2 * stride_multiplier, 2 * stride_multiplier, 1), strides=(1, 2 * stride_multiplier, 2 * stride_multiplier, 1),
    #         padding='SAME')
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel

def get_net_layer_sizes(net):
    net_layer_sizes = {}
    for key, val in net.iteritems():
        net_layer_sizes[key] = map(lambda i: i.value, val.get_shape())
    return net_layer_sizes

def weights_shape_multiply(weights, shape_multiplier):
    resized = scipy.ndimage.zoom(weights, (shape_multiplier,shape_multiplier,1,1), order=1)
    return resized