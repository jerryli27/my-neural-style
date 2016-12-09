import math
from operator import mul

import tensorflow as tf


def conv2d(input_layer, w, b, stride=1):
    """

    :param input_layer: Input tensor.
    :param w: Weight. Either tensorflow constant or variable.
    :param b: Bias. Either tensorflow constant or variable.
    :param stride: Stride of the conv.
    :return: the 2d convolution tensor.
    """
    conv_output = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(conv_output, b)


def conv2d_mirror_padding(input_layer, w, b, kernel_size, stride=1):
    """

    :param input_layer: Input tensor.
    :param w: Weight. Either tensorflow constant or variable.
    :param b: Bias. Either tensorflow constant or variable.
    :param stride: Stride of the conv.
    :return: the 2d convolution tensor with mirror padding.
    """
    # N_out = N_in / stride + 2N_pad - N_kernel_size + 1. We have N_out and N_in fixed (treat as 0) and solve for N_pad.
    n_pad = (kernel_size - 1) / 2
    padding = [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]
    mirror_padded_input_layer = tf.pad(input_layer, padding, "REFLECT", name='mirror_padding')
    conv_output = tf.nn.conv2d(mirror_padded_input_layer, w, strides=[1, stride, stride, 1], padding='VALID')
    if b is not None:
        return tf.nn.bias_add(conv_output, b)
    else:
        return conv_output


def conv2d_transpose_mirror_padding(input_layer, w, b, output_shape, kernel_size, stride=1):
    """
    TODO: For some reason transpose mirror padding is causing some error in the optimization step
    :param input_layer: Input tensor.
    :param w: Weight. Either tensorflow constant or variable.
    :param b: Bias. Either tensorflow constant or variable.
    :param stride: Stride of the conv.
    :return: the 2d convolution tensor with mirror padding.
    """
    # N_out = N_in / stride + 2N_pad - N_kernel_size + 1. We have N_out and N_in fixed (treat as 0) and solve for N_pad.
    n_pad = (kernel_size - 1) / 2
    padding = [[0, 0], [n_pad, n_pad], [n_pad, n_pad], [0, 0]]
    mirror_padded_input_layer = tf.pad(input_layer, padding, "REFLECT", name='mirror_padding')
    conv_output = tf.nn.conv2d_transpose(mirror_padded_input_layer, w, output_shape, strides=[1, stride, stride, 1], padding='VALID')
    if b is not None:
        return tf.nn.bias_add(conv_output, b)
    else:
        return conv_output

def leaky_relu(input_layer, alpha):
    return tf.maximum(input_layer * alpha, input_layer)

def gram_experiment(features, shift = None, shift_is_horizontal = True):
    _, height, width, number = map(lambda i: i.value, features.get_shape())
    size = height * width * number
    if shift is None:
        features = tf.reshape(features, (-1, number))
        gram = tf.matmul(tf.transpose(features), features) / size
    else:
        if shift_is_horizontal:
            left = tf.slice(features, [0, 0, 0, 0], [-1, -1, width - shift, -1])
            right = tf.slice(features, [0, 0, shift, 0], [-1, -1, -1, -1])
            left_reshaped = tf.reshape(left, (-1, number))
            right_reshaped = tf.reshape(right, (-1, number))
            gram = tf.matmul(tf.transpose(left_reshaped), right_reshaped) / size
        else:
            top = tf.slice(features, [0, 0, 0, 0], [-1, height - shift, -1, -1])
            bottom = tf.slice(features, [0, shift, 0, 0], [-1, -1, -1, -1])
            top_reshaped = tf.reshape(top, (-1, number))
            bottom_reshaped = tf.reshape(bottom, (-1, number))
            gram = tf.matmul(tf.transpose(top_reshaped), bottom_reshaped) / size


    return gram


def gram_stacks(features):
    _, height, width, number = map(lambda i: i.value, features.get_shape())
    good_old_gram = gram_experiment(features)
    gram = [good_old_gram]
    for shift in range(1,int(width),1):
        shifted_gram = gram_experiment(features, shift=shift)
        gram.append(shifted_gram)
    for shift in range(1,int(height),1):
        shifted_gram = gram_experiment(features, shift=shift, shift_is_horizontal=False)
        gram.append(shifted_gram)
    gram_stack = tf.pack(gram)  / math.sqrt(len(gram))
    return gram_stack

def get_tensor_num_elements(tensor):
    tensor_shape = map(lambda i: i.value, tensor.get_shape())
    return reduce(mul, tensor_shape, 1)