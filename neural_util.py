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
