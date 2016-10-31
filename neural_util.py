
import tensorflow as tf

def conv2d(input_layer, w, b, stride=1):
    """

    :param input_layer: input tensor
    :param w: Weight. Either tensorflow constant or variable.
    :param b: Bias. Either tensorflow constant or variable.
    :param stride: Stride of the conv.
    :return: the 2d convolution tensor.
    """
    conv_output = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.bias_add(conv_output, b)

def leaky_relu(input_layer, alpha):
    return tf.maximum(tf.mul(input_layer, alpha), input_layer)