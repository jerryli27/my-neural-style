from operator import mul

import numpy as np
import tensorflow as tf

import vgg


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


def gram_experiment(features, horizontal_shift = 0, vertical_shift = 0):
    _, height, width, number = map(lambda i: i.value, features.get_shape())
    size = height * width * number

    features_unpacked = tf.unpack(features)

    grams = []
    for current_feature in features_unpacked:
        current_feature = tf.expand_dims(current_feature, 0)
        original = tf.slice(current_feature, [0, 0, 0, 0], [-1, height - vertical_shift, width - horizontal_shift, -1])
        shifted = tf.slice(current_feature, [0, vertical_shift, horizontal_shift, 0], [-1, -1, -1, -1])
        left_reshaped = tf.reshape(original, (-1, number))
        right_reshaped = tf.reshape(shifted, (-1, number))
        gram = tf.matmul(tf.transpose(left_reshaped), right_reshaped) / (size)
        grams.append(gram)
    grams = tf.pack(grams)
    return grams



def gram_stacks(features, shift_size=2, stride = 1):
    # This is the second attempt. It shifts the gram in a m x n range and calculate gram for each shift.
    batch_size, height, width, number = map(lambda i: i.value, features.get_shape())
    gram = []
    assert shift_size * stride < height and shift_size * stride < width

    for vertical_shift in range(shift_size):
        for horizontal_shift in range(shift_size):
            shifted_gram = gram_experiment(features, horizontal_shift * stride, vertical_shift * stride)
            gram.append(shifted_gram)
    gram_stack = tf.pack(gram)
    gram_stack = tf.transpose(gram_stack, (1,2,3,0)) # Shape = [batch_size, number, number, num_shifts]

    gram_stack_num_elements = get_tensor_num_elements(gram_stack)
    assert gram_stack_num_elements == (batch_size * number * number * shift_size * shift_size)
    return gram_stack

def get_tensor_num_elements(tensor):
    tensor_shape = map(lambda i: i.value, tensor.get_shape())
    return reduce(mul, tensor_shape, 1)

def apply_style_weight_mask_to_feature_layer(feature_layer, style_weight_mask_for_that_layer):
    return

def add_content_img_style_weight_mask_to_input(input, content_img_style_weight_mask):
    assert content_img_style_weight_mask is not None
    input_concatenated = tf.concat(3,(input, content_img_style_weight_mask))
    return input_concatenated


def spatial_batch_norm(input_layer, input_style_placeholder, name='spatial_batch_norm', reuse=False):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    with tf.variable_scope(name, reuse=reuse):
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
        # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
        variance = tf.abs(variance)
        variance_epsilon = 0.001
        num_channels = input_layer.get_shape().as_list()[3]
        scale = tf.get_variable('scale', [num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_channels], tf.float32, tf.constant_initializer())
        return_val = tf.nn.batch_normalization(input_layer, mean, variance, offset, scale, variance_epsilon, name=name)
        return return_val


def instance_norm(input_layer, name='instance_norm', reuse=False):
    """
    Instance-normalize the layer as in https://arxiv.org/abs/1607.08022
    """
    # calculate the mean and variance for width and height axises.
    with tf.variable_scope(name, reuse=reuse):
        input_layers = tf.unpack(input_layer)
        return_val = []
        num_channels = input_layer.get_shape().as_list()[3]
        # The scale and offset variable is reused for all batches in this norm.
        # NOTE: it is ok to use a different scale and offset for each batch. The meaning of doing so is not so clear but
        # it will still work. The resulting coloring of the image is different from the current implementation.
        scale = tf.get_variable('scale', [num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_channels], tf.float32, tf.constant_initializer())
        for l in input_layers:
            l = tf.expand_dims(l, 0)
            # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
            mean, variance = tf.nn.moments(l, [0, 1, 2])
            variance = tf.abs(variance)
            variance_epsilon = 0.001
            return_val.append(
                tf.squeeze(tf.nn.batch_normalization(l, mean, variance, offset, scale, variance_epsilon, name=name),
                           [0]))
        return_val = tf.pack(return_val)
        return return_val


def conditional_instance_norm(input_layer, input_style_placeholder, name='conditional_instance_norm', reuse=False):
    """
    Instance-normalize the layer conditioned on the style as in https://arxiv.org/abs/1610.07629
    input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different style
    images.
    """
    # calculate the mean and variance for width and height axises.
    with tf.variable_scope(name, reuse=reuse):
        input_layers = tf.unpack(input_layer)
        return_val = []
        num_styles = input_style_placeholder.get_shape().as_list()[1]
        num_channels = input_layer.get_shape().as_list()[3]
        scale = tf.get_variable('scale', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_styles, num_channels], tf.float32, tf.constant_initializer())
        scale_for_current_style = tf.matmul(input_style_placeholder, scale)
        offset_for_current_style = tf.matmul(input_style_placeholder, offset)
        for l in input_layers:
            l = tf.expand_dims(l, 0)
            # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
            mean, variance = tf.nn.moments(l, [0, 1, 2])
            variance = tf.abs(variance)
            variance_epsilon = 0.001
            return_val.append(tf.squeeze(tf.nn.batch_normalization(
                l, mean, variance, offset_for_current_style, scale_for_current_style, variance_epsilon, name=name),
                [0]))
        return_val = tf.pack(return_val)
        return return_val


def gramian(layer):
    # Takes (batches, height, width, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    _, height, width, number = map(lambda i: i.value, layer.get_shape())
    size = height * width * number
    layer_unpacked = tf.unpack(layer)
    grams = []
    for single_layer in layer_unpacked:
        feats = tf.reshape(single_layer, (-1, number))
        grams.append(tf.matmul(tf.transpose(feats),
                               feats) / size)  # TODO: find out the right normalization. This might be wrong.
    return tf.pack(grams)


def total_variation(image_batch):
    """
    :param image_batch: A 4D tensor of shape [batch_size, height, width, channels]
    """
    batch_shape = image_batch.get_shape().as_list()
    batch_size = batch_shape[0]
    height = batch_shape[1]
    left = tf.slice(image_batch, [0, 0, 0, 0], [-1, height - 1, -1, -1])
    right = tf.slice(image_batch, [0, 1, 0, 0], [-1, -1, -1, -1])

    width = batch_shape[2]
    top = tf.slice(image_batch, [0, 0, 0, 0], [-1, -1, width - 1, -1])
    bottom = tf.slice(image_batch, [0, 0, 1, 0], [-1, -1, -1, -1])

    # left and right are 1 less wide than the original, top and bottom 1 less tall
    # In order to combine them, we take 1 off the height of left-right, and 1 off width of top-bottom
    vertical_diff = tf.slice(tf.sub(left, right), [0, 0, 0, 0], [-1, -1, width - 1, -1])
    horizontal_diff = tf.slice(tf.sub(top, bottom), [0, 0, 0, 0], [-1, height - 1, -1, -1])

    vertical_diff_shape = vertical_diff.get_shape().as_list()
    num_pixels_in_vertical_diff = vertical_diff_shape[0] * vertical_diff_shape[1] * vertical_diff_shape[2] * \
                                  vertical_diff_shape[3]
    horizontal_diff_shape = horizontal_diff.get_shape().as_list()
    num_pixels_in_horizontal_diff = horizontal_diff_shape[0] * horizontal_diff_shape[1] * horizontal_diff_shape[2] * \
                                    horizontal_diff_shape[3]

    # Why there's a 2 here? I added it according to https://github.com/antlerros/tensorflow-fast-neuralstyle and
    # https://github.com/anishathalye/neural-style
    total_variation = 2 * (tf.nn.l2_loss(horizontal_diff) / num_pixels_in_horizontal_diff + tf.nn.l2_loss(
        vertical_diff) / num_pixels_in_vertical_diff) / batch_size

    return total_variation


def precompute_image_features(image_path, layers, shape, vgg_data, mean_pixel, use_mrf, use_semantic_masks):
    features_dict = {}
    g = tf.Graph()
    # If using gpu, uncomment the following line.
    # with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    with g.as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.pre_read_net(vgg_data, image)
        style_pre = np.array([vgg.preprocess(image_path, mean_pixel)])
        for layer in layers:
            if use_mrf or use_semantic_masks:
                features = net[layer].eval(feed_dict={image: style_pre})
                features_dict[layer] = features
            else:
                # Calculate and store gramian.
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                features_dict[layer] = gram
    return features_dict