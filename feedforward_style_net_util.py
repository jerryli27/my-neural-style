import tensorflow as tf
import scipy.misc

from general_util import *

def input_pyramid(name, M, batch_size, k=5, with_content_image=False):
    """
    Generates k inputs at different scales, with MxM being the largest.
    If with_content_image is true, add 3 to the last rgb dim because we are appending the content image to the noise
    generated.
    """
    with tf.get_default_graph().name_scope(name):
        return_val = [
            tf.placeholder(tf.float32, [batch_size, M // (2 ** x), M // (2 ** x), 6 if with_content_image else 3],
                           name=str(x)) for x in range(k)]
        return_val.reverse()
    return return_val


def noise_pyramid(M, batch_size, k=5):
    return [np.random.rand(batch_size, M // (2 ** x), M // (2 ** x), 3) for x in range(k)][::-1]


def noise_pyramid_w_content_img(M, batch_size, content_image_pyramid, k=5):
    """If an additional input tensor, the content image tensor, is
    provided, then we concatenate the downgraded versions of that tensor to the noise tensors."""
    return [np.concatenate((np.random.rand(batch_size, M // (2 ** x), M // (2 ** x), 3),
                            content_image_pyramid[x]), axis=3) for x in range(k)][::-1]


def content_img_pyramid(M, batch_size, content_image, k=5):
    return [np.array(
        [scipy.misc.imresize(content_image[0], (M // (2 ** x), M // (2 ** x), 3)) for batch in range(batch_size)]) for x
            in range(k)]

def spatial_batch_norm(input_layer, name='spatial_batch_norm'):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance_epsilon = 0.00001
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3]
    scale = tf.Variable(tf.random_uniform([num_channels]), name='scale')
    offset = tf.Variable(tf.random_uniform([num_channels]), name='offset')
    # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
    # return return_val
    return_val = tf.nn.batch_normalization(input_layer, mean, variance, offset, scale, variance_epsilon, name=name)
    return return_val


def instance_norm(input_layer, name='spatial_batch_norm'):
    """
    Instance-normalize the layer as in https://arxiv.org/abs/1607.08022
    """
    # calculate the mean and variance for width and height axises.
    input_layers = tf.unpack(input_layer)
    return_val = []
    num_channels = input_layer.get_shape().as_list()[3]
    # The scale and offset variable is reused for all batches in this norm.
    # NOTE: it is ok to use a different scale and offset for each batch. The meaning of doing so is not so clear but
    # it will still work. The resulting coloring of the image is different from the current implementation.
    scale = tf.Variable(tf.random_uniform([num_channels]), name='scale')
    offset = tf.Variable(tf.random_uniform([num_channels]), name='offset')
    for l in input_layers:
        # # A potential problem with doing so: the scale and offset variable is different for every batch.
        # return_val.append(tf.squeeze(spatial_batch_norm(tf.expand_dims(l, 0)), [0]))
        l = tf.expand_dims(l, 0)
        mean, variance = tf.nn.moments(l, [0, 1, 2])
        variance_epsilon = 0.00001
        inv = tf.rsqrt(variance + variance_epsilon)
        # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
        # return return_val
        return_val.append(
            tf.squeeze(tf.nn.batch_normalization(l, mean, variance, offset, scale, variance_epsilon, name=name), [0]))
    return_val = tf.pack(return_val)
    return return_val


def conditional_instance_norm(input_layer, input_style_placeholder, name='spatial_batch_norm'):
    """
    Instance-normalize the layer conditioned on the style as in https://arxiv.org/abs/1610.07629
    input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different style
    images.
    """
    # calculate the mean and variance for width and height axises.
    input_layers = tf.unpack(input_layer)
    return_val = []
    num_styles = input_style_placeholder.get_shape().as_list()[1]
    num_channels = input_layer.get_shape().as_list()[3]
    scale = tf.Variable(tf.random_uniform([num_styles, num_channels]), name='scale')
    offset = tf.Variable(tf.random_uniform([num_styles, num_channels]), name='offset')
    for l in input_layers:
        # # A potential problem with doing so: the scale and offset variable is different for every batch.
        # return_val.append(tf.squeeze(spatial_batch_norm(tf.expand_dims(l, 0)), [0]))
        l = tf.expand_dims(l, 0)
        mean, variance = tf.nn.moments(l, [0, 1, 2])
        variance_epsilon = 0.00001
        inv = tf.rsqrt(variance + variance_epsilon)
        # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
        # return return_val
        scale_for_current_style = tf.matmul(input_style_placeholder, scale)
        offset_for_current_style = tf.matmul(input_style_placeholder, offset)
        return_val.append(tf.squeeze(tf.nn.batch_normalization(
            l, mean, variance, offset_for_current_style, scale_for_current_style, variance_epsilon, name=name), [0]))
    return_val = tf.pack(return_val)
    return return_val


def gramian(layer):
    # Takes (batches, width, height, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    # vectorized_activations = tf.reshape(tf.transpose(activations, perm=[0, 3, 1, 2]),
    #                                     [activations_shape[0], activations_shape[3], -1])
    # transposed_vectorized_activations = tf.transpose(vectorized_activations, perm=[0, 2, 1])
    # mult = tf.batch_matmul(vectorized_activations, transposed_vectorized_activations)
    # return mult

    # The old way of calculating gramian only works for batch = 1
    _, height, width, number = map(lambda i: i.value, layer.get_shape())
    size = height * width * number
    layer_unpacked = tf.unpack(layer)
    grams = []
    for single_layer in layer_unpacked:
        feats = tf.reshape(single_layer, (-1, number))
        grams.append(tf.matmul(tf.transpose(feats), feats) / size)
    return tf.pack(grams)

def get_scale_offset_var():
    scale_offset_variables = []
    for d in range(8,48,8):
        for layer in range(1,4):
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/block_low_%d/layer%d/scale' % (d, layer))
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/block_high_%d/layer%d/scale' % (d, layer))

    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer1/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer2/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer3/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output/scale')
    return scale_offset_variables