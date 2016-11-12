import tensorflow as tf
import scipy.misc
from sys import stderr

from general_util import *

# TODO: Needs reformatting.

def input_pyramid(name, height, width, batch_size, k=5, with_content_image=False):
    """
    Generates k inputs at different scales, with height x width being the largest.
    If with_content_image is true, add 3 to the last rgb dim because we are appending the resized content image to the
    noise generated.
    """
    if height % (2 ** (k-1)) != 0 or width % (2 ** (k-1)) != 0:
        stderr('Warning: Input width or height cannot be divided by 2^(k-1). This might cause problems when generating '
               'images.')
    with tf.get_default_graph().name_scope(name):
        return_val = [tf.placeholder(tf.float32, [batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)),
                                                  6 if with_content_image else 3], name=str(x)) for x in range(k)]
        return_val.reverse()
    return return_val


def noise_pyramid(height, width, batch_size, k=5, ablation_layer = None):
    """
    :param height: Height of the largest noise image.
    :param width: Width of the largest noise image.
    :param batch_size: Number of images we generate per batch.
    :param k: Number of inputs generated at different scales. If k = 3 for example, it will generate images with
    h x w, h/2 x w/2, h/4 x w/4.
    :param ablation_layer: If not none, every layer except for this one will be zeros. 0 <= ablation_layer < k-1.
    :return: A list of numpy arrays with size (batch_size, h/2^?, w/2^?, 3)
    """
    if height % (2 ** (k-1)) != 0 or width % (2 ** (k-1)) != 0:
        stderr('Warning: Input width or height cannot be divided by 2^(k-1). This might cause problems when generating '
               'images.')
    # return [np.random.rand(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3)
    #         if (ablation_layer is None or ablation_layer < 0 or (k-1-ablation_layer) != x) else
    #         np.zeros((batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3), dtype=np.float64) + 0.5
    #         for x in range(k)][::-1]

    return [np.random.rand(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3)
            if (ablation_layer is None or ablation_layer < 0 or (k - 1 - ablation_layer) == x) else
            np.random.rand(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3) * 0.0
            for x in range(k)][::-1]

# TODO: went till here. Continue from here.
def noise_pyramid_w_content_img(height, width, batch_size, content_image_pyramid, k=5, ablation_layer = None):
    """
    :param height: Height of the largest noise image.
    :param width: Width of the largest noise image.
    :param batch_size: Number of images we generate per batch.
    :param k: Number of inputs generated at different scales. If k = 3 for example, it will generate images with
    h x w, h/2 x w/2, h/4 x w/4.
    :param ablation_layer: If not none, every layer except for this one will be zeros. 0 <= ablation_layer < k-1.
    :return: A list of numpy arrays with size (batch_size, h/2^?, w/2^?, 3)
    """
    """If an additional input tensor, the content image tensor, is
    provided, then we concatenate the downgraded versions of that tensor to the noise tensors."""
    return [np.concatenate((np.random.rand(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3)
            if (ablation_layer is None or ablation_layer < 0 or (k - 1 - ablation_layer) == x) else
            np.random.rand(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3) * 0.0,
                            content_image_pyramid[x]), axis=3) for x in range(k)][::-1]


def generate_image_pyramid(height, width, batch_size, content_image, k=5):
    return [np.array([scipy.misc.imresize(content_image[0],
                                          (max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3))
                      for batch in range(batch_size)]) for x in range(k)]

# TODO: add support for reuse.
def spatial_batch_norm(input_layer, input_style_placeholder, name='spatial_batch_norm', reuse = False):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    # NOTE: the variance calculated may be a small negative value due to numeric imprecision.
    # The variance epsilon should be larger than 0.001 to overcome this issue.
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance = tf.abs(variance)
    variance_epsilon = 0.001
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3]
    scale = tf.get_variable('scale', [num_channels], tf.float32, tf.random_uniform_initializer())
    offset = tf.get_variable('offset', [num_channels], tf.float32, tf.random_uniform_initializer())
    # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
    # return return_val
    return_val = tf.nn.batch_normalization(input_layer, mean, variance, offset, scale, variance_epsilon, name=name)
    return return_val


# TODO: add support for reuse.
def instance_norm(input_layer, name='instance_norm', reuse = False):
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
        # NOTE: the variance calculated may be a small negative value due to numeric imprecision.
        # The variance epsilon should be larger than 0.001 to overcome this issue.
        mean, variance = tf.nn.moments(l, [0, 1, 2])
        variance = tf.abs(variance)
        variance_epsilon = 0.001
        inv = tf.rsqrt(variance + variance_epsilon)
        # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
        # return return_val
        return_val.append(
            tf.squeeze(tf.nn.batch_normalization(l, mean, variance, offset, scale, variance_epsilon, name=name), [0]))
    return_val = tf.pack(return_val)
    return return_val


def conditional_instance_norm(input_layer, input_style_placeholder, name='conditional_instance_norm', reuse = False):
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
        # scale = tf.Variable(tf.random_uniform([num_styles, num_channels]), name='scale')
        # offset = tf.Variable(tf.random_uniform([num_styles, num_channels]), name='offset')
        scale = tf.get_variable('scale', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        scale_for_current_style = tf.matmul(input_style_placeholder, scale)
        offset_for_current_style = tf.matmul(input_style_placeholder, offset)
        for l in input_layers:
            # # A potential problem with doing so: the scale and offset variable is different for every batch.
            # return_val.append(tf.squeeze(spatial_batch_norm(tf.expand_dims(l, 0)), [0]))
            l = tf.expand_dims(l, 0)
            # NOTE: the variance calculated may be a small negative value due to numeric imprecision.
            # The variance epsilon should be larger than 0.001 to overcome this issue.
            mean, variance = tf.nn.moments(l, [0, 1, 2])
            variance = tf.abs(variance)
            variance_epsilon = 0.001
            inv = tf.rsqrt(variance + variance_epsilon)
            # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
            # return return_val
            return_val.append(tf.squeeze(tf.nn.batch_normalization(
                l, mean, variance, offset_for_current_style, scale_for_current_style, variance_epsilon, name=name), [0]))
        return_val = tf.pack(return_val)
        return return_val


def gramian(layer):
    # Takes (batches, height, width, channels) and computes gramians of dimension (batches, channels, channels)
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
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/block_low_%d/layer%d/conditional_instance_norm/scale' % (d, layer))
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/block_high_%d/layer%d/conditional_instance_norm/scale' % (d, layer))

    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer1/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer2/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output_chain/layer3/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES, scope='texture_nets/output/conditional_instance_norm/scale')
    return scale_offset_variables

def total_variation(image_batch, divide_by_num_pixels = False):
    """
    :param image_batch: A 4D tensor of shape [batch_size, height, width, channels]
    """
    batch_shape = image_batch.get_shape().as_list()
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
    num_pixels_in_vertical_diff = vertical_diff_shape[0] * vertical_diff_shape[1] * vertical_diff_shape[2] * vertical_diff_shape[3]
    horizontal_diff_shape = vertical_diff.get_shape().as_list()
    num_pixels_in_horizontal_diff = horizontal_diff_shape[0] * horizontal_diff_shape[1] * horizontal_diff_shape[2] * horizontal_diff_shape[3]

    # sum_of_pixel_diffs_squared = tf.add(tf.square(horizontal_diff), tf.square(vertical_diff))
    # total_variation = tf.reduce_sum(tf.sqrt(sum_of_pixel_diffs_squared))
    if divide_by_num_pixels:
        total_variation = tf.sqrt(tf.reduce_sum(tf.square(horizontal_diff))) / num_pixels_in_horizontal_diff + tf.sqrt(tf.reduce_sum(tf.square(vertical_diff))) / num_pixels_in_vertical_diff
    else:
        total_variation = tf.sqrt(tf.reduce_sum(tf.square(horizontal_diff))) + tf.sqrt(tf.reduce_sum(tf.square(vertical_diff)))

    return total_variation

