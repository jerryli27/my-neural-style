"""
This file contains functions for experimental purposes. They probably won't have full documentations and you can safely
ignore them for most of the time... I hope.
"""
import tensorflow as tf

from mrf_util import create_local_patches
from neural_util import get_tensor_num_elements


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


def patch_matching_experiment(generated_layer_patches, style_layer_patches, patch_size, name='patch_matching'):
    """
    The patch matching (Equation 3) is implemented as an additional convolutional layer for fast computation.
    In this case patches sampled from the style image are treated as the filters.
    :param generated_layer_patches: Size (batch, height, width, patch_size * patch_size * feature)
    :param style_layer_patches:Size (1, height, width, patch_size * patch_size * feature)
    :param patch_size:Size (1, height, width, patch_size * patch_size * feature)
    :return: Best matching patch with size (batch, height, width, patch_size * patch_size * feature)
    """
    # TODO: nn is not differentiable...

    # Every patch and every feature layer are treated as equally important after normalization.
    normalized_generated_layer_patches = tf.nn.l2_normalize(generated_layer_patches, dim=[3])
    normalized_style_layer_patches = tf.nn.l2_normalize(style_layer_patches, dim=[3])

    # TODO: Now what happens if we train a variable that can learn which feature layer is more important?
    # This does not seem to work.
    # patch_feature_weights = tf.get_variable('patch_feature_weights_' + name, generated_layer_patches.get_shape().as_list()[3], tf.float32,tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32))
    # # patch_feature_weights_normalized = tf.nn.l2_normalize(patch_feature_weights, dim = [0])
    # normalized_generated_layer_patches = tf.mul(normalized_generated_layer_patches, patch_feature_weights)
    # normalized_style_layer_patches = tf.mul(normalized_style_layer_patches, patch_feature_weights)

    # A better way to do this is to treat them as convolutions.
    # They have to be in dimension
    # (height * width, patch_size, patch_size, feature) <=> (batch, in_height, in_width, in_channels)
    # (patch_size, patch_size, feature, height * width) <= > (filter_height, filter_width, in_channels, out_channels)
    # Initially they are in [batch, out_rows, out_cols, patch_size * patch_size * depth]
    original_shape = normalized_style_layer_patches.get_shape().as_list()
    height = original_shape[1]
    width = original_shape[2]
    depth = original_shape[3] / patch_size / patch_size
    normalized_style_layer_patches = tf.squeeze(normalized_style_layer_patches)

    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height, width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height * width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.transpose(normalized_style_layer_patches, perm=[1, 2, 3, 0])

    style_layer_patches_reshaped = tf.reshape(style_layer_patches, [height, width, patch_size, patch_size, depth])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches_reshaped,
                                              [height * width, patch_size, patch_size, depth])

    normalized_generated_layer_patches_per_batch = tf.unpack(normalized_generated_layer_patches, axis=0)
    ret = []
    nns = []
    for batch in normalized_generated_layer_patches_per_batch:
        original_shape = batch.get_shape().as_list()
        height = original_shape[0]
        width = original_shape[1]
        depth = original_shape[2] / patch_size / patch_size
        batch = tf.squeeze(batch)

        batch = tf.reshape(batch, [height, width, patch_size, patch_size, depth])
        batch = tf.reshape(batch, [height * width, patch_size, patch_size, depth])
        # According to images-analogies github, for cross-correlation, we should flip the kernels
        # That is normalized_style_layer_patches should be [:, ::-1, ::-1, :]
        # I didn't see that in any other source, nor do I see why I should do so.
        convs = tf.nn.conv2d(batch, normalized_style_layer_patches, strides=[1, 1, 1, 1], padding='VALID')
        argmax = tf.squeeze(tf.argmax(convs, dimension=3))
        # best_match has shape [height * width, patch_size, patch_size, depth]
        best_match = tf.gather(style_layer_patches_reshaped, indices=argmax)
        best_match = tf.reshape(best_match, [height, width, patch_size, patch_size, depth])
        best_match = tf.reshape(best_match, [height, width, patch_size * patch_size * depth])
        ret.append(best_match)
        argmax_reshaped = tf.cast(tf.reshape(argmax, [height, width]),tf.float32)
        nns.append(argmax_reshaped)
    ret = tf.pack(ret)
    nns = tf.pack(nns)
    return ret, nns


def mrf_loss_experiment(style_layer, generated_layer, patch_size=3, name='mrf_loss'):
    generated_layer_patches = create_local_patches(generated_layer, patch_size)
    style_layer_patches = create_local_patches(style_layer, patch_size)
    generated_layer_nn_matched_patches, nearest_neighbor_indices = patch_matching_experiment(generated_layer_patches, style_layer_patches, patch_size,
                                                        name=name)
    _, height, width, number = map(lambda i: i.value, generated_layer.get_shape())
    size = height * width * number
    # Normalize by the size of the image as well as the patch area.
    # loss = tf.reduce_sum(tf.square(tf.sub(generated_layer_patches, generated_layer_nn_matched_patches))) / size / (
    # patch_size ** 2)
    loss = 0
    nn_loss = nearest_neighbor_indices_loss(nearest_neighbor_indices)
    total_loss = nn_loss
    return total_loss


def nearest_neighbor_indices_loss(nn):
    """
    :param nn: A 2D tensor of shape [batch, height, width]
    """
    batch_shape = nn.get_shape().as_list()
    batch_size = batch_shape[0]
    height = batch_shape[1]
    width = batch_shape[2]


    left = tf.slice(nn, [0, 0, 0], [-1, height - 1, -1])
    right = tf.slice(nn, [0, 1, 0], [-1, -1, -1])

    left_x = tf.mod(left,height)
    right_x = tf.mod(right, height)
    left_y = tf.div(tf.sub(left, left_x), height)
    right_y = tf.div(tf.sub(right, right_x), height)

    top = tf.slice(nn, [0, 0, 0], [-1, -1, width - 1])
    bottom = tf.slice(nn, [0, 0, 1], [-1, -1, -1])

    top_x = tf.mod(top,height)
    bottom_x = tf.mod(bottom, height)
    top_y = tf.div(tf.sub(top, top_x), height)
    bottom_y = tf.div(tf.sub(bottom, bottom_x), height)

    # # left and right are 1 less wide than the original, top and bottom 1 less tall
    # # In order to combine them, we take 1 off the height of left-right, and 1 off width of top-bottom
    # vertical_x_diff = tf.slice(tf.sub(left_x, right_x), [0, 0, 0], [-1, -1, width - 1])
    # horizontal_x_diff = tf.slice(tf.sub(top_x, bottom_x), [0, 0, 0], [-1, height - 1, -1])
    #
    # vertical_y_diff = tf.slice(tf.sub(left_y, right_y), [0, 0, 0], [-1, -1, width - 1])
    # horizontal_y_diff = tf.slice(tf.sub(top_y, bottom_y), [0, 0, 0], [-1, height - 1, -1])
    #
    # vertical_diff_shape = vertical_x_diff.get_shape().as_list()
    # num_pixels_in_vertical_diff = vertical_diff_shape[0] * vertical_diff_shape[1] * vertical_diff_shape[2]
    # horizontal_diff_shape = horizontal_x_diff.get_shape().as_list()
    # num_pixels_in_horizontal_diff = horizontal_diff_shape[0] * horizontal_diff_shape[1] * horizontal_diff_shape[2]
    #
    # # Why there's a 2 here? I added it according to https://github.com/antlerros/tensorflow-fast-neuralstyle and
    # # https://github.com/anishathalye/neural-style
    #
    # # The target diff should be left x is smaller than right x by 1 and top y is smaller than bottom y by 1.
    # total_variation = 2 * (tf.nn.l2_loss(horizontal_x_diff) / num_pixels_in_horizontal_diff +
    #                        tf.nn.l2_loss(vertical_x_diff + 1)/ num_pixels_in_vertical_diff +
    #                        tf.nn.l2_loss(horizontal_y_diff + 1) / num_pixels_in_horizontal_diff +
    #                        tf.nn.l2_loss(vertical_y_diff)/ num_pixels_in_vertical_diff) / batch_size
    #
    #
    # return total_variation

    # left and right are 1 less wide than the original, top and bottom 1 less tall
    # In order to combine them, we take 1 off the height of left-right, and 1 off width of top-bottom
    horizontal_diff = tf.slice(tf.sub(left, right), [0, 0, 0], [-1, -1, width - 1])
    vertical_diff = tf.slice(tf.sub(top, bottom), [0, 0, 0], [-1, height - 1, -1])

    vertical_diff_shape = horizontal_diff.get_shape().as_list()
    num_pixels_in_vertical_diff = vertical_diff_shape[0] * vertical_diff_shape[1] * vertical_diff_shape[2]
    horizontal_diff_shape = vertical_diff.get_shape().as_list()
    num_pixels_in_horizontal_diff = horizontal_diff_shape[0] * horizontal_diff_shape[1] * horizontal_diff_shape[2]

    # Why there's a 2 here? I added it according to https://github.com/antlerros/tensorflow-fast-neuralstyle and
    # https://github.com/anishathalye/neural-style
    total_variation = 2 * (tf.nn.l2_loss(vertical_diff + height) / num_pixels_in_horizontal_diff + tf.nn.l2_loss(horizontal_diff + 1)/ num_pixels_in_vertical_diff) / batch_size


    return total_variation