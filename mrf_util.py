# This file contains utility functions to implement support for mrf.
# Please refer to https://arxiv.org/abs/1601.04589 for more details.

import tensorflow as tf


# TODO: I should make style_layer static to improve speed.
def mrf_loss(style_layer, generated_layer, patch_size=3, name='mrf_loss'):
    generated_layer_patches = create_local_patches(generated_layer, patch_size)
    style_layer_patches = create_local_patches(style_layer, patch_size)
    generated_layer_nn_matched_patches = patch_matching(generated_layer_patches, style_layer_patches, patch_size,
                                                        name=name)
    _, height, width, number = map(lambda i: i.value, generated_layer.get_shape())
    size = height * width * number
    # Normalize by the size of the image as well as the patch area.
    loss = tf.reduce_sum(tf.square(tf.sub(generated_layer_patches, generated_layer_nn_matched_patches))) / size / (
    patch_size ** 2)
    return loss


def create_local_patches(layer, patch_size, padding='VALID'):
    """

    :param layer: Feature layer tensor with dimension (1, height, width, feature)
    :param patch_size: The width and height of the patch. It is set to 3 in the paper https://arxiv.org/abs/1601.04589
    :return: Patches with dimension (cardinality, patch_size, patch_size, feature)
    """
    return tf.extract_image_patches(layer, ksizes=[1, patch_size, patch_size, 1],
                                    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding=padding)


def patch_matching(generated_layer_patches, style_layer_patches, patch_size, name='patch_matching'):
    """
    The patch matching (Equation 3) is implemented as an additional convolutional layer for fast computation.
    In this case patches sampled from the style image are treated as the filters.
    :param generated_layer_patches: Size (batch, height, width, patch_size * patch_size * feature)
    :param style_layer_patches:Size (1, height, width, patch_size * patch_size * feature)
    :param patch_size:Size (1, height, width, patch_size * patch_size * feature)
    :return: Best matching patch with size (batch, height, width, patch_size * patch_size * feature)
    """
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
    ret = tf.pack(ret)
    return ret
