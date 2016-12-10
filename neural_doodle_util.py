# This file contains functions related to neural doodle. That is we feed in two additional semantic mask layers to
# tell the model which part of the object is what. Using mrf loss and nearest neighbor matching, this technique can
# essentially "draw" according to the mask layers provided.

import tensorflow as tf

from general_util import *

def concatenate_mask_layer_tf(mask_layer, original_layer):
    """

    :param mask_layer: One layer of mask
    :param original_layer: The original layer before concatenating the mask layer to it.
    :return: A layer with the mask layer concatenated to the end.
    """
    return tf.concat(3, [mask_layer, original_layer])

def concatenate_mask_layer_np(mask_layer, original_layer):
    """

    :param mask_layer: One layer of mask
    :param original_layer: The original layer before concatenating the mask layer to it.
    :return: A layer with the mask layer concatenated to the end.
    """
    return np.concatenate((mask_layer, original_layer), axis=3)

def concatenate_mask(mask, original, layers):
    ret = {}
    for layer in layers:
        ret[layer] = concatenate_mask_layer_tf(mask[layer], original[layer])
    return ret

def vgg_layer_dot_mask(masks, vgg_layer):
    masks_dim_expanded = tf.expand_dims(masks, 4)
    vgg_layer_dim_expanded = tf.expand_dims(vgg_layer, 3)
    dot = tf.mul(masks_dim_expanded, vgg_layer_dim_expanded)

    batch_size, height, width, num_mask, num_features = map(lambda i: i.value, dot.get_shape())
    dot = tf.reshape(dot, [batch_size, height, width, num_mask * num_features])
    return dot