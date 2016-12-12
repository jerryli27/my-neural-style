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

def masks_average_pool(masks):
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
    ret = {}
    current = masks
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            current = tf.contrib.layers.avg_pool2d(current, kernel_size=[3,3], stride=[1,1],padding='SAME')
        elif kind == 'relu':
            pass
        elif kind == 'pool':
            current = tf.contrib.layers.avg_pool2d(current, kernel_size=[2,2], stride=[2,2],padding='SAME')
        ret[name] = current

    assert len(ret) == len(layers)
    return ret