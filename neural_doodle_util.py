# This file contains functions related to neural doodle. That is we feed in two additional semantic mask layers to
# tell the model which part of the object is what. Using mrf loss and nearest neighbor matching, this technique can
# essentially "draw" according to the mask layers provided.

import tensorflow as tf

from general_util import *
import vgg

def read_mask(image_path):
    """
    :param image_path: the path of the mask image.
    :return: An np array containing the image.
    """
    return imread(image_path)

def generate_mask_layers(mask, layers, vgg_data, mean_pixel):
    """

    :param mask: The mask image represented in an np array.
    :param layers:
    :param vgg_data:
    :param mean_pixel:
    :return: A list of constant tensors, one for each layer.
    """
    ret = {}
    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=mask.shape)
        net = vgg.pre_read_net(vgg_data, image)
        preprocessed_list = np.array([vgg.preprocess(mask, mean_pixel)])
        for layer in layers:
            features = net[layer].eval(feed_dict={image: preprocessed_list})
            ret[layer] = features
    return ret

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

