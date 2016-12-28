# This file contains functions related to neural doodle. That is we feed in two additional semantic mask layers to
# tell the model which part of the object is what. Using mrf loss and nearest neighbor matching, this technique can
# essentially "draw" according to the mask layers provided.

import tensorflow as tf

import neural_util
import vgg
from feedforward_style_net_util import gramian
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


def gramian_with_mask(layer, masks, new_gram = False, shift_size = None):
    """TODO"""
    assert new_gram is False or shift_size is not None
    mask_list = tf.unpack(masks, axis=3) # A list of masks with dimension (1,height, width)

    gram_list = []

    for mask in mask_list:
        mask = tf.expand_dims(mask, dim=3)
        layer_dotted_with_mask = vgg_layer_dot_mask(mask, layer)
        if new_gram:
            layer_dotted_with_mask_gram = neural_util.gram_stacks(layer_dotted_with_mask, shift_size)
        else:
            layer_dotted_with_mask_gram = gramian(layer_dotted_with_mask)
        # Normalization is very importantant here. Because otherwise there is no way to compare two gram matrices
        # with different masks applied to them.
        layer_dotted_with_mask_gram_normalized = layer_dotted_with_mask_gram / (tf.reduce_mean(mask) + 0.000001) # Avoid division by zero.
        gram_list.append(layer_dotted_with_mask_gram_normalized)
    grams = tf.pack(gram_list)

    if isinstance(layer, np.ndarray):
        _, _, _, num_features = layer.shape
    else:
        _,_,_,num_features  =  map(lambda i: i.value, layer.get_shape())

    if new_gram:
        number_colors, _, gram_height, gram_width, new_gram_num_additional_features = map(lambda i: i.value, grams.get_shape())
    else:
        number_colors,_, gram_height, gram_width,  = map(lambda i: i.value, grams.get_shape())

    assert num_features == gram_height
    assert num_features == gram_width
    assert number_colors == len(mask_list)

    return grams


def construct_masks_and_features(style_semantic_masks, styles, style_features, batch_size, height, width, semantic_masks_num_layers, style_layer_names, net_layer_sizes, semantic_masks_weight, vgg_data, mean_pixel, mask_resize_as_feature, use_mrf, new_gram = False, shift_size = None, average_pool = False):
    # Variables to be returned.
    output_semantic_mask_features = {}

    content_semantic_mask = tf.placeholder(tf.float32, [batch_size, height, width,
                                                        semantic_masks_num_layers],
                                           name='content_semantic_mask')
    if mask_resize_as_feature:
        # TODO: According to http://dmitryulyanov.github.io/feed-forward-neural-doodle/,
        # resizing might not be sufficient. "Use 3x3 mean filter for mask when the data goes through
        # convolutions and average pooling along with pooling layers."
        # But this is just a minor improvement that should not affect the final result too much.
        # prev_layer = None
        if average_pool:
            output_semantic_masks_for_each_layer = masks_average_pool(content_semantic_mask)
        for layer in style_layer_names:
            if average_pool:
                output_semantic_mask_feature = output_semantic_masks_for_each_layer[layer]
            else:
                output_semantic_mask_feature = tf.image.resize_images(content_semantic_mask, (
                    net_layer_sizes[layer][1], net_layer_sizes[layer][2]))

            output_semantic_mask_shape = map(lambda i: i.value, output_semantic_mask_feature.get_shape())
            if (net_layer_sizes[layer][1] != output_semantic_mask_shape[1]) or (
                net_layer_sizes[layer][1] != output_semantic_mask_shape[1]):
                print("Semantic masks shape not equal. Net layer %s size is: %s, semantic mask size is: %s"
                      % (layer, str(net_layer_sizes[layer]), str(output_semantic_mask_shape)))
                raise AssertionError

            # Must be normalized (/ 255), otherwise the style loss just gets out of control.
            output_semantic_mask_features[layer] = output_semantic_mask_feature * semantic_masks_weight / 255.0
            # prev_layer = layer
    else:
        content_semantic_mask_pre = vgg.preprocess(content_semantic_mask, mean_pixel)
        semantic_mask_net, _ = vgg.pre_read_net(vgg_data, content_semantic_mask_pre)
        for layer in style_layer_names:
            output_semantic_mask_feature = semantic_mask_net[layer] * semantic_masks_weight
            output_semantic_mask_features[layer] = output_semantic_mask_feature

    style_semantic_masks_pres = []
    style_semantic_masks_images = []
    style_semantic_masks_for_each_layer = []
    for i in range(len(styles)):
        current_style_shape = styles[i].shape # Shape has format : height width rgb
        style_semantic_masks_images.append(
            tf.placeholder('float',
                           shape=(1, current_style_shape[0], current_style_shape[1], semantic_masks_num_layers),
                           name='style_mask_%d' % i))

        if not mask_resize_as_feature:
            style_semantic_masks_pres.append(
                np.array([vgg.preprocess(style_semantic_masks[i], mean_pixel)]))
            semantic_mask_net, _ = vgg.pre_read_net(vgg_data, style_semantic_masks_pres[-1])
        else:
            style_semantic_masks_for_each_layer.append(
                masks_average_pool(style_semantic_masks_images[-1]))

        for layer in style_layer_names:
            if mask_resize_as_feature:
                # Must be normalized (/ 255), otherwise the style loss just gets out of control.
                # features = tf.image.resize_images(style_semantic_masks_images[-1],
                #                                   (net_layer_sizes[layer][1], net_layer_sizes[layer][2])) / 255.0
                features = style_semantic_masks_for_each_layer[-1][layer] / 255.0

                # TODO: fix this. The shapes of content masks and style masks are different.
                # features_shape = map(lambda i: i.value, features.get_shape())
                # if (net_layer_sizes[layer][1] != features_shape[1]) or (net_layer_sizes[layer][1] != features_shape[1]):
                #     print("Semantic masks shape not equal. Net layer %s size is: %s, semantic mask size is: %s"
                #           % (layer, str(net_layer_sizes[layer]), str(features_shape)))
                #     raise AssertionError

            else:
                features = semantic_mask_net[layer]
            features = features * semantic_masks_weight
            if use_mrf:
                style_features[i][layer] = \
                    concatenate_mask_layer_tf(features, style_features[i][layer])
            else:
                # TODO :testing new gram with masks.
                style_feature_size = map(lambda i: i.value, style_features[i][layer].get_shape())
                gram = gramian_with_mask(style_features[i][layer], features, new_gram=new_gram, shift_size=shift_size)
                #
                # features = neural_doodle_util.vgg_layer_dot_mask(features, style_features[i][layer])
                # # TODO: testing gram stacks
                # gram = gramian(features)
                # # If we want to use gram stacks instead of simple gram, uncomment the line below.
                # # gram = neural_util.gram_stacks(features)
                style_features[i][layer] = gram

    return output_semantic_mask_features, style_features, content_semantic_mask, style_semantic_masks_images