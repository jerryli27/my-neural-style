"""
This file contains one function that implemented the papers:
"A Neural Algorithm of Artistic Style" (https://arxiv.org/abs/1508.06576),
"Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis" (arxiv.org/abs/1601.04589),
"Instance Normalization - The Missing Ingredient for Fast Stylization" (https://arxiv.org/abs/1607.08022),
"Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks" (https://arxiv.org/abs/1603.01768).

In addition, it contains one more functionality to control the degree of stylization of the content image by using a
weighted mask for the content image ("content_img_style_weight_mask" in the code)
The code skeleton was borrowed from https://github.com/anishathalye/neural-style.
"""

from sys import stderr

import numpy as np
import tensorflow as tf

import experimental_util
import neural_doodle_util
import neural_util
import vgg
from mrf_util import mrf_loss

try:
    reduce
except NameError:
    from functools import reduce

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')  # This is used for texture generation (without content)
STYLE_LAYERS_WITH_CONTENT = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.


def stylize(network, initial, content, styles, shape, iterations,
            content_weight, style_weight, style_blend_weights, tv_weight,
            learning_rate, use_mrf = False, use_semantic_masks = False, mask_resize_as_feature = True,
            output_semantic_mask = None, style_semantic_masks = None, semantic_masks_weight = 1.0,
            print_iterations=None, checkpoint_iterations=None, new_gram = False, new_gram_shift_size = 4,
            new_gram_stride = 1, semantic_masks_num_layers=4,
            content_img_style_weight_mask = None, feature_size = 1):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    global STYLE_LAYERS
    if content is not None:
        STYLE_LAYERS = STYLE_LAYERS_WITH_CONTENT
    if use_mrf:
        STYLE_LAYERS = STYLE_LAYERS_MRF  # Easiest way to be compatible with no-mrf versions.
    if use_semantic_masks:
        assert semantic_masks_weight is not None
        assert output_semantic_mask is not None
        assert style_semantic_masks is not None
    if content_img_style_weight_mask is not None:
        if shape[1] != content_img_style_weight_mask.shape[1] or shape[2] != content_img_style_weight_mask.shape[2]:
            raise AssertionError("The shape of style_weight_mask is incorrect. It must have the same height and width "
                                 "as the output image. The output image has shape: %s and the style weight mask has "
                                 "shape: %s" % (str(shape), str(content_img_style_weight_mask.shape)))
        if content_img_style_weight_mask.dtype!=np.float32:
            raise AssertionError('The dtype of style_weight_mask must be float32. it is now %s' % str(content_img_style_weight_mask.dtype))
    assert isinstance(feature_size, int) and feature_size >= 1

    # Note: the "feature size' option is not so well developed yet. I tried to use it to enlarge the features.
    # The following is for preventing the usage of experimental feature "feature_size". It is not doing what I want.
    assert feature_size == 1

    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width, 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    output_semantic_mask_features = {}

    # Make stylized image using back-propogation.
    with tf.Graph().as_default():

        vgg_data, mean_pixel = vgg.read_net(network)

        # Compute content features in feed-forward mode
        content_image = tf.placeholder('float', shape=shape, name='content_image')
        net = vgg.pre_read_net(vgg_data, content_image, stride_multiplier=feature_size)
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER]
        net_layer_sizes = vgg.get_net_layer_sizes(net)

        if content is not None:
            content_pre = np.array([vgg.preprocess(content, mean_pixel)])

        # Compute style features in feed-forward mode.
        style_images = []
        style_pres = []
        if content_img_style_weight_mask is not None:
            style_weight_mask_layer_dict = neural_doodle_util.masks_average_pool(content_img_style_weight_mask)

        for i in range(len(styles)):
            style_images.append(tf.placeholder('float', shape=style_shapes[i], name='style_image_%d' % i))
            print(style_shapes[i])
            net = vgg.pre_read_net(vgg_data, style_images[-1], stride_multiplier=feature_size)
            style_pres.append(np.array([vgg.preprocess(styles[i], mean_pixel)]))
            for layer in STYLE_LAYERS:
                features = net[layer]
                if use_mrf or use_semantic_masks:
                    style_features[i][layer] = features  # Compute gram later if use semantic masks
                else:
                    if new_gram:
                        gram = experimental_util.gram_stacks(features, shift_size=new_gram_shift_size, stride=new_gram_stride)
                    else:
                        gram = neural_util.gramian(features)
                    style_features[i][layer] = gram
        if use_semantic_masks:
            output_semantic_mask_features, style_features, content_semantic_mask, style_semantic_masks_images = neural_doodle_util.construct_masks_and_features(
                style_semantic_masks, styles, style_features, shape[0], shape[1], shape[2], semantic_masks_num_layers,
                STYLE_LAYERS, net_layer_sizes, semantic_masks_weight, vgg_data, mean_pixel, mask_resize_as_feature, use_mrf, new_gram=new_gram, shift_size=new_gram_shift_size, stride=new_gram_stride, average_pool=False) # TODO: average pool is not working so well in practice??

        if initial is None:
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, _ = vgg.net(network, image, stride_multiplier=feature_size)

        # content loss
        _, height, width, number = map(lambda i: i.value, content_features[CONTENT_LAYER].get_shape())
        content_features_size = height * width * number
        content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                         content_features_size)
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                if content_img_style_weight_mask is not None:
                    # Apply_style_weight_mask_to_feature_layer, then normalize with average of that style weight mask.
                    layer = neural_doodle_util.vgg_layer_dot_mask(style_weight_mask_layer_dict[style_layer], layer) \
                            / (tf.reduce_mean(style_weight_mask_layer_dict[style_layer]) + 0.000001)

                if use_mrf:
                    if use_semantic_masks:
                        # TODO: change it back, or make sure it is better than just concatenating. Also if you change this to dot, don't forget to also change that in neural_doodle_util.
                        layer = neural_doodle_util.concatenate_mask_layer_tf(output_semantic_mask_features[style_layer], layer)
                        # layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer], layer)
                    style_losses.append(mrf_loss(style_features[i][style_layer], layer, name = '%d%s' % (i, style_layer)))
                else:
                    if use_semantic_masks:
                        gram = neural_doodle_util.gramian_with_mask(layer, output_semantic_mask_features[style_layer], new_gram=new_gram, shift_size=new_gram_shift_size, stride=new_gram_stride)
                    else:
                        if new_gram:
                            gram = experimental_util.gram_stacks(layer, shift_size=new_gram_shift_size, stride=new_gram_stride)
                        else:
                            gram = neural_util.gramian(layer)
                    style_gram = style_features[i][style_layer]

                    if new_gram:
                        style_gram_size = neural_util.get_tensor_num_elements(style_gram) / (new_gram_shift_size ** 2) # 2 is the shift size, 3 squared is the number of gram matrices we have.
                    else:
                        style_gram_size = neural_util.get_tensor_num_elements(style_gram)
                    style_losses.append(tf.nn.l2_loss(gram - style_gram) / style_gram_size) # TODO: Check normalization constants. the style loss is way too big compared to the other two.
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        # total variation denoising
        tv_loss = tf.mul(neural_util.total_variation(image), tv_weight)

        # tv_y_size = _tensor_size(image[:,1:,:,:])
        # tv_x_size = _tensor_size(image[:,:,1:,:])
        # tv_loss = tv_weight * 2 * (
        #         (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
        #             tv_y_size) +
        #         (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
        #             tv_x_size))

        # overall loss
        if content is None: # If we are doing style/texture regeration only.
            loss = style_loss + tv_loss
        else:
            loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        def print_progress(i, feed_dict, last=False):
            stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
            if last or (print_iterations and i % print_iterations == 0):
                if content is not None:
                    stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                stderr.write('    style loss: %g\n' % style_loss.eval(feed_dict=feed_dict))
                stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                stderr.write('    total loss: %g\n' % loss.eval(feed_dict=feed_dict))

        # optimization
        best_loss = float('inf')
        best = None

        # TODO: TESTING
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            feed_dict = {}
            if content is not None:
                feed_dict[content_image] = content_pre
            for i in range(len(styles)):
                feed_dict[style_images[i]] = style_pres[i]
            if use_semantic_masks:
                feed_dict[content_semantic_mask] = output_semantic_mask
                for styles_iter in range(len(styles)):
                    feed_dict[style_semantic_masks_images[styles_iter]] = style_semantic_masks[styles_iter]
            sess.run(tf.initialize_all_variables(), feed_dict=feed_dict)
            for i in range(iterations):
                last_step = (i == iterations - 1)
                print_progress(i, feed_dict, last=last_step)
                train_step.run(feed_dict=feed_dict)

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval(feed_dict=feed_dict)
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()
                    yield (
                        (None if last_step else i),
                        vgg.unprocess(best.reshape(shape[1:]), mean_pixel)
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)
