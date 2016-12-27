# The code skeleton mainly comes from https://github.com/anishathalye/neural-style.
from sys import stderr

import numpy as np
import tensorflow as tf

import feedforward_style_net_util
import neural_doodle_util
import neural_util
import vgg
from mrf_util import mrf_loss

try:
    reduce
except NameError:
    from functools import reduce

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu3_1', 'relu4_1')
STYLE_LAYERS_WITH_CONTENT = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1')
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.
SHIFT_SIZE = 4 # The shift size for the new loss function.


def stylize(network, initial, content, styles, shape, iterations,
            content_weight, style_weight, style_blend_weights, tv_weight,
            learning_rate, use_mrf = False, use_semantic_masks = False, mask_resize_as_feature = True,
            output_semantic_mask = None, style_semantic_masks = None, semantic_masks_weight = 1.0,
            print_iterations=None, checkpoint_iterations=None, new_gram = False, semantic_masks_num_layers=4,
            content_img_style_weight_mask = None):
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
            print("The shape of style_weight_mask is incorrect. It must have the same height and width as the "
                  "output image. The output image has shape: %s and the style weight mask has shape: %s"
                  % (str(shape), str(content_img_style_weight_mask.shape)))
            raise AssertionError
        if content_img_style_weight_mask.dtype!=np.float32:
            print('The dtype of style_weight_mask must be float32. it is now %s' % str(content_img_style_weight_mask.dtype))
            raise AssertionError


    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    output_semantic_mask_features = {}

    # make stylized image using backpropogation
    with tf.Graph().as_default():

        vgg_data, mean_pixel = vgg.read_net(network)

        # compute content features in feedforward mode
        content_image = tf.placeholder('float', shape=shape, name='content_image')
        net = vgg.pre_read_net(vgg_data, content_image)
        # net, mean_pixel = vgg.net(network, content_image)
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER]
        net_layer_sizes = vgg.get_net_layer_sizes(net)

        if content is not None:
            content_pre = np.array([vgg.preprocess(content, mean_pixel)])



        # compute style features in feedforward mode
        style_images = []
        style_pres = []
        if content_img_style_weight_mask is not None:
            # *** TESTING
            # Compute style weight masks for each feature layer in vgg.
            style_weight_mask_layer_dict = neural_doodle_util.masks_average_pool(content_img_style_weight_mask)
            # *** END TESTING

        for i in range(len(styles)):
            style_images.append(tf.placeholder('float', shape=style_shapes[i], name='style_image_%d' % i))
            print(style_shapes[i])
            net = vgg.pre_read_net(vgg_data, style_images[-1])
            # net, _ = vgg.net(network, style_images[-1])
            style_pres.append(np.array([vgg.preprocess(styles[i], mean_pixel)]))
            for layer in STYLE_LAYERS:
                features = net[layer]
                # # *** TESTING
                # # apply_style_weight_mask_to_feature_layer. But we don't need to do this to style image.
                # features = neural_doodle_util.vgg_layer_dot_mask(style_weight_mask_layer_dict[layer], features)
                # # *** END TESTING
                if use_mrf or use_semantic_masks:
                    style_features[i][layer] = features  # Compute gram later if use semantic masks
                else:

                    # ***** TEST GRAM*****
                    # TODO: Testing new loss function.
                    if new_gram:
                        gram = neural_util.gram_stacks(features, shift_size=SHIFT_SIZE)
                    else:
                        gram = feedforward_style_net_util.gramian(features)
                        # _, height, width, number = map(lambda i: i.value, features.get_shape())
                        # size = height * width * number
                        # features = tf.reshape(features, (-1, number))
                        # gram = tf.matmul(tf.transpose(features), features) / size
                    style_features[i][layer] = gram
                    # ***** END TEST GRAM*****
        if use_semantic_masks:
            output_semantic_mask_features, style_features, content_semantic_mask, style_semantic_masks_images = neural_doodle_util.construct_masks_and_features(
                style_semantic_masks, styles, style_features, shape[0], shape[1], shape[2], semantic_masks_num_layers,
                STYLE_LAYERS, net_layer_sizes, semantic_masks_weight, vgg_data, mean_pixel, mask_resize_as_feature, use_mrf, new_gram=new_gram, shift_size=SHIFT_SIZE)

        if initial is None:
            # if content is None:
            #     noise = np.random.normal(size=shape, scale=0.1)
            # else:
            #     noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, mean_pixel)])
            initial = initial.astype('float32')
        image = tf.Variable(initial)
        net, _ = vgg.net(network, image)

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


                # *** TESTING
                if content_img_style_weight_mask is not None:
                    # Apply_style_weight_mask_to_feature_layer, then normalize with average of that style weight mask.
                    layer = neural_doodle_util.vgg_layer_dot_mask(style_weight_mask_layer_dict[style_layer], layer) \
                            / (tf.reduce_mean(style_weight_mask_layer_dict[style_layer]) + 0.000001)
                # *** END TESTING


                if use_mrf:
                    if use_semantic_masks:
                        # TODO: change it back, or make sure it is better than just concatenating.
                        # layer = neural_doodle_util.concatenate_mask_layer_tf(output_semantic_mask_features[style_layer], layer)
                        layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer], layer)
                    style_losses.append(mrf_loss(style_features[i][style_layer], layer, name = '%d%s' % (i, style_layer)))
                else:


                    # ***** TEST GRAM*****
                    # TODO: Testing new loss function.

                    if use_semantic_masks:
                        gram = neural_doodle_util.gramian_with_mask(layer, output_semantic_mask_features[style_layer], new_gram=new_gram, shift_size=SHIFT_SIZE)
                    else:
                        if new_gram:
                            gram = neural_util.gram_stacks(layer, shift_size=SHIFT_SIZE)
                        else:
                            gram = feedforward_style_net_util.gramian(layer)
                        # _, height, width, number = map(lambda i: i.value, layer.get_shape())
                        # size = height * width * number
                        # feats = tf.reshape(layer, (-1, number))
                        # gram = tf.matmul(tf.transpose(feats), feats) / size

                    style_gram = style_features[i][style_layer]

                    # ***** END TEST GRAM*****
                    if new_gram:
                        style_gram_size = neural_util.get_tensor_num_elements(style_gram) / (SHIFT_SIZE ** 2) # 2 is the shift size, 3 squared is the number of gram matrices we have.
                    else:
                        style_gram_size = neural_util.get_tensor_num_elements(style_gram)
                    style_losses.append(tf.nn.l2_loss(gram - style_gram) / style_gram_size) # TODO: Check normalization constants. the style loss is way too big compared to the other two
                    # style_losses.append(tf.nn.l2_loss(gram - style_gram))
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        # TODO: don't forget to change it back.
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
        with tf.Session() as sess:
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
