# The code skeleton mainly comes from https://github.com/anishathalye/neural-style.
from sys import stderr

import numpy as np
import tensorflow as tf

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
            print_iterations=None, checkpoint_iterations=None, new_gram = False, semantic_masks_num_layers=4):
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

    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    output_semantic_mask_features = {}
    #
    #
    # # compute content features in feedforward mode
    # g = tf.Graph()
    # with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    #     image = tf.placeholder('float', shape=shape)
    #     net, mean_pixel = vgg.net(network, image)
    #     content_pre = np.array([vgg.preprocess(content, mean_pixel)])
    #     content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
    #             feed_dict={image: content_pre})
    #
    # # compute style features in feedforward mode
    # for i in range(len(styles)):
    #     g = tf.Graph()
    #     with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    #         image = tf.placeholder('float', shape=style_shapes[i])
    #         net, _ = vgg.net(network, image)
    #         style_pre = np.array([vgg.preprocess(styles[i], mean_pixel)])
    #         for layer in STYLE_LAYERS:
    #             features = net[layer].eval(feed_dict={image: style_pre})
    #             if use_mrf or use_semantic_masks:
    #                 style_features[i][layer] = features # Compute gram later if use semantic masks
    #             else:
    #                 features = np.reshape(features, (-1, features.shape[3]))
    #                 gram = np.matmul(features.T, features) / features.size
    #                 style_features[i][layer] = gram
    #
    # if use_semantic_masks:
    #     with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    #         image = tf.placeholder('float', shape=shape)
    #         net, mean_pixel = vgg.net(network, image)
    #         output_semantic_mask_pre = np.array([vgg.preprocess(output_semantic_mask, mean_pixel)])
    #         for layer in STYLE_LAYERS:
    #             # output_semantic_mask_features[layer] = net[layer].eval(
    #             #     feed_dict={image: output_semantic_mask_pre})
    #             # ***** TEST*****
    #             output_semantic_mask_feature = net[layer].eval(
    #                      feed_dict={image: output_semantic_mask_pre})
    #             features_tf_constant = tf.constant(output_semantic_mask_feature)
    #             output_semantic_mask_features[layer] = tf.get_variable('feature_masks_' + layer, initializer=features_tf_constant, trainable=True)
    #             # ***** END TEST*****
    #
    #
    #     # compute style features in feedforward mode
    #     for i in range(len(styles)):
    #         g = tf.Graph()
    #         with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    #             image = tf.placeholder('float', shape=style_shapes[i])
    #             net, _ = vgg.net(network, image)
    #             style_semantic_masks_pre = np.array([vgg.preprocess(style_semantic_masks[i], mean_pixel)])
    #             for layer in STYLE_LAYERS:
    #                 features = net[layer].eval(feed_dict={image: style_semantic_masks_pre})
    #                 # ***** TEST*****
    #                 features_tf_constant = tf.constant(features)
    #                 features = tf.get_variable('style_masks_' + layer, initializer=features_tf_constant, trainable=True)
    #                 # ***** END TEST*****
    #                 if use_mrf:
    #                     style_features[i][layer] = neural_doodle_util.concatenate_mask_layer_np(features, style_features[i][layer])
    #                 else:
    #                     features = neural_doodle_util.concatenate_mask_layer_np(features, style_features[i][layer])
    #                     features = np.reshape(features, (-1, features.shape[3]))
    #                     gram = np.matmul(features.T, features) / features.size
    #                     style_semantic_masks_features[i][layer] = gram

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
        for i in range(len(styles)):
            style_images.append(tf.placeholder('float', shape=style_shapes[i], name='style_image_%d' % i))
            net = vgg.pre_read_net(vgg_data, style_images[-1])
            # net, _ = vgg.net(network, style_images[-1])
            style_pres.append(np.array([vgg.preprocess(styles[i], mean_pixel)]))
            for layer in STYLE_LAYERS:
                features = net[layer]
                if use_mrf or use_semantic_masks:
                    style_features[i][layer] = features  # Compute gram later if use semantic masks
                else:

                    # ***** TEST GRAM*****
                    # TODO: Testing new loss function.
                    if new_gram:
                        gram = neural_util.gram_stacks(features, shift_size=SHIFT_SIZE)
                    else:
                        _, height, width, number = map(lambda i: i.value, features.get_shape())
                        size = height * width * number
                        features = tf.reshape(features, (-1, number))
                        gram = tf.matmul(tf.transpose(features), features) / size
                    style_features[i][layer] = gram
                    # ***** END TEST GRAM*****

        # if use_semantic_masks:
        #     output_semantic_mask_pre = np.array([vgg.preprocess(output_semantic_mask, mean_pixel)])
        #
        #     content_semantic_mask = tf.placeholder('float', shape=shape, name='content_mask')
        #     if mask_resize_as_feature:
        #
        #         for layer in STYLE_LAYERS:
        #             output_semantic_mask_feature = tf.image.resize_images(content_semantic_mask, (net_layer_sizes[layer][1], net_layer_sizes[layer][2])) \
        #                                            * semantic_masks_weight / 255.0
        #             output_semantic_mask_features[layer] = tf.get_variable('feature_masks_' + layer,
        #                                                                    initializer=output_semantic_mask_feature,
        #                                                                    trainable=False)
        #     else:
        #         net, _ = vgg.net(network, content_semantic_mask)
        #         for layer in STYLE_LAYERS:
        #             # output_semantic_mask_features[layer] = net[layer].eval(
        #             #     feed_dict={image: output_semantic_mask_pre})
        #             # ***** TEST*****
        #             output_semantic_mask_feature = net[layer] * semantic_masks_weight
        #             output_semantic_mask_features[layer] = tf.get_variable('feature_masks_' + layer,
        #                                                                    initializer=output_semantic_mask_feature,
        #                                                                    trainable=False)
        #             # ***** END TEST*****
        #
        #     # compute style features in feedforward mode
        #     style_semantic_masks_pres = []
        #     style_semantic_masks_images=[]
        #     for i in range(len(styles)):
        #         style_semantic_masks_images.append(tf.placeholder('float', shape=style_shapes[i], name='style_mask_%d' % i))
        #         style_semantic_masks_pres.append(np.array([vgg.preprocess(style_semantic_masks[i], mean_pixel)]))
        #
        #         if not mask_resize_as_feature:
        #             net, _ = vgg.net(network, style_semantic_masks_images[-1])
        #
        #         for layer in STYLE_LAYERS:
        #             if mask_resize_as_feature:
        #                 features = tf.image.resize_images(style_semantic_masks_images[-1], (net_layer_sizes[layer][1], net_layer_sizes[layer][2])) / 255.0
        #             else:
        #                 features = net[layer]
        #             features = features * semantic_masks_weight
        #             # ***** TEST*****
        #             features = tf.get_variable('style_masks_' + layer, initializer=features,
        #                                        trainable=False)
        #             # ***** END TEST*****
        #             if use_mrf:
        #                 # TODO: change it back, or make sure it is better than just concatenating.
        #                 # style_features[i][layer] = neural_doodle_util.concatenate_mask_layer_tf(features,
        #                 #                                                                         style_features[i][
        #                 #                                                                             layer])
        #
        #                 style_features[i][layer] =  neural_doodle_util.vgg_layer_dot_mask(features, style_features[i][layer])
        #             else:
        #                 # ***** TEST GRAM*****
        #                 # TODO: Testing new loss function
        #                 if new_gram:
        #                     features = neural_doodle_util.vgg_layer_dot_mask(features, style_features[i][layer])
        #                     gram = neural_util.gram_stacks(features, shift_size=SHIFT_SIZE)
        #                 else:
        #                     # TODO: change this to follows: each color has one corresponding gram matrix (now all shares one large one).
        #
        #                     gram = neural_doodle_util.gramian_with_mask(style_features[i][layer], features)
        #
        #                     # _, height, width, number = map(lambda i: i.value, features.get_shape())
        #                     # size = height * width * number
        #                     #
        #                     # features = tf.reshape(features, (-1, number))
        #                     # gram = tf.matmul(tf.transpose(features), features) / size
        #
        #                 style_semantic_masks_features[i][layer] = gram
        #                 # ***** END TEST GRAM*****

        if use_semantic_masks:

            content_semantic_mask = tf.placeholder(tf.float32, [shape[0], shape[1], shape[2],
                                                                semantic_masks_num_layers],
                                                   name='content_semantic_mask')
            if mask_resize_as_feature:
                # TODO: According to http://dmitryulyanov.github.io/feed-forward-neural-doodle/,
                # resizing might not be sufficient. "Use 3x3 mean filter for mask when the data goes through
                # convolutions and average pooling along with pooling layers."
                # But this is just a minor improvement that should not affect the final result too much.
                # prev_layer = None

                output_semantic_masks_for_each_layer = neural_doodle_util.masks_average_pool(content_semantic_mask)
                for layer in STYLE_LAYERS:
                    # output_semantic_mask_feature = tf.image.resize_images(content_semantic_mask, (
                    #     net_layer_sizes[layer][1], net_layer_sizes[layer][2]))
                    #


                    output_semantic_mask_feature = output_semantic_masks_for_each_layer[layer]

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
                for layer in STYLE_LAYERS:
                    output_semantic_mask_feature = semantic_mask_net[layer] * semantic_masks_weight
                    output_semantic_mask_features[layer] = output_semantic_mask_feature

            style_semantic_masks_pres = []
            style_semantic_masks_images = []
            style_semantic_masks_for_each_layer = []
            for i in range(len(styles)):
                style_semantic_masks_images.append(
                    tf.placeholder('float',
                                   shape=(1, style_shapes[i][1], style_shapes[i][2], semantic_masks_num_layers),
                                   name='style_mask_%d' % i))

                if not mask_resize_as_feature:
                    style_semantic_masks_pres.append(
                        np.array([vgg.preprocess(style_semantic_masks[i], mean_pixel)]))
                    semantic_mask_net, _ = vgg.pre_read_net(vgg_data, style_semantic_masks_pres[-1])
                else:
                    style_semantic_masks_for_each_layer.append(
                        neural_doodle_util.masks_average_pool(style_semantic_masks_images[-1]))

                for layer in STYLE_LAYERS:
                    if mask_resize_as_feature:
                        # Must be normalized (/ 255), otherwise the style loss just gets out of control.
                        # features = tf.image.resize_images(style_semantic_masks_images[-1],
                        #                                   (net_layer_sizes[layer][1], net_layer_sizes[layer][2])) / 255.0
                        features = style_semantic_masks_for_each_layer[-1][layer] / 255.0

                        features_shape = map(lambda i: i.value, features.get_shape())
                        if (net_layer_sizes[layer][1] != features_shape[1]) or (
                            net_layer_sizes[layer][1] != features_shape[1]):
                            print("Semantic masks shape not equal. Net layer %s size is: %s, semantic mask size is: %s"
                                  % (layer, str(net_layer_sizes[layer]), str(features_shape)))
                            raise AssertionError

                    else:
                        features = semantic_mask_net[layer]
                    features = features * semantic_masks_weight
                    if use_mrf:
                        style_features[i][layer] = \
                            neural_doodle_util.concatenate_mask_layer_tf(features, style_features[i][layer])
                    else:
                        # TODO :testing new gram with masks.
                        gram = neural_doodle_util.gramian_with_mask(style_features[i][layer], features)
                        #
                        # features = neural_doodle_util.vgg_layer_dot_mask(features, style_features[i][layer])
                        # # TODO: testing gram stacks
                        # gram = gramian(features)
                        # # If we want to use gram stacks instead of simple gram, uncomment the line below.
                        # # gram = neural_util.gram_stacks(features)
                        style_features[i][layer] = gram


        if initial is None:
            if content is None:
                noise = np.random.normal(size=shape, scale=0.1)
            else:
                noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
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

                if use_mrf:
                    if use_semantic_masks:
                        # TODO: change it back, or make sure it is better than just concatenating.
                        # layer = neural_doodle_util.concatenate_mask_layer_tf(output_semantic_mask_features[style_layer], layer)
                        layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer], layer)
                    style_losses.append(mrf_loss(style_features[i][style_layer], layer, name = '%d%s' % (i, style_layer)))
                else:


                    # ***** TEST GRAM*****
                    # TODO: Testing new loss function.
                    if new_gram:
                        if use_semantic_masks:
                            layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer],
                                                                          layer)
                        gram = neural_util.gram_stacks(layer, shift_size=SHIFT_SIZE)
                    else:
                        if use_semantic_masks:
                            gram = neural_doodle_util.gramian_with_mask(layer, output_semantic_mask_features[style_layer])
                        else:
                            _, height, width, number = map(lambda i: i.value, layer.get_shape())
                            size = height * width * number
                            feats = tf.reshape(layer, (-1, number))
                            gram = tf.matmul(tf.transpose(feats), feats) / size

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


                # feed_dict[content_semantic_mask]= output_semantic_mask_pre
                # for i in range(len(styles)):
                #     feed_dict[style_semantic_masks_images[i]] = style_semantic_masks_pres[i]
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
