"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

import tensorflow as tf
import numpy as np
from sys import stderr
import scipy.misc
import gtk.gdk
from tensorflow.python.ops import math_ops

import neural_util
import vgg
from general_util import *
from feedforward_style_net_util import *
from mrf_util import mrf_loss
import misc_util

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.

# TODO: Needs reformatting.
def style_synthesis_net(path_to_network, contents, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, style_only = False, style_as_content = False,
                        multiple_styles_train_scale_offset_only = False, use_mrf = False,
                        print_iterations=None,
                        checkpoint_iterations=None, save_dir = "models/", do_restore_and_generate = False,
                        do_restore_and_train = False,
                        from_screenshot = False, ablation_layer = None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    global STYLE_LAYERS
    if use_mrf:
        STYLE_LAYERS = STYLE_LAYERS_MRF  # Easiest way to be compatible with no-mrf versions.

    input_shape = (1,) + contents[0].shape
    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = [{} for _ in range(len(contents))]
    style_features = [{} for _ in styles]
    ### TEST ###
    if style_as_content:
        style_content_features = [{} for _ in styles]
    ### TEST ENDS ###

    # Read the vgg net
    vgg_data, mean_pixel = vgg.read_net(path_to_network)
    print('Finished loading VGG.')


    if not do_restore_and_generate:  #if not do_restore_and_generate:
        # compute content features in feedforward mode
        content_pre_list = []
        for i in range(len(contents)):
            g = tf.Graph()
            with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
                image = tf.placeholder('float', shape=input_shape)
                # This mean pixel variable is unique to the input trained vgg network. It is independent of the input image.
                net = vgg.pre_read_net(vgg_data, image)
                content_pre_list.append(np.array([vgg.preprocess(contents[i], mean_pixel)]))
                content_features[i][CONTENT_LAYER] = net[CONTENT_LAYER].eval(
                    feed_dict={image: content_pre_list[-1]})
        print('Finished loading passing content image to it.')

        # compute style features in feedforward mode
        style_pre_list = []
        for i in range(len(styles)):
            g = tf.Graph()
            with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
                image = tf.placeholder('float', shape=style_shapes[i])
                net = vgg.pre_read_net(vgg_data, image)
                style_pre_list.append(np.array([vgg.preprocess(styles[i], mean_pixel)]))
                for layer in STYLE_LAYERS:
                    features = net[layer].eval(feed_dict={image: style_pre_list[-1]})
                    if use_mrf:
                        style_features[i][layer] = features
                    else:
                        # Calculate and store gramian.
                        features = np.reshape(features, (-1, features.shape[3]))
                        gram = np.matmul(features.T, features) / features.size
                        style_features[i][layer] = gram
                ### TEST ###
                if style_as_content:
                    style_content_features[i][CONTENT_LAYER]= net[CONTENT_LAYER].eval(feed_dict={image: style_pre_list[-1]})
                ### TEST ENDS ###
        print('Finished loading VGG and passing content and style image to it.')


    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if style_only:
            noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size, with_content_image=False)
        else:
            noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size, with_content_image=True)
        # input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different
        # style images.
        input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)], name='input_style_placeholder')
        image = generator_net_n_styles(noise_inputs, input_style_placeholder)
        net = vgg.pre_read_net(vgg_data, image)
        if not do_restore_and_generate:  #if not do_restore_and_generate:
            # content loss
            content_loss_for_each_content = [content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[i][CONTENT_LAYER]) /
                                             content_features[i][CONTENT_LAYER].size) for i in range(len(contents))]
            # style loss
            style_loss_for_each_style = []
            ### TEST ###
            if style_as_content:
                content_loss_for_each_style = []
            ### TEST ENDS ###
            for i in range(len(styles)):
                style_losses_for_each_style_layer = []
                for style_layer in STYLE_LAYERS:
                    layer = net[style_layer]
                    if use_mrf:
                        print('mrfing %d %s' %(i, style_layer))
                        style_losses_for_each_style_layer.append(mrf_loss(style_features[i][style_layer],layer))
                        print('mrfed %d %s' %(i, style_layer))
                    else:
                        # Use gramian loss.
                        gram = gramian(layer)
                        style_gram = style_features[i][style_layer]
                        style_losses_for_each_style_layer.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)

                style_loss_for_each_style.append(style_weight * style_blend_weights[i] * reduce(tf.add, style_losses_for_each_style_layer))

                ### TEST ###
                if style_as_content:
                    content_loss_for_each_style.append(content_weight * style_blend_weights[i] * (2 * tf.nn.l2_loss(
                        net[CONTENT_LAYER] - style_content_features[i][CONTENT_LAYER]) /
                                                                         style_content_features[i][CONTENT_LAYER].size))
                ### TEST ENDS ###

            # According to https://arxiv.org/abs/1610.07629 when "zero-padding is replaced with mirror-padding, and transposed convolutions (also sometimes called deconvolutions) are replaced with nearest-neighbor upsampling followed by a convolution.", tv is no longer needed.
            tv_loss = 0
            # tv_loss = tv_weight * total_variation(image)

            # overall loss
            if style_only:
                losses_for_each_content_and_style = [[style_loss for _ in content_loss_for_each_content] for
                                                     style_loss in style_loss_for_each_style]
            else:
                losses_for_each_content_and_style = [[style_loss + content_loss + tv_loss for content_loss in content_loss_for_each_content] for
                                                     style_loss in style_loss_for_each_style]
            overall_loss = 0
            for i, loss_for_each_content in enumerate(losses_for_each_content_and_style):
                for loss in loss_for_each_content:
                    overall_loss += loss
            # losses_for_each_style = [[content_loss + tv_loss + style_loss for content_loss in content_loss_for_each_content] for style_loss in style_loss_for_each_style]

            ### TEST ###
            if style_as_content:
                losses_for_each_style_when_input_is_style = [content_loss_for_each_style[i] + tv_loss + style_loss for i, style_loss in enumerate(style_loss_for_each_style)]
            ### TEST ENDS ###

            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            # They used learning rate = 0.001
            # Get all variables
            # TODO: tell which one is better, training all variables or training only scale and offset.
            scale_offset_var = get_scale_offset_var()
            if multiple_styles_train_scale_offset_only:
                train_step_for_each_content_and_style = [[
                    tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss, var_list = scale_offset_var)
                    if i != 0 else
                    tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss) for loss in loss_for_each_content]
                                             for i, loss_for_each_content in enumerate(losses_for_each_content_and_style)]
            else:
                train_step_for_each_content_and_style = [[
                    tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss) for loss in loss_for_each_content]
                                             for i, loss_for_each_content in enumerate(losses_for_each_content_and_style)]

            ### TEST ###
            if style_as_content:
                if multiple_styles_train_scale_offset_only:
                    train_step_for_each_style_when_input_is_style = [
                        tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss, var_list = scale_offset_var)
                        if i != 0 else
                        tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
                                                 for i, loss in enumerate(losses_for_each_style_when_input_is_style)]
                else:
                    train_step_for_each_style_when_input_is_style = [
                        tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
                                                 for i, loss in enumerate(losses_for_each_style_when_input_is_style)]
            ### TEST ENDS ###

            def print_progress(i, feed_dict, last=False):
                # stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
                    # Assume that the feed_dict is for the last content and style.
                    #stderr.write('  content loss: %g\n' % content_loss_for_each_content[-1].eval(feed_dict=feed_dict))
                    #stderr.write('    style loss: %g\n' % style_loss_for_each_style[-1].eval(feed_dict=feed_dict))
                    # stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                    stderr.write('    total loss: %g\n' % overall_loss.eval(feed_dict=feed_dict))

        # Optimization
        best_loss_for_each_content_and_style = [[float('inf') for content_i in range(len(contents))]for style_i in range(len(styles))]
        best_for_each_content_and_style = [[None for content_i in range(len(contents))] for style_i in range(len(styles))]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            if do_restore_and_generate:
                ckpt = tf.train.get_checkpoint_state(save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    stderr("No checkpoint found. Exiting program")
                    return

                if from_screenshot:
                    iterator = 0
                    kScreenX = 300
                    kScreenY = 300
                    if style_only:
                        noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                                     with_content_image=False)
                    else:
                        noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                                     with_content_image=True)

                    # input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different
                    # style images.
                    input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)],
                                                             name='input_style_placeholder')
                    image = generator_net_n_styles(noise_inputs, input_style_placeholder, reuse=True)
                    # generator_layers = get_all_layers_generator_net_n_styles(noise_inputs, input_style_placeholder)


                    # style_image_pyramids = [generate_image_pyramid(input_shape[1], input_shape[2], batch_size, style_pre) for style_pre in
                    #                         style_pre_list]
                    while True:
                        w = gtk.gdk.get_default_root_window()
                        sz = w.get_size()
                        print "The size of the window is %d x %d" % sz
                        # pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
                        # pb = pb.get_from_drawable(w, w.get_colormap(), 0, 0, 0, 0, sz[0], sz[1])
                        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, input_shape[1], input_shape[2])
                        pb = pb.get_from_drawable(w, w.get_colormap(), kScreenX, kScreenY, 0, 0, input_shape[1], input_shape[2])
                        content_image = pb.pixel_array
                        content_pre = np.array([vgg.preprocess(content_image, mean_pixel)])
                        # Now generate an image using the style_blend_weights given.
                        content_image_pyramid = generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre)
                        feed_dict = {}

                        if style_only:
                            noise = noise_pyramid(input_shape[1], input_shape[2], batch_size, ablation_layer=ablation_layer)
                        else:
                            noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size,
                                                                content_image_pyramid)

                        for index, noise_frame in enumerate(noise_inputs):
                            feed_dict[noise_frame] = noise[index]
                        feed_dict[input_style_placeholder] = \
                            np.array([[style_blend_weights[current_style_i]
                                       for current_style_i in range(len(styles))]])
                        # feed_dict = {}
                        # noise = noise_pyramid_w_content_img(m, batch_size, style_image_pyramids[0])
                        # for index, noise_frame in enumerate(noise_inputs):
                        #     feed_dict[noise_frame] = noise[index]
                        # feed_dict[input_style_placeholder] = \
                        #     np.array([[style_blend_weights[current_style_i]
                        #                for current_style_i in range(len(styles))]])
                        generated_image = image.eval(feed_dict=feed_dict)
                        # for generator_layer_name, generator_layer in generator_layers.iteritems():
                        #
                        #     try:
                        #         generator_layer_eval = generator_layer.eval(feed_dict=feed_dict)
                        #     except:
                        #         generator_layer_eval = generator_layer.eval()
                        #     generator_layer_contains_nan = np.isnan(np.sum(generator_layer_eval))
                        #     print('%s - %s: %s' % (generator_layer_name, str(generator_layer_contains_nan), str(generator_layer_eval)))
                        # raw_input()
                        iterator += 1
                        yield (iterator, vgg.unprocess(generated_image[0, :, :, :].reshape(input_shape[1:]), mean_pixel))  # Can't return because we are in a generator.

                # # Now generate an image using the style_blend_weights given.
                # content_image_pyramid = generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre)
                # feed_dict = {}
                # noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size, content_image_pyramid)
                # for index, noise_frame in enumerate(noise_inputs):
                #     feed_dict[noise_frame] = noise[index]
                # feed_dict[input_style_placeholder] = \
                #     np.array([[style_blend_weights[current_style_i]
                #                for current_style_i in range(len(styles))]])
                # generated_image = image.eval(feed_dict=feed_dict)
                # yield (None, vgg.unprocess(generated_image[0, :, :, :].reshape(input_shape[1:]), mean_pixel))  # Can't return because we are in a generator.

            else:
                # Do Training.
                iter_start = 0
                if do_restore_and_train:
                    ckpt = tf.train.get_checkpoint_state(save_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        iter_start = misc_util.get_global_step_from_save_dir(ckpt.model_checkpoint_path)
                    else:
                        stderr("No checkpoint found. Exiting program")
                        return
                else:
                    sess.run(tf.initialize_all_variables())
                content_image_pyramids = [generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre) for content_pre in content_pre_list]
                style_image_pyramids = [generate_image_pyramid(input_shape[1], input_shape[2], batch_size, style_pre) for style_pre in style_pre_list]
                for i in range(iter_start, iterations):
                    for style_i in range(len(styles)):
                        for content_i in range(len(contents)):
                            last_step = (i == iterations - 1)

                            if style_as_content:
                                # First feed the style image as content image. It does not make sense that we apply the style to style image and the style image changes.
                                feed_dict = {}

                                noise = noise_pyramid_w_content_img(style_pre_list[style_i].shape[1],style_pre_list[style_i].shape[2], batch_size, style_image_pyramids[style_i])
                                for index, noise_frame in enumerate(noise_inputs):
                                    feed_dict[noise_frame] = noise[index]
                                feed_dict[input_style_placeholder] = \
                                    np.array([[1.0 if current_style_i == style_i else 0.0
                                               for current_style_i in range(len(styles))]])
                                train_step_for_each_style_when_input_is_style[style_i].run(feed_dict=feed_dict)
                            # Then feed the content image.
                            feed_dict = {}

                            if style_only:
                                noise = noise_pyramid(input_shape[1], input_shape[2], batch_size, ablation_layer=ablation_layer)
                            else:
                                noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size,
                                                                    content_image_pyramids[content_i])

                            for index, noise_frame in enumerate(noise_inputs):
                                feed_dict[noise_frame] = noise[index]
                            feed_dict[input_style_placeholder] = \
                                np.array([[1.0 if current_style_i == style_i else 0.0
                                           for current_style_i in range(len(styles))]])
                            train_step_for_each_content_and_style[style_i][content_i].run(feed_dict=feed_dict)
                            if style_i == len(styles) - 1 and content_i == len(contents) - 1:
                                print_progress(i, feed_dict=feed_dict, last=last_step)

                            if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                                this_loss = losses_for_each_content_and_style[style_i][content_i].eval(feed_dict=feed_dict)
                                if this_loss < best_loss_for_each_content_and_style[style_i][content_i]:
                                    best_loss_for_each_content_and_style[style_i][content_i] = this_loss
                                    best_for_each_content_and_style[style_i][content_i] = image.eval(feed_dict=feed_dict)
                                if style_i == len(styles) - 1 and content_i == len(contents) - 1:
                                    saver.save(sess, save_dir + 'model.ckpt', global_step=i)
                                    # TODO: change description of what we yield.
                                    yield (
                                        (None if last_step else i),
                                        [[vgg.unprocess(best_for_each_content_and_style[style_i][content_i][0, :, :, :].reshape(input_shape[1:]), mean_pixel)
                                         for content_i in range(len(contents))] for style_i in range(len(styles))]
                                    )  # Because we now have batch, choose the first one in the batch as our sample image.


def generator_net_n_styles(input_noise_z, input_style_placeholder, reuse = False):
    """
    This function takes a list of tensors as input, and outputs a tensor with a given width and height. The output
    should be similar in style to the target image.
    :param input_noise_z: A list of tensors seeded with noise.
    :param input_style_placeholder: A one-hot tensor indicating which style we chose.
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return: the full sized generated image/texture.
    """

    # Different from Ulyanov et el, in the original paper z_k is the smallest input and z_1 is the largest.
    # Here we reverse the order
    with tf.variable_scope('texture_nets', reuse=reuse):
        noise_joined = input_noise_z[0]
        current_channels = 8
        channel_step_size = 8
        for noise_layer in input_noise_z[1:]:
            low_res = conv_block('block_low_%d' % current_channels, noise_joined, input_style_placeholder, current_channels, reuse = reuse)
            high_res = conv_block('block_high_%d' % current_channels, noise_layer, input_style_placeholder, channel_step_size, reuse = reuse)
            current_channels += channel_step_size
            noise_joined = join_block('join_%d' % current_channels, low_res, high_res)
        final_chain = conv_block("output_chain", noise_joined, input_style_placeholder, current_channels, reuse = reuse)
        return conv_relu_layers("output", final_chain, input_style_placeholder, kernel_size=1, out_channels=3, reuse = reuse)


def conv_block(name, input_layer, input_style_placeholder, out_channels, reuse = False):
    """
    Each convolution block in Figure 2 contains three convo-
    lutional layers, each of which is followed by a ReLU acti-
    vation layer. The convolutional layers contain respectively
    3 x 3, 3 x 3 and 1 x 1 filters. Filers are computed densely
    (stride one) and applied using circular convolution to re-
    move boundary effects, which is appropriate for textures.
    :param name:
    :param input_layer:
    :param out_channels:
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        block1 = conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = reuse)
        block2 = conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = reuse)
        block3 = conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels, reuse = reuse)
    return block3


def conv_relu_layers(name, input_layer, input_style_placeholder,  kernel_size, out_channels, relu_leak=0.01, reuse = False):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Instance Norm
        - LeakyReLu
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    """

    with tf.variable_scope(name, reuse=reuse):
        in_channels = input_layer.get_shape().as_list()[-1]

        # per https://arxiv.org/abs/1610.07629
        # initialize weights and bias using isotropic gaussian.
        # weights = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_channels, out_channels],
        #                                        mean=0.0, stddev=0.01, name='weights'))
        # biases = tf.Variable(tf.random_normal([out_channels], mean=0.0, stddev=0.01, name='biases'))

        weights = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels], tf.float32,tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [out_channels], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        conv = neural_util.conv2d_mirror_padding(input_layer, weights, biases, kernel_size)
        # conv = neural_util.conv2d(input_layer, weights, biases)
        norm = conditional_instance_norm(conv, input_style_placeholder, reuse = reuse)
        # norm = spatial_batch_norm(conv, input_style_placeholder, reuse = reuse)
        # relu = neural_util.leaky_relu(norm, relu_leak)
        # relu = tf.nn.relu(norm)
        relu = tf.nn.elu(norm, 'elu')
        return relu


def join_block(name, lower_res_layer, higher_res_layer):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    A block that combines two resolutions by upsampling the lower, batchnorming both, and concatting.
    """
    with tf.variable_scope(name):
        upsampled = tf.image.resize_nearest_neighbor(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        # # TODO: DO we need to normalize here?
        # No need to normalize here. According to https://arxiv.org/abs/1610.07629  normalize only after convolution.
        return tf.concat(3, [upsampled, higher_res_layer])

        # According to https://arxiv.org/abs/1603.03417 figure 8, we need to normalize after join block.
        # batch_norm_lower = spatial_batch_norm(upsampled, 'normLower')
        # batch_norm_higher = spatial_batch_norm(higher_res_layer, 'normHigher')
        # return tf.concat(3, [batch_norm_lower, batch_norm_higher])



def get_all_layers_generator_net_n_styles(input_noise_z, input_style_placeholder):
    """
    This function takes a list of tensors as input, and outputs a tensor with a given width and height. The output
    should be similar in style to the target image.
    :param input_noise_z: A list of tensors seeded with noise.
    :param input_style_placeholder: A one-hot tensor indicating which style we chose.
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return: the full sized generated image/texture.
    """

    # Different from Ulyanov et el, in the original paper z_k is the smallest input and z_1 is the largest.
    # Here we reverse the order
    ret = {}
    with tf.variable_scope('texture_nets', reuse=True):
        noise_joined_layers = [input_noise_z[0]]
        current_channels = 8
        channel_step_size = 8
        for noise_layer in input_noise_z[1:]:
            low_res = conv_block('block_low_%d' % current_channels, noise_joined_layers[-1], input_style_placeholder, current_channels, reuse = True)
            high_res = conv_block('block_high_%d' % current_channels, noise_layer, input_style_placeholder, channel_step_size, reuse = True)
            low_res_all_layers = get_all_layers_conv_block('block_low_%d' % current_channels, noise_joined_layers[-1], input_style_placeholder, current_channels)
            high_res_all_layers = get_all_layers_conv_block('block_high_%d' % current_channels, noise_layer, input_style_placeholder, channel_step_size)

            current_channels += channel_step_size
            noise_joined_layers.append(join_block('join_%d' % current_channels, low_res, high_res))
            ret['block_low_%d' % current_channels] = low_res
            ret['block_high_%d' % current_channels] = high_res
            ret['join_%d' % current_channels] = noise_joined_layers[-1]
            for key, val in low_res_all_layers.iteritems():
                ret[key] = val
            for key, val in high_res_all_layers.iteritems():
                ret[key] = val
        final_chain = conv_block("output_chain", noise_joined_layers[-1], input_style_placeholder, current_channels, reuse = True)
        final_layer = conv_relu_layers("output", final_chain, input_style_placeholder, kernel_size=1, out_channels=3, reuse = True)

    return ret


def get_all_layers_conv_block(name, input_layer, input_style_placeholder, out_channels):
    """
    Each convolution block in Figure 2 contains three convo-
    lutional layers, each of which is followed by a ReLU acti-
    vation layer. The convolutional layers contain respectively
    3 x 3, 3 x 3 and 1 x 1 filters. Filers are computed densely
    (stride one) and applied using circular convolution to re-
    move boundary effects, which is appropriate for textures.
    :param name:
    :param input_layer:
    :param out_channels:
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return:
    """
    with tf.variable_scope(name, reuse=True):
        block1 = conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = True)
        block2 = conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = True)
        block3 = conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels, reuse = True)
        layer1_layers = get_all_layers_conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = True)
        layer2_layers = get_all_layers_conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = True)
        layer3_layers = get_all_layers_conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels, reuse = True)
    ret = {}
    for key, val in layer1_layers.iteritems():
        ret[name + 'layer1' + key] = val
    for key, val in layer2_layers.iteritems():
        ret[name + 'layer2' + key] = val
    for key, val in layer3_layers.iteritems():
        ret[name + 'layer3' + key] = val
    return ret

def get_all_layers_conv_relu_layers(name, input_layer, input_style_placeholder,  kernel_size, out_channels, relu_leak=0.01, reuse = False):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Instance Norm
        - LeakyReLu
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    """

    with tf.variable_scope(name, reuse=reuse):
        in_channels = input_layer.get_shape().as_list()[-1]

        # per https://arxiv.org/abs/1610.07629
        # initialize weights and bias using isotropic gaussian.
        # weights = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_channels, out_channels],
        #                                        mean=0.0, stddev=0.01, name='weights'))
        # biases = tf.Variable(tf.random_normal([out_channels], mean=0.0, stddev=0.01, name='biases'))

        weights = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels], tf.float32,tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [out_channels], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        conv = neural_util.conv2d_mirror_padding(input_layer, weights, biases, kernel_size)
        # conv = neural_util.conv2d(input_layer, weights, biases)
        norm = conditional_instance_norm(conv, input_style_placeholder, reuse = reuse)
        # norm = spatial_batch_norm(conv, input_style_placeholder, reuse = reuse)
        # relu = neural_util.leaky_relu(norm, relu_leak)
        # relu = tf.nn.relu(norm)
        relu = tf.nn.elu(norm, 'elu')

        num_channels = conv.get_shape().as_list()[3]

        num_styles = input_style_placeholder.get_shape().as_list()[1]
        scale = tf.get_variable('scale', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())

        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        variance = tf.abs(variance)
        variance_epsilon = 0.001
        inv = math_ops.rsqrt(variance + variance_epsilon)
        return {'conv': conv, 'norm': norm, 'relu':relu, 'scale':scale, 'offset':offset, 'mean':mean, 'variance':variance, 'inv':inv}