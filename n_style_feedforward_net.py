"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

import tensorflow as tf
import numpy as np
from sys import stderr
import scipy.misc
import gtk.gdk

import neural_util
import vgg
from general_util import *
from feedforward_style_net_util import *

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# TODO: reformat and check comments. 20161105
def style_synthesis_net(path_to_network, contents, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, style_as_content = False, multiple_styles_train_scale_offset_only = False, print_iterations=None, checkpoint_iterations=None,
                        save_dir = "models/", do_restore_and_generate = False, from_screenshot = False):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
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
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram
            ### TEST ###
            if style_as_content:
                style_content_features[i][CONTENT_LAYER]= net[CONTENT_LAYER].eval(feed_dict={image: style_pre_list[-1]})
            ### TEST ENDS ###

    # make stylized image using backpropogation
    with tf.Graph().as_default():

        noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size, with_content_image=True)
        # input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different
        # style images.
        input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)], name='input_style_placeholder')
        image = generator_net_n_styles(noise_inputs, input_style_placeholder)
        net = vgg.pre_read_net(vgg_data, image)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                         content_features[CONTENT_LAYER].size)
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

        # TODO: THIS PART WAS GIVING ME NAN WHEN I ADDED THE CONTENT IMAGE.
        # # total variation denoising
        # tv_y_size = tensor_size(image[:, 1:, :, :])
        # tv_x_size = tensor_size(image[:, :, 1:, :])
        # tv_loss = tv_weight * 2 * (
        #     (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
        #      tv_y_size) +
        #     (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
        #      tv_x_size))
        # # overall loss
        # loss = content_loss + style_loss + tv_loss


        tv_loss = tv_weight * total_variation(image)

        # overall loss
        losses_for_each_style = [content_loss + tv_loss + style_loss for style_loss in style_loss_for_each_style]
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
            train_step_for_each_style = [
                tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss, var_list = scale_offset_var)
                if i != 0 else
                tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
                                         for i, loss in enumerate(losses_for_each_style)]
        else:
            train_step_for_each_style = [
                tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss)
                                         for i, loss in enumerate(losses_for_each_style)]

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
            stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
            if last or (print_iterations and i % print_iterations == 0):
                stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                stderr.write('    style loss: %g\n' % style_loss.eval(feed_dict=feed_dict))
                stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                stderr.write('    total loss: %g\n' % loss.eval(feed_dict=feed_dict))

        # Optimization
        best_loss_for_each_style = [float('inf') for style_i in range(len(styles))]
        best_for_each_style = [None for style_i in range(len(styles))]

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
                    kScreenX = 100
                    kScreenY = 100
                    noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size, with_content_image=True)
                    # input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different
                    # style images.
                    input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)],
                                                             name='input_style_placeholder')
                    image = generator_net_n_styles(noise_inputs, input_style_placeholder, reuse=True)


                    style_image_pyramids = [generate_image_pyramid(input_shape[1], input_shape[2], batch_size, style_pre) for style_pre in
                                            style_pre_list]
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
                        noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size, content_image_pyramid)
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
                        iterator += 1
                        yield (iterator, vgg.unprocess(generated_image[0, :, :, :].reshape(input_shape[1:]), mean_pixel))  # Can't return because we are in a generator.

                # Now generate an image using the style_blend_weights given.
                content_image_pyramid = generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre)
                feed_dict = {}
                noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size, content_image_pyramid)
                for index, noise_frame in enumerate(noise_inputs):
                    feed_dict[noise_frame] = noise[index]
                feed_dict[input_style_placeholder] = \
                    np.array([[style_blend_weights[current_style_i]
                               for current_style_i in range(len(styles))]])
                generated_image = image.eval(feed_dict=feed_dict)
                yield (None, vgg.unprocess(generated_image[0, :, :, :].reshape(input_shape[1:]), mean_pixel))  # Can't return because we are in a generator.

            else:
                # Do Training.
                sess.run(tf.initialize_all_variables())
                content_image_pyramid = generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre)
                style_image_pyramids = [generate_image_pyramid(input_shape[1], input_shape[2], batch_size, style_pre) for style_pre in style_pre_list]
                for i in range(iterations):
                    for style_i in range(len(styles)):
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
                        noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size, content_image_pyramid)
                        for index, noise_frame in enumerate(noise_inputs):
                            feed_dict[noise_frame] = noise[index]
                        feed_dict[input_style_placeholder] = \
                            np.array([[1.0 if current_style_i == style_i else 0.0
                                       for current_style_i in range(len(styles))]])
                        train_step_for_each_style[style_i].run(feed_dict=feed_dict)
                        if style_i == len(styles) - 1:
                            print_progress(i, feed_dict=feed_dict, last=last_step)

                        if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                            this_loss = losses_for_each_style[style_i].eval(feed_dict=feed_dict)
                            if this_loss < best_loss_for_each_style[style_i]:
                                best_loss_for_each_style[style_i] = this_loss
                                best_for_each_style[style_i] = image.eval(feed_dict=feed_dict)
                            if style_i == len(styles) - 1:
                                saver.save(sess, save_dir + 'model.ckpt', global_step=i)
                                # TODO: change description of what we yield.
                                yield (
                                    (None if last_step else i),
                                    [vgg.unprocess(best_for_each_style[style_i][0, :, :, :].reshape(input_shape[1:]), mean_pixel)
                                     for style_i in range(len(styles))]
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
    with tf.variable_scope('texture_nets'):
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
    with tf.variable_scope(name):
        block1 = conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = reuse)
        block2 = conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels, reuse = reuse)
        block3 = conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels, reuse = reuse)
    return block3


def conv_relu_layers(name, input_layer, input_style_placeholder,  kernel_size, out_channels, relu_leak=0.01, reuse = False):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad (TODO)
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
        # Do mirror padding in the future.
        conv = neural_util.conv2d(input_layer, weights, biases)
        norm = conditional_instance_norm(conv, input_style_placeholder, reuse = reuse)
        relu = neural_util.leaky_relu(norm, relu_leak)
        return relu


def join_block(name, lower_res_layer, higher_res_layer):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    A block that combines two resolutions by upsampling the lower, batchnorming both, and concatting.
    """
    with tf.variable_scope(name):
        upsampled = tf.image.resize_nearest_neighbor(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        # # TODO: DO we need to normalize here?
        # # No need to normalize here. According to https://arxiv.org/abs/1610.07629  normalize only after convolution.
        # return tf.concat(3, [upsampled, higher_res_layer])

        # According to https://arxiv.org/abs/1603.03417 figure 8, we need to normalize after join block.
        batch_norm_lower = spatial_batch_norm(upsampled, 'normLower')
        batch_norm_higher = spatial_batch_norm(higher_res_layer, 'normHigher')
        return tf.concat(3, [batch_norm_lower, batch_norm_higher])

