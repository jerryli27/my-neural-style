"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417 and
https://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

import gtk.gdk
import cv2
import operator

import vgg
from feedforward_style_net_util import *
from mrf_util import mrf_loss
import johnson_feedforward_net_util

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
# STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
STYLE_LAYERS = (
'relu1_2', 'relu2_2', 'relu3_2', 'relu4_2')  # Set according to https://github.com/DmitryUlyanov/texture_nets
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.


# TODO: Needs reformatting. Rewrite it so that content images now appear in the same batch instead of using a big for
# loop. it will save lots of memory.

# TODO: change rtype
def style_synthesis_net(path_to_network, height, width, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, lr_decay_steps=200, min_lr=0.001, lr_decay_rate=0.7,
                        style_only=False,
                        multiple_styles_train_scale_offset_only=False, use_mrf=False,
                        use_johnson=False, print_iterations=None,
                        checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                        do_restore_and_train=False, content_folder = None, style_folder = None,
                        from_screenshot=False, from_webcam=False, test_img_dir = None, ablation_layer=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :param: lr_decay_steps: learning rate decays by lr_decay_rate after lr_decay steps.
    Default per https://arxiv.org/abs/1603.03417
    :param: min_lr: The minimum learning rate. Default per https://arxiv.org/abs/1603.03417
    :param: lr_decay_rate: learning rate decays by lr_decay_rate after lr_decay steps.
    Default per https://arxiv.org/abs/1603.03417
    :rtype: iterator[tuple[int|None,image]]
    """
    global STYLE_LAYERS
    if use_mrf:
        STYLE_LAYERS = STYLE_LAYERS_MRF  # Easiest way to be compatible with no-mrf versions.

    input_shape = (1,height, width, 3)
    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    # content_features = [{} for _ in range(len(contents))]
    style_features = [{} for _ in styles]

    # Read the vgg net
    vgg_data, mean_pixel = vgg.read_net(path_to_network)
    # vgg_data_dict = loadWeightsData('./vgg19.npy')
    print('Finished loading VGG.')

    # if not do_restore_and_generate:
    #     # Compute style features in feedforward mode.
    #     style_pre_list = []
    #     for i in range(len(styles)):
    #         g = tf.Graph()
    #         with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    #             image = tf.placeholder('float', shape=style_shapes[i])
    #             net = vgg.pre_read_net(vgg_data, image)
    #             style_pre_list.append(np.array([vgg.preprocess(styles[i], mean_pixel)]))
    #             for layer in STYLE_LAYERS:
    #                 current_feature = net[layer].eval(feed_dict={image: style_pre_list[-1]})
    #                 if use_mrf:
    #                     style_features[i][layer] = current_feature
    #                 else:
    #                     # Calculate and store gramian.
    #                     current_feature = np.reshape(current_feature, (-1, current_feature.shape[3]))
    #                     gram = np.matmul(current_feature.T, current_feature) / current_feature.size
    #                     style_features[i][layer] = gram
    #     print('Finished loading VGG and passing content and style image to it.')

    # Make stylized image using backpropogation.
    with tf.Graph().as_default():
        if use_johnson:
            inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
            image = johnson_feedforward_net_util.net(inputs / 255.0)
            image = vgg.preprocess(image, mean_pixel)
        else:
            if style_only:
                noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                             with_content_image=False)
            else:
                noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                             with_content_image=True)
            # Input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of
            # different style images.
            input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)], name='input_style_placeholder')
            image = generator_net_n_styles(noise_inputs, input_style_placeholder)
            # TODO: Do I need to preprocess the generated image here? When I use the johnson model it seems I have to do so.
            # image = vgg.preprocess(image, mean_pixel)
        net = vgg.pre_read_net(vgg_data, image)
        if not do_restore_and_generate:
            global_step_init = tf.constant(0)
            global_step = tf.get_variable(name='global_step', trainable=False, initializer=global_step_init)
            learning_rate_decayed_init = tf.constant(learning_rate)
            learning_rate_decayed = tf.get_variable(name='learning_rate_decayed', trainable=False,
                                                    initializer=learning_rate_decayed_init)
            # content loss
            content_images = tf.placeholder(tf.float32, [batch_size, input_shape[1], input_shape[2], 3],
                                            name='content_images_placeholder')
            content_pre = vgg.preprocess(content_images, mean_pixel)
            content_net = vgg.pre_read_net(vgg_data, content_pre)
            content_features = content_net[CONTENT_LAYER]
            content_features_shape = content_features.get_shape().as_list()
            content_features_size = content_features_shape[0] * content_features_shape[1] * content_features_shape[2] * content_features_shape[3]
            content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features) / content_features_size)


            # style loss
            style_images = tf.placeholder('float', shape=[batch_size, input_shape[1], input_shape[2], 3])
            style_pre = vgg.preprocess(style_images, mean_pixel)
            style_net = vgg.pre_read_net(vgg_data, style_pre)
            style_features = {}
            for layer in STYLE_LAYERS:
                current_feature = style_net[layer]
                if use_mrf:
                    style_features[layer] = current_feature
                else:
                    # Calculate and store gramian.
                    style_features[layer] = gramian(current_feature)

            # style_loss_for_each_style = []
            # for i in range(len(styles)):
            #     style_losses_for_each_style_layer = []
            #     for style_layer in STYLE_LAYERS:
            #         layer = net[style_layer]
            #         if use_mrf:
            #             print('mrfing %d %s' % (i, style_layer))
            #             style_losses_for_each_style_layer.append(
            #                 mrf_loss(style_features[i][style_layer], layer, name='%d%s' % (i, style_layer)))
            #             print('mrfed %d %s' % (i, style_layer))
            #         else:
            #             # Use gramian loss.
            #             gram = gramian(layer)
            #             style_gram = style_features[i][style_layer]
            #             style_losses_for_each_style_layer.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            #
            #     style_loss_for_each_style.append(
            #         style_weight * style_blend_weights[i] * reduce(tf.add, style_losses_for_each_style_layer) / batch_size)

            style_losses_for_each_style_layer = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                if use_mrf:
                    raise NotImplementedError
                    print('mrfing %s' % (style_layer))
                    style_losses_for_each_style_layer.append(
                        mrf_loss(style_features[style_layer], layer, name='%s' % style_layer))
                    print('mrfed %s' % (style_layer))
                else:
                    # Use gramian loss.
                    gram = gramian(layer)
                    style_gram = style_features[style_layer]
                    style_gram_size = reduce(operator.mul, style_gram.get_shape().as_list(),1)
                    # TODO: DO I need  ?
                    style_losses_for_each_style_layer.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram_size)

            style_loss = style_weight * reduce(tf.add, style_losses_for_each_style_layer) / batch_size
            # According to https://arxiv.org/abs/1610.07629 when "zero-padding is replaced with mirror-padding,
            # and transposed convolutions (also sometimes called deconvolutions) are replaced with nearest-neighbor
            # upsampling followed by a convolution.", tv is no longer needed.
            # But in other papers I've seen tv-loss still applicable, like in https://arxiv.org/abs/1603.08155.
            # TODO: find out the difference between having tv loss and not.
            # tv_loss = 0
            tv_loss = tv_weight * total_variation(image)

            # overall loss
            if style_only:
                overall_loss = style_loss
            else:
                overall_loss = style_loss + content_loss + tv_loss
            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            # TODO: tell which one is better, training all variables or training only scale and offset.
            # Get all variables
            scale_offset_var = get_scale_offset_var()
            if multiple_styles_train_scale_offset_only:
                raise NotImplementedError
                train_step= tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                                beta2=0.999).minimize(overall_loss,
                                                                      var_list=scale_offset_var) if i != 0 else tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                                beta2=0.999).minimize(overall_loss)
            else:
                train_step = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9, beta2=0.999).minimize(overall_loss)

            def print_progress(i, feed_dict, last=False):
                stderr.write(
                    'Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    # stderr.write(
                    #     'Iteration %d/%d\tLearning rate %f\n' % (i + 1, iterations, learning_rate_decayed.eval()))
                    # Assume that the feed_dict is for the last content and style.
                    if not style_only:
                        stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                    stderr.write('    style loss: %g\n' % style_loss.eval(feed_dict=feed_dict))
                    if not style_only:
                        stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                    stderr.write('    total loss: %g\n' % overall_loss.eval(feed_dict=feed_dict))

        # Optimization
        best_loss = float('inf')
        best_image = None

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
                    kScreenX = 300
                    kScreenY = 300
                elif from_webcam:
                    cap = cv2.VideoCapture(0)
                    ret = cap.set(3, 1280)
                    ret = cap.set(4, 960)
                    ret, frame = cap.read()
                    print('The dimension of this camera is : %d x %d' % (frame.shape[1], frame.shape[0]))
                else:
                    assert test_img_dir is not None

                if use_johnson:
                    inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
                    image = johnson_feedforward_net_util.net(inputs, reuse=True)
                    image = vgg.preprocess(image, mean_pixel)
                else:
                    if style_only:
                        noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                                     with_content_image=False)
                    else:
                        noise_inputs = input_pyramid("noise", input_shape[1], input_shape[2], batch_size,
                                                     with_content_image=True)

                    # input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of
                    # different style images.
                    input_style_placeholder = tf.placeholder(tf.float32, [1, len(styles)],
                                                             name='input_style_placeholder')

                    image = generator_net_n_styles(noise_inputs, input_style_placeholder, reuse=True)
                # FOR DEBUGGING:
                # generator_layers = get_all_layers_generator_net_n_styles(noise_inputs, input_style_placeholder)
                # END
                iterator = 0

                while from_screenshot or from_webcam or (iterator == 0):
                    if from_screenshot:
                        w = gtk.gdk.get_default_root_window()
                        sz = w.get_size()
                        print "The size of the window is %d x %d" % sz
                        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, input_shape[1], input_shape[2])
                        pb = pb.get_from_drawable(w, w.get_colormap(), kScreenX, kScreenY, 0, 0, input_shape[1],
                                                  input_shape[2])
                        content_image = pb.pixel_array
                    elif from_webcam:
                        ret, frame = cap.read()
                        content_image = scipy.misc.imresize(frame, (input_shape[1], input_shape[2]))
                    else:
                        content_image = imread(test_img_dir, (input_shape[1], input_shape[2]))

                    content_pre = np.array([vgg.preprocess(content_image, mean_pixel)])
                    # Now generate an image using the style_blend_weights given.
                    content_image_pyramid = generate_image_pyramid(input_shape[1], input_shape[2], batch_size,
                                                                   content_pre)
                    feed_dict = {}

                    if use_johnson:
                        feed_dict[inputs] = content_pre
                    else:
                        if style_only:
                            noise = noise_pyramid(input_shape[1], input_shape[2], batch_size,
                                                  ablation_layer=ablation_layer)
                        else:
                            noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size,
                                                                content_image_pyramid, ablation_layer=ablation_layer)

                        for index, noise_frame in enumerate(noise_inputs):
                            feed_dict[noise_frame] = noise[index]
                        feed_dict[input_style_placeholder] = \
                            np.array([[style_blend_weights[current_style_i]
                                       for current_style_i in range(len(styles))]])
                    generated_image = image.eval(feed_dict=feed_dict)

                    # FOR DEBUGGING:
                    # for generator_layer_name, generator_layer in generator_layers.iteritems():
                    #
                    #     try:
                    #         generator_layer_eval = generator_layer.eval(feed_dict=feed_dict)
                    #     except:
                    #         generator_layer_eval = generator_layer.eval()
                    #     generator_layer_contains_nan = np.isnan(np.sum(generator_layer_eval))
                    #     print('%s - %s: %s' % (generator_layer_name, str(generator_layer_contains_nan), str(generator_layer_eval)))
                    # raw_input()
                    # END
                    iterator += 1
                    # Can't return because we are in a generator.
                    yield (iterator, vgg.unprocess(scipy.misc.imresize(generated_image[0, :, :, :], (input_shape[1], input_shape[2])), mean_pixel))
            else:
                # Do Training.
                iter_start = 0
                if do_restore_and_train:
                    ckpt = tf.train.get_checkpoint_state(save_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        iter_start = get_global_step_from_save_dir(ckpt.model_checkpoint_path)
                    else:
                        stderr("No checkpoint found. Exiting program")
                        return
                else:
                    sess.run(tf.initialize_all_variables())

                # Get path to all content images.
                content_dirs =get_all_image_paths_in_dir(content_folder)
                # Ignore the ones at the end to avoid
                if batch_size != 1:
                    content_dirs = content_dirs[:-(len(content_dirs) % batch_size)]
                style_dirs =get_all_image_paths_in_dir(style_folder)
                # Ignore the ones at the end to avoid
                if batch_size != 1:
                    style_dirs = style_dirs[:-(len(style_dirs) % batch_size)]

                for i in range(iter_start, iterations):
                    # First decay the learning rate if we need to
                    if (i % lr_decay_steps == 0):
                        current_lr = learning_rate_decayed.eval()
                        sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))

                    # Load content images
                    current_content_dirs = get_batch(content_dirs, i * batch_size, batch_size)
                    content_pre_list = read_and_resize_batch_images(current_content_dirs, input_shape[1], input_shape[2])

                    # Load style images
                    current_style_dirs = get_batch(style_dirs, i * batch_size, batch_size)
                    style_pre_list = read_and_resize_batch_images(current_style_dirs, input_shape[1], input_shape[2])

                    last_step = (i == iterations - 1)
                    # Feed the content image.
                    feed_dict = {content_images:content_pre_list, style_images:style_pre_list}
                    if use_johnson:
                        if style_only:
                            feed_dict[inputs] = np.random.rand(batch_size, input_shape[1], input_shape[2], 3)
                        else:
                            feed_dict[inputs] = content_pre_list
                    else:
                        if style_only:
                            noise = noise_pyramid(input_shape[1], input_shape[2], batch_size,
                                                  ablation_layer=ablation_layer)
                        else:
                            raise NotImplementedError
                            content_image_pyramids = generate_image_pyramid_from_content_list(input_shape[1], input_shape[2], content_pre_list)
                            # TODO: This is for sure not working.
                            noise = noise_pyramid_w_content_img(input_shape[1], input_shape[2], batch_size,
                                                                content_image_pyramids)

                        for index, noise_frame in enumerate(noise_inputs):
                            feed_dict[noise_frame] = noise[index]
                        feed_dict[input_style_placeholder] = \
                            np.array([[1.0 if current_style_i == style_i else 0.0
                                       for current_style_i in range(len(styles))]])



                    train_step.run(feed_dict=feed_dict)
                    print_progress(i, feed_dict=feed_dict, last=last_step)

                    if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                        saver.save(sess, save_dir + 'model.ckpt', global_step=i)

                        if test_img_dir is not None:
                            test_image = imread(test_img_dir)
                            test_image_shape = test_image.shape
                            # The for loop will run once and terminate. Can't use return and yield in the same function so this is a hacky way to do it.
                            for _, generated_image in style_synthesis_net(path_to_network, test_image_shape[0],
                                                                  test_image_shape[1], styles, iterations,
                                                                  1,
                                                                  content_weight, style_weight,
                                                                  style_blend_weights, tv_weight,
                                                                  learning_rate,
                                                                  style_only=style_only,
                                                                  multiple_styles_train_scale_offset_only=multiple_styles_train_scale_offset_only,
                                                                  use_mrf=use_mrf,
                                                                  use_johnson=use_johnson,
                                                                  save_dir=save_dir,
                                                                  do_restore_and_generate=True,
                                                                  do_restore_and_train=False,
                                                                  from_screenshot=False, from_webcam=False,
                                                                  test_img_dir=test_img_dir,
                                                                  ablation_layer=ablation_layer):
                                pass

                            best_image = generated_image

                        # Because we now have batch, choose the first one in the batch as our sample image.
                        yield (
                            (None if last_step else i), None if test_img_dir is None else best_image
                        )
