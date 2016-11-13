"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417 and
https://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

import gtk.gdk
import cv2

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
def style_synthesis_net(path_to_network, contents, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, lr_decay_steps=200, min_lr=0.001, lr_decay_rate=0.7,
                        style_only=False,
                        multiple_styles_train_scale_offset_only=False, use_mrf=False,
                        use_johnson=False, print_iterations=None,
                        checkpoint_iterations=None, save_dir="models/", do_restore_and_generate=False,
                        do_restore_and_train=False,
                        from_screenshot=False, ablation_layer=None):
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

    input_shape = (1,) + contents[0].shape
    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = [{} for _ in range(len(contents))]
    style_features = [{} for _ in styles]

    # Read the vgg net
    vgg_data, mean_pixel = vgg.read_net(path_to_network)
    print('Finished loading VGG.')

    if not do_restore_and_generate:
        # Compute content features in feedforward mode.
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

        # Compute style features in feedforward mode.
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
        print('Finished loading VGG and passing content and style image to it.')

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
            content_loss_for_each_content = [content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[i][CONTENT_LAYER]) /
                                                               content_features[i][CONTENT_LAYER].size) for i in
                                             range(len(contents))]
            # style loss
            style_loss_for_each_style = []
            for i in range(len(styles)):
                style_losses_for_each_style_layer = []
                for style_layer in STYLE_LAYERS:
                    layer = net[style_layer]
                    if use_mrf:
                        print('mrfing %d %s' % (i, style_layer))
                        style_losses_for_each_style_layer.append(
                            mrf_loss(style_features[i][style_layer], layer, name='%d%s' % (i, style_layer)))
                        print('mrfed %d %s' % (i, style_layer))
                    else:
                        # Use gramian loss.
                        gram = gramian(layer)
                        style_gram = style_features[i][style_layer]
                        style_losses_for_each_style_layer.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)

                style_loss_for_each_style.append(
                    style_weight * style_blend_weights[i] * reduce(tf.add, style_losses_for_each_style_layer))
            # According to https://arxiv.org/abs/1610.07629 when "zero-padding is replaced with mirror-padding,
            # and transposed convolutions (also sometimes called deconvolutions) are replaced with nearest-neighbor
            # upsampling followed by a convolution.", tv is no longer needed.
            # But in other papers I've seen tv-loss still applicable, like in https://arxiv.org/abs/1603.08155.
            # TODO: find out the difference between having tv loss and not.
            tv_loss = 0
            # tv_loss = tv_weight * total_variation(image)

            # overall loss
            if style_only:
                losses_for_each_content_and_style = [[style_loss for _ in content_loss_for_each_content] for
                                                     style_loss in style_loss_for_each_style]
            else:
                losses_for_each_content_and_style = [
                    [style_loss + content_loss + tv_loss for content_loss in content_loss_for_each_content] for
                    style_loss in style_loss_for_each_style]
            overall_loss = 0
            for i, loss_for_each_content in enumerate(losses_for_each_content_and_style):
                for loss in loss_for_each_content:
                    overall_loss += loss
            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            # TODO: tell which one is better, training all variables or training only scale and offset.
            # Get all variables
            scale_offset_var = get_scale_offset_var()
            if multiple_styles_train_scale_offset_only:
                train_step_for_each_content_and_style = [[
                                                             tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                                                                    beta2=0.999).minimize(loss,
                                                                                                          var_list=scale_offset_var)
                                                             if i != 0 else
                                                             tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                                                                    beta2=0.999).minimize(loss) for loss
                                                             in loss_for_each_content]
                                                         for i, loss_for_each_content in
                                                         enumerate(losses_for_each_content_and_style)]
            else:
                train_step_for_each_content_and_style = [[
                                                             tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                                                                    beta2=0.999).minimize(loss) for loss
                                                             in loss_for_each_content]
                                                         for i, loss_for_each_content in
                                                         enumerate(losses_for_each_content_and_style)]

            def print_progress(i, feed_dict, last=False):
                # stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    stderr.write(
                        'Iteration %d/%d\tLearning rate %f\n' % (i + 1, iterations, learning_rate_decayed.eval()))
                    # Assume that the feed_dict is for the last content and style.
                    # stderr.write('  content loss: %g\n' % content_loss_for_each_content[-1].eval(feed_dict=feed_dict))
                    # stderr.write('    style loss: %g\n' % style_loss_for_each_style[-1].eval(feed_dict=feed_dict))
                    # stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                    stderr.write('    total loss: %g\n' % overall_loss.eval(feed_dict=feed_dict))

        # Optimization
        best_loss_for_each_content_and_style = [[float('inf') for content_i in range(len(contents))] for style_i in
                                                range(len(styles))]
        best_for_each_content_and_style = [[None for content_i in range(len(contents))] for style_i in
                                           range(len(styles))]

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
                else:
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    print('The dimension of this camera is : %d x %d' % (frame.shape[0], frame.shape[1]))

                if use_johnson:
                    inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
                    image = johnson_feedforward_net_util.net(inputs / 255.0, reuse=True)
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

                while True:
                    if from_screenshot:
                        w = gtk.gdk.get_default_root_window()
                        sz = w.get_size()
                        print "The size of the window is %d x %d" % sz
                        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, input_shape[1], input_shape[2])
                        pb = pb.get_from_drawable(w, w.get_colormap(), kScreenX, kScreenY, 0, 0, input_shape[1],
                                                  input_shape[2])
                        content_image = pb.pixel_array
                    else:
                        ret, frame = cap.read()
                        content_image = frame

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
                    yield (iterator, vgg.unprocess(generated_image[0, :, :, :].reshape(input_shape[1:]), mean_pixel))
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
                content_image_pyramids = [
                    generate_image_pyramid(input_shape[1], input_shape[2], batch_size, content_pre) for content_pre in
                    content_pre_list]
                for i in range(iter_start, iterations):
                    # First decay the learning rate if we need to
                    if (i % lr_decay_steps == 0):
                        current_lr = learning_rate_decayed.eval()
                        sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))

                    for style_i in range(len(styles)):
                        for content_i in range(len(contents)):
                            last_step = (i == iterations - 1)
                            # Feed the content image.
                            feed_dict = {}
                            if use_johnson:
                                if style_only:
                                    feed_dict[inputs] = np.random.rand(batch_size, style_pre_list[style_i].shape[1],
                                                                       style_pre_list[style_i].shape[2], 3)
                                else:
                                    feed_dict[inputs] = content_pre_list[content_i]
                            else:
                                if style_only:
                                    noise = noise_pyramid(input_shape[1], input_shape[2], batch_size,
                                                          ablation_layer=ablation_layer)
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
                                this_loss = losses_for_each_content_and_style[style_i][content_i].eval(
                                    feed_dict=feed_dict)
                                if this_loss < best_loss_for_each_content_and_style[style_i][content_i]:
                                    best_loss_for_each_content_and_style[style_i][content_i] = this_loss
                                    best_for_each_content_and_style[style_i][content_i] = image.eval(
                                        feed_dict=feed_dict)
                                if style_i == len(styles) - 1 and content_i == len(contents) - 1:
                                    saver.save(sess, save_dir + 'model.ckpt', global_step=i)
                                    # Because we now have batch, choose the first one in the batch as our sample image.
                                    yield (
                                        (None if last_step else i),
                                        [[vgg.unprocess(
                                            best_for_each_content_and_style[style_i][content_i][0, :, :, :].reshape(
                                                input_shape[1:]), mean_pixel)
                                          for content_i in range(len(contents))] for style_i in range(len(styles))]
                                    )
