"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417 and
https://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

# import gtk.gdk
from sys import stderr

import cv2

import johnson_feedforward_net_util
import neural_doodle_util
import skip_noise_4_feedforward_net
from feedforward_style_net_util import *
from mrf_util import mrf_loss

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1') # According to http://arxiv.org/abs/1603.03417
# STYLE_LAYERS = (
#     'relu1_2', 'relu2_2', 'relu3_2', 'relu4_2')  # Set according to https://github.com/DmitryUlyanov/texture_nets
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.
NUM_NOISE_LAYERS = 0  # This is for adding noise to the input to the generator network. Maybe I should delete this.


# TODO: change rtype
def style_synthesis_net(path_to_network, height, width, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, lr_decay_steps=200, min_lr=0.001, lr_decay_rate=0.7,
                        style_only=False,
                        multiple_styles_train_scale_offset_only=False, use_mrf=False,
                        use_johnson=False, use_skip_noise_4 = False, print_iterations=None,
                        checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                        do_restore_and_train=False, content_folder=None,
                        use_semantic_masks=False, mask_folder=None, mask_resize_as_feature=True,
                        style_semantic_masks=None, semantic_masks_weight=1.0, semantic_masks_num_layers=1,
                        from_screenshot=False, from_webcam=False, test_img_dir=None, ablation_layer=None,
                        content_img_style_weight_mask=None, style_weight_mask_for_training = None):
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
    :param: use_semantic_masks: If it is true, the input to the generator network will be the semantic masks instead
    of the content image. The content image will serve as ground truth for loss (I haven't decided whether to use content
    or style loss).
    :rtype: iterator[tuple[int|None,image]]
    """

    # Before training, make sure everything is set correctly.
    global STYLE_LAYERS
    if not use_johnson and not use_skip_noise_4:
        print("Please select one generator network, either johnson or skip_noise_4.")
        raise AssertionError

    if use_mrf:
        STYLE_LAYERS = STYLE_LAYERS_MRF  # Easiest way to be compatible with no-mrf versions.
    if use_semantic_masks:
        assert mask_folder is not None
    assert not (use_skip_noise_4 and use_johnson)

    if content_img_style_weight_mask is not None:
        if do_restore_and_train or not do_restore_and_generate:
            assert style_weight_mask_for_training is not None
        if do_restore_and_generate and (height != content_img_style_weight_mask.shape[1] or width != content_img_style_weight_mask.shape[2]):
            print("The shape of style_weight_mask is incorrect. It must have the same height and width as the "
                  "output image. The output image has shape: %s and the style weight mask has shape: %s"
                  % (str((height, width)), str(content_img_style_weight_mask.shape)))
            raise AssertionError
        if content_img_style_weight_mask.dtype!=np.float32:
            print('The dtype of style_weight_mask must be float32. it is now %s' % str(content_img_style_weight_mask.dtype))
            raise AssertionError

    input_shape = (1, height, width, 3)
    print('The input shape is: %s' % (str(input_shape)))
    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles] #  TODO: Maybe I should delete this. For now I assume that this size is the same as the input_shape.

    content_features = {}
    style_features = [{} for _ in styles]
    output_semantic_mask_features = {}

    # Read the vgg net
    vgg_data, mean_pixel = vgg.read_net(path_to_network)
    print('Finished loading VGG.')

    if not do_restore_and_generate:
        # # Compute style features in feedforward mode.
        for i in range(len(styles)):
            style_features[i] = precompute_image_features(styles[i], STYLE_LAYERS, style_shapes[i], vgg_data, mean_pixel, use_mrf, use_semantic_masks)
        print('Finished loading VGG and passing content and style image to it.')

    # Define tensorflow placeholders and variables.
    with tf.Graph().as_default():
        if use_johnson:
            if use_semantic_masks:
                inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers])
                # inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers + NUM_NOISE_LAYERS])
                # inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers * NUM_NOISE_LAYERS])
            else:
                # Else, the input is the content images.
                inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
            if content_img_style_weight_mask is not None:
                content_img_style_weight_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1], name='content_img_style_weight_mask')
                input_concatenated = neural_util.add_content_img_style_weight_mask_to_input(inputs, content_img_style_weight_mask_placeholder)
                image = johnson_feedforward_net_util.net(input_concatenated)
            else:
                image = johnson_feedforward_net_util.net(inputs)  # Deleting the  / 255.0 because the network normalizes automatically.
        elif use_skip_noise_4:
            if use_semantic_masks:
                # inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers])
                inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers + NUM_NOISE_LAYERS * semantic_masks_num_layers + NUM_NOISE_LAYERS])
            else:
                inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
            if content_img_style_weight_mask is not None:
                content_img_style_weight_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1], name='content_img_style_weight_mask')
                input_concatenated = neural_util.add_content_img_style_weight_mask_to_input(inputs, content_img_style_weight_mask_placeholder)
                image, skip_noise_list = skip_noise_4_feedforward_net.net(input_concatenated)
            else:
                image, skip_noise_list = skip_noise_4_feedforward_net.net(inputs)
        else:
            raise AssertionError
        # To my understanding, preprocessing the images generated can make sure that their gram matrices will look
        # similar to the preprocessed content/style images. The image generated is in the normal rgb, not the
        # preprocessed/shifted version. Same reason applies to the other generator network below.
        image = vgg.preprocess(image, mean_pixel)

        # Feed the generated images, content images, and style images to vgg network and get each layers' activations.
        net = vgg.pre_read_net(vgg_data, image)
        net_layer_sizes = vgg.get_net_layer_sizes(net)
        if not do_restore_and_generate:
            learning_rate_decayed_init = tf.constant(learning_rate)
            learning_rate_decayed = tf.get_variable(name='learning_rate_decayed', trainable=False,
                                                    initializer=learning_rate_decayed_init)
            # compute content features in feedforward mode
            content_images = tf.placeholder(tf.float32, [batch_size, input_shape[1], input_shape[2], 3],
                                            name='content_images_placeholder')
            content_pre = vgg.preprocess(content_images, mean_pixel)
            content_net = vgg.pre_read_net(vgg_data, content_pre)
            content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

            if use_semantic_masks:
                output_semantic_mask_features, style_features, content_semantic_mask, style_semantic_masks_images = neural_doodle_util.construct_masks_and_features(style_semantic_masks, styles, style_features, batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers, STYLE_LAYERS, net_layer_sizes, semantic_masks_weight, vgg_data, mean_pixel, mask_resize_as_feature, use_mrf)


            # content loss
            _, height, width, number = map(lambda i: i.value, content_features[CONTENT_LAYER].get_shape())
            content_features_size = batch_size * height * width * number
            content_loss = content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                             content_features_size)

            if content_img_style_weight_mask is not None:
                # *** TESTING
                # Compute style weight masks for each feature layer in vgg.
                style_weight_mask_layer_dict = neural_doodle_util.masks_average_pool(content_img_style_weight_mask_placeholder)
                # *** END TESTING

            # style loss
            style_loss_for_each_style = []
            for i in range(len(styles)):
                style_losses_for_each_style_layer = []
                for style_layer in STYLE_LAYERS:
                    layer = net[style_layer]
                    if content_img_style_weight_mask is not None:
                        # Apply_style_weight_mask_to_feature_layer, then normalize with average of that style weight mask.
                        layer = neural_doodle_util.vgg_layer_dot_mask(style_weight_mask_layer_dict[style_layer], layer) \
                                / (tf.reduce_mean(style_weight_mask_layer_dict[style_layer]) + 0.000001)
                    if use_mrf:
                        if use_semantic_masks:
                            # If we use mrf for the style loss, we concatenate the mask layer to the features and
                            # essentially just treat it as another addditional feature that we added.
                            layer = neural_doodle_util.concatenate_mask_layer_tf(
                                output_semantic_mask_features[style_layer], layer)
                        print('mrfing %d %s' % (i, style_layer))
                        style_losses_for_each_style_layer.append(
                            mrf_loss(style_features[i][style_layer], layer, name='%d%s' % (i, style_layer)))
                        print('mrfed %d %s' % (i, style_layer))
                    else:
                        if use_semantic_masks:
                            gram = neural_doodle_util.gramian_with_mask(layer, output_semantic_mask_features[style_layer])

                            # layer = neural_doodle_util.vgg_layer_dot_mask(output_semantic_mask_features[style_layer],
                            #                                               layer)
                            # gram = gramian(layer)
                        else:
                            # Use gramian loss.
                            # TODO: testing gram stacks
                            gram = gramian(layer)
                            # If we want to use gram stacks instead of simple gram, uncomment the line below.
                            # gram = neural_util.gram_stacks(layer)
                        style_gram = style_features[i][style_layer]
                        if use_semantic_masks:
                            # Dividing by semantic_masks_num_layers because the masks should have one 1 in each pixel
                            # and we should not divide by the number of extra elements with 0.
                            style_gram_num_elements = neural_util.get_tensor_num_elements(style_gram) / semantic_masks_num_layers
                        else:
                            style_gram_num_elements = get_np_array_num_elements(style_gram)
                        style_losses_for_each_style_layer.append(
                            2 * tf.nn.l2_loss(gram - style_gram) / style_gram_num_elements)

                style_loss_for_each_style.append(
                    style_weight * style_blend_weights[i] * reduce(tf.add,
                                                                   style_losses_for_each_style_layer) / batch_size)
            # According to https://arxiv.org/abs/1610.07629 when "zero-padding is replaced with mirror-padding,
            # and transposed convolutions (also sometimes called deconvolutions) are replaced with nearest-neighbor
            # upsampling followed by a convolution.", tv is no longer needed.
            # But in other papers I've seen tv-loss still applicable, like in https://arxiv.org/abs/1603.08155.
            # TODO: side task: find out the difference between having tv loss and not.
            # tv_loss = 0
            tv_loss = tv_weight * total_variation(image)

            # overall loss
            if style_only or use_semantic_masks:
                losses_for_each_style = [style_loss + tv_loss for style_loss in style_loss_for_each_style]
            else:
                losses_for_each_style = [style_loss + content_loss + tv_loss for style_loss in
                                         style_loss_for_each_style]
            overall_loss = 0
            for loss_for_each_style in losses_for_each_style:
                overall_loss += loss_for_each_style
            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            # TODO:  side task: tell which one is better, training all variables or training only scale and offset.
            if multiple_styles_train_scale_offset_only:
                if use_johnson:
                    scale_offset_var = johnson_feedforward_net_util.get_johnson_scale_offset_var()
                elif use_skip_noise_4:
                    raise NotImplementedError
                else:
                    raise AssertionError
                train_step_for_each_style = [
                    tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                           beta2=0.999).minimize(loss,
                                                                 var_list=scale_offset_var)
                    if i != 0 else
                    tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                           beta2=0.999).minimize(loss)
                    for i, loss in
                    enumerate(losses_for_each_style)]
            else:
                train_step_for_each_style = [
                    tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                           beta2=0.999).minimize(loss)
                    for i, loss in
                    enumerate(losses_for_each_style)]

            def print_progress(i, feed_dict, last=False):
                stderr.write(
                    'Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    stderr.write('Learning rate %f\n' % (learning_rate_decayed.eval()))
                    # Assume that the feed_dict is for the last content and style.
                    if not (style_only or use_semantic_masks):
                        stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                    stderr.write('    style loss: %g\n' % style_loss_for_each_style[-1].eval(feed_dict=feed_dict))
                    stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                    stderr.write('    total loss: %g\n' % overall_loss.eval(feed_dict=feed_dict))

        # Optimization
        # It used to track and record only the best one with lowest loss. This is no longer necessary and I think
        # just recording the one generated at each round will make it easier to debug.
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
                    # This is the x and y offset, the coordinate where we start capturing screen shot.
                    kScreenX = 300
                    kScreenY = 300
                elif from_webcam:
                    cap = cv2.VideoCapture(0)
                    # Set width and height.
                    ret = cap.set(3, 1280)
                    ret = cap.set(4, 960)
                    ret, frame = cap.read()
                    print('The dimension of this camera is : %d x %d' % (frame.shape[1], frame.shape[0]))
                else:
                    assert test_img_dir is not None

                if use_johnson:
                    if use_semantic_masks:
                        inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2],
                                                                   semantic_masks_num_layers])
                        #
                        # inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2],
                        #                                            semantic_masks_num_layers + NUM_NOISE_LAYERS])

                        # inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2],
                        #                                            semantic_masks_num_layers * NUM_NOISE_LAYERS])
                    else:
                        inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])

                    if content_img_style_weight_mask is not None:
                        content_img_style_weight_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1], name='content_img_style_weight_mask')
                        input_concatenated = neural_util.add_content_img_style_weight_mask_to_input(inputs, content_img_style_weight_mask_placeholder)
                        image = johnson_feedforward_net_util.net(input_concatenated, reuse=True)
                    else:
                        image = johnson_feedforward_net_util.net(inputs, reuse=True)
                elif use_skip_noise_4:
                    if use_semantic_masks:
                        inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], semantic_masks_num_layers + NUM_NOISE_LAYERS * semantic_masks_num_layers + NUM_NOISE_LAYERS])
                    else:
                        inputs = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 3])
                    if content_img_style_weight_mask is not None:
                        content_img_style_weight_mask_placeholder = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1], name='content_img_style_weight_mask')
                        input_concatenated = neural_util.add_content_img_style_weight_mask_to_input(inputs, content_img_style_weight_mask_placeholder)
                        image, skip_noise_list = skip_noise_4_feedforward_net.net(input_concatenated, reuse=True)
                    else:
                        image, skip_noise_list = skip_noise_4_feedforward_net.net(inputs, reuse=True)
                else:
                    raise AssertionError
                image = vgg.preprocess(image, mean_pixel)
                iterator = 0

                while from_screenshot or from_webcam or (iterator == 0):
                    if from_screenshot:
                        pass
                        # w = gtk.gdk.get_default_root_window()
                        # sz = w.get_size()
                        # print "The size of the window is %d x %d" % sz
                        # pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, input_shape[1], input_shape[2])
                        # pb = pb.get_from_drawable(w, w.get_colormap(), kScreenX, kScreenY, 0, 0, input_shape[1],
                        #                           input_shape[2])
                        # content_image = pb.pixel_array
                    elif from_webcam:
                        ret, frame = cap.read()
                        content_image = scipy.misc.imresize(frame, (input_shape[1], input_shape[2]))
                    elif use_semantic_masks:
                        # Dummy content image
                        content_image = np.zeros((batch_size, input_shape[1], input_shape[2], 3))
                    else:
                        content_image = imread(test_img_dir, (input_shape[1], input_shape[2]))

                    content_pre = np.array([vgg.preprocess(content_image, mean_pixel)])
                    # Now generate an image using the style_blend_weights given.
                    feed_dict = {}

                    if use_semantic_masks:
                        # read semantic masks
                        mask_dirs = get_all_image_paths_in_dir(test_img_dir)

                        if not len(mask_dirs) >= (batch_size * semantic_masks_num_layers):
                            print('ERROR: The number of images in mask_dirs has to be larger than batch size times '
                                  'number of semantic masks. Path to test_img_dir is: %s. '
                                  'number of images in mask dirs is : %d' % (test_img_dir, len(mask_dirs)))
                            raise AssertionError
                        if len(mask_dirs) > (batch_size * semantic_masks_num_layers):
                            mask_dirs = mask_dirs[:batch_size * semantic_masks_num_layers]

                        for semantic_masks_i in range(semantic_masks_num_layers):
                            expected_end_str=  '%d.png' %semantic_masks_i
                            if mask_dirs[semantic_masks_i][-5:] != expected_end_str:
                                print('%s did not end with %s' %(mask_dirs[semantic_masks_i],expected_end_str))
                                raise AssertionError
                            if semantic_masks_i != 0 and mask_dirs[semantic_masks_i - 1][:-5] != mask_dirs[semantic_masks_i][:-5]:
                                print('%s did not start with %s' %(mask_dirs[semantic_masks_i],mask_dirs[semantic_masks_i - 1][:-5]))
                                raise AssertionError

                        mask_pre_list = read_and_resize_bw_mask_images(mask_dirs, input_shape[1], input_shape[2],
                                                                       batch_size, semantic_masks_num_layers)

                    if use_johnson:
                        if use_semantic_masks:
                            feed_dict[inputs] = mask_pre_list
                            # feed_dict[inputs] = np.concatenate((np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], NUM_NOISE_LAYERS)), mask_pre_list), axis=3)
                            # feed_dict[inputs] = np_image_dot_mask(mask_pre_list, np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], NUM_NOISE_LAYERS)))


                        elif style_only:
                            feed_dict[inputs] = np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
                        else:
                            feed_dict[inputs] = content_pre
                    elif use_skip_noise_4:
                        if use_semantic_masks:
                            feed_dict[inputs] = mask_pre_list
                        elif style_only:
                            feed_dict[inputs] = np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
                        else:
                            feed_dict[inputs] = content_pre
                        for noise_i, skip_noise in enumerate(skip_noise_list):
                            skip_noise_shape = map(lambda i: i.value, skip_noise.get_shape())


                            feed_dict[skip_noise] = np.random.uniform(
                                size=(skip_noise_shape[0], skip_noise_shape[1], skip_noise_shape[2], skip_noise_4_feedforward_net.nums_noise[noise_i]))
                    else:
                        raise AssertionError

                    if content_img_style_weight_mask is not None:
                        feed_dict[content_img_style_weight_mask_placeholder] = content_img_style_weight_mask
                    generated_image = image.eval(feed_dict=feed_dict)
                    iterator += 1
                    # Can't return because we are in a generator.
                    # yield (iterator, vgg.unprocess(
                    #     scipy.misc.imresize(generated_image[0, :, :, :], (input_shape[1], input_shape[2])), mean_pixel))
                    # No need to unprocess it because we've preprocessed the generated image in the network. That means
                    # the generated image is before preprocessing.
                    yield (iterator, scipy.misc.imresize(generated_image[0, :, :, :], (input_shape[1], input_shape[2])))

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
                content_dirs = get_all_image_paths_in_dir(content_folder)
                # Ignore the ones at the end.
                if batch_size != 1:
                    content_dirs = content_dirs[:-(len(content_dirs) % batch_size)]

                if use_semantic_masks:
                    # Get path to all mask images.
                    mask_dirs = get_all_image_paths_in_dir(mask_folder)
                    if len(mask_dirs) < batch_size * semantic_masks_num_layers:
                        print('ERROR: The number of images in mask_folder has to be larger than batch size times '
                              'number of semantic masks. Path to mask_folder is: %s. '
                              'number of images in mask dirs is : %d' % (mask_folder, len(mask_dirs)))
                        raise AssertionError
                    # Ignore the ones at the end.
                    if batch_size * semantic_masks_num_layers != 1 and len(mask_dirs) % (
                        batch_size * semantic_masks_num_layers) != 0:
                        mask_dirs = mask_dirs[:-(len(mask_dirs) % (batch_size * semantic_masks_num_layers))]

                for i in range(iter_start, iterations):
                    # First decay the learning rate if we need to
                    if (i % lr_decay_steps == 0):
                        current_lr = learning_rate_decayed.eval()
                        sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))

                    if not style_only:
                        # Load content images
                        current_content_dirs = get_batch(content_dirs, i * batch_size, batch_size)
                        content_pre_list = read_and_resize_batch_images(current_content_dirs, input_shape[1],
                                                                        input_shape[2])

                    # Load mask images
                    if use_semantic_masks:
                        current_mask_dirs = get_batch(mask_dirs, i * batch_size * semantic_masks_num_layers,
                                                      batch_size * semantic_masks_num_layers)
                        # DEBUG
                        for semantic_masks_i in range(semantic_masks_num_layers):
                            expected_end_str=  '%d.png' %semantic_masks_i
                            if current_mask_dirs[semantic_masks_i][-5:] != expected_end_str:
                                print('%s did not end with %s' %(current_mask_dirs[semantic_masks_i],expected_end_str))
                                raise AssertionError
                            if semantic_masks_i != 0 and current_mask_dirs[semantic_masks_i - 1][:-5] != current_mask_dirs[semantic_masks_i][:-5]:
                                print('%s did not start with %s' %(current_mask_dirs[semantic_masks_i],current_mask_dirs[semantic_masks_i - 1][:-5]))
                                raise AssertionError

                        mask_pre_list = read_and_resize_bw_mask_images(current_mask_dirs, input_shape[1],
                                                                       input_shape[2], batch_size,
                                                                       semantic_masks_num_layers)

                    for style_i in range(len(styles)):
                        last_step = (i == iterations - 1)
                        # Feed the content image.
                        feed_dict = {content_images: content_pre_list} if not style_only else {}
                        if use_johnson:
                            if use_semantic_masks:
                                feed_dict[inputs] = mask_pre_list
                                # feed_dict[inputs] = np.concatenate((np.random.uniform(size=(input_shape[0], input_shape[1],
                                #                                                    input_shape[2], NUM_NOISE_LAYERS)),
                                #                                     mask_pre_list), axis=3)
                                # feed_dict[inputs] = np_image_dot_mask(mask_pre_list,np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], NUM_NOISE_LAYERS)))

                                feed_dict[content_semantic_mask] = mask_pre_list
                                for styles_iter in range(len(styles)):
                                    feed_dict[style_semantic_masks_images[styles_iter]] = np.expand_dims(
                                        style_semantic_masks[styles_iter], axis=0)
                            else:
                                if style_only:
                                    feed_dict[inputs] = np.random.uniform(size=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
                                else:
                                    feed_dict[inputs] = content_pre_list
                        elif use_skip_noise_4:
                            if use_semantic_masks:
                                # Note: the following comment may not be directly related to the code. Please ignore
                                # this unless you want to find out where I get the skip_noise_4 generator network.
                                # According to github.com/DmitryUlyanov/online-neural-doodle/blob/master/src/utils.lua
                                # The first # semantic_masks_num_layers layers will be filled with the mask itself
                                # The second # semantic_masks_num_layers*num_mask_noise_times will be filled with mask dot
                                # uniform noise
                                # And the last # num_mask_noise_times layers will be filled with uniform noise.
                                # Then I realized that although that git repo implemented this feature, it did not
                                # actually used it.
                                feed_dict[inputs] = mask_pre_list
                                feed_dict[content_semantic_mask] = mask_pre_list
                                for styles_iter in range(len(styles)):
                                    feed_dict[style_semantic_masks_images[styles_iter]] = np.expand_dims(
                                        style_semantic_masks[styles_iter], axis=0)
                            elif style_only:
                                feed_dict[inputs] = np.random.uniform(
                                    size=(input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
                            else:
                                feed_dict[inputs] = content_pre_list
                            for noise_i, skip_noise in enumerate(skip_noise_list):
                                skip_noise_shape = map(lambda i: i.value, skip_noise.get_shape())
                                feed_dict[skip_noise] = np.random.uniform(size=(skip_noise_shape[0], skip_noise_shape[1], skip_noise_shape[2], skip_noise_4_feedforward_net.nums_noise[noise_i]))
                        else:
                            raise NotImplementedError

                        if content_img_style_weight_mask is not None:
                            # TODO: check if we can just feed in completely random tensor. because in real life
                            # the tensor should be continuous, more like the heat map. So maybe I should use the
                            # Diamond generator again, but with more output than just 0 and 255.
                            content_img_style_weight_mask_shape = map(lambda s: s.value, content_img_style_weight_mask_placeholder.get_shape())
                            style_weight_mask_for_training_shape = style_weight_mask_for_training.shape
                            if content_img_style_weight_mask_shape[1] != style_weight_mask_for_training_shape[1] or content_img_style_weight_mask_shape[2] != style_weight_mask_for_training_shape[2]:
                                print("The training masks' shape does not correspond with the place holder's shape. The training mask shape is: %s and the place holder shape is: %s. They should have the same height and width." %(str(style_weight_mask_for_training_shape), str(content_img_style_weight_mask_shape)))
                            content_img_style_weight_mask_batch_i = get_batch_indices(style_weight_mask_for_training_shape[0], i * batch_size, batch_size)
                            feed_dict[content_img_style_weight_mask_placeholder] = style_weight_mask_for_training[content_img_style_weight_mask_batch_i, :, :, :]

                        train_step_for_each_style[style_i].run(feed_dict=feed_dict)
                        if style_i == len(styles) - 1:
                            print_progress(i, feed_dict=feed_dict, last=last_step)
                            # Record loss after each training round.
                            with open(save_dir + 'loss.tsv','a') as loss_record_file:
                                loss_record_file.write('%d\t%g\n' % (i, overall_loss.eval(feed_dict=feed_dict)))

                        if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                            if style_i == len(styles) - 1:
                                saver.save(sess, save_dir + 'model.ckpt', global_step=i)

                                if test_img_dir is not None:
                                    if use_semantic_masks:
                                        test_mask_dirs = get_all_image_paths_in_dir(test_img_dir)
                                        test_image = imread(test_mask_dirs[0])
                                        test_image_shape = test_image.shape
                                    else:
                                        test_image = imread(test_img_dir)
                                        test_image_shape = test_image.shape
                                    # The for loop will run once and terminate. Can't use return and yield in the same function so this is a hacky way to do it.
                                    for _, generated_image in style_synthesis_net(path_to_network, test_image_shape[0],
                                                                                  test_image_shape[1], styles,
                                                                                  iterations,
                                                                                  1,
                                                                                  content_weight, style_weight,
                                                                                  style_blend_weights, tv_weight,
                                                                                  learning_rate,
                                                                                  style_only=style_only,
                                                                                  multiple_styles_train_scale_offset_only=multiple_styles_train_scale_offset_only,
                                                                                  use_mrf=use_mrf,
                                                                                  use_johnson=use_johnson,
                                                                                  use_skip_noise_4=use_skip_noise_4,
                                                                                  save_dir=save_dir,
                                                                                  do_restore_and_generate=True,
                                                                                  do_restore_and_train=False,
                                                                                  from_screenshot=False,
                                                                                  from_webcam=False,
                                                                                  use_semantic_masks=use_semantic_masks,
                                                                                  mask_folder=mask_folder,
                                                                                  mask_resize_as_feature=mask_resize_as_feature,
                                                                                  style_semantic_masks=style_semantic_masks,
                                                                                  semantic_masks_weight=semantic_masks_weight,
                                                                                  semantic_masks_num_layers=semantic_masks_num_layers,
                                                                                  test_img_dir=test_img_dir,
                                                                                  ablation_layer=ablation_layer,
                                                                                  content_img_style_weight_mask=content_img_style_weight_mask):
                                        pass

                                    best_for_each_style[style_i] = generated_image

                                # Because we now have batch, choose the first one in the batch as our sample image.
                                yield (
                                    (None if last_step else i),
                                    [None if test_img_dir is None else
                                     best_for_each_style[style_i] for style_i in range(len(styles))]
                                )
