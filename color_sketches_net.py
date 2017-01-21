"""
This file implements a network that can learn to color sketches.
I first saw it at http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
"""

# import gtk.gdk
from sys import stderr

import cv2
import scipy
import tensorflow as tf

import adv_net_util
import colorful_img_network_connected_util
import colorful_img_network_mod_util
import colorful_img_network_util
import johnson_feedforward_net_util
import sketches_util
import unet_mod_util
import unet_util
from general_util import *


try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter


COLORFUL_IMG_NUM_BIN = 6  # Temporary

# TODO: change rtype
def color_sketches_net(height, width, iterations, batch_size, content_weight, tv_weight,
                       learning_rate, generator_network='unet',
                       use_adversarial_net = False, use_hint = False,
                       adv_net_weight = 10000000.0,lr_decay_steps=10000,
                       min_lr=0.00003, lr_decay_rate=0.9,print_iterations=None,
                       checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                       do_restore_and_train=False, restore_from_noadv_to_adv = False, content_folder=None,
                       content_preprocessed_folder = None,
                       from_screenshot=False, from_webcam=False, test_img_dir=None, test_img_hint=None):
    """
    Stylize images.
    TODO: modify the description.

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
    if use_hint:
        assert test_img_hint is not None

    input_shape = (1, height, width, 3)
    print('The input shape is: %s. Using %s generator network' % (str(input_shape), generator_network))

    if generator_network == 'colorful_img' or generator_network =='backprop' or generator_network == 'unet_mod':
        img_to_rgb_bin_encoder=colorful_img_network_util.ImgToRgbBinEncoder(COLORFUL_IMG_NUM_BIN)

    content_img_preprocessed = None
    sketches_preprocessed = None
    prev_content_preprocessed_file_i = 0

    # Define tensorflow placeholders and variables.
    with tf.Graph().as_default():

        input_sketches = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 1], name='input_sketches')
        if use_hint:
            input_hint = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 4], name='input_hint')
            input_concatenated = tf.concat(3, (input_sketches, input_hint))
            if generator_network == 'unet':
                generator_output = unet_util.net(input_concatenated)
            elif generator_network == 'unet_mod':
                generator_output = unet_mod_util.net(input_concatenated)
            elif generator_network == 'johnson':
                generator_output = johnson_feedforward_net_util.net(input_concatenated)
            elif generator_network == 'colorful_img':
                generator_output = colorful_img_network_util.net(input_concatenated)
            elif generator_network == 'colorful_img_mod':
                generator_output = colorful_img_network_mod_util.net(input_concatenated)
            elif generator_network == 'colorful_img_connected':
                generator_output = colorful_img_network_connected_util.net(input_concatenated)
            elif generator_network =='backprop':
                generator_output = tf.get_variable('backprop_input_var',
                                                   shape=[batch_size, input_shape[1], input_shape[2], COLORFUL_IMG_NUM_BIN**3],
                                                   initializer=tf.random_normal_initializer()) + 0 * input_concatenated
            else:
                raise AssertionError("Please input a valid generator network name. Either unet or johnson. Got: %s"
                                     % (generator_network))

        else:
            if generator_network == 'unet':
                generator_output = unet_util.net(input_sketches)
            elif generator_network == 'unet_mod':
                generator_output = unet_mod_util.net(input_sketches)
            elif generator_network == 'johnson':
                generator_output = johnson_feedforward_net_util.net(input_sketches)
            elif generator_network == 'colorful_img':
                generator_output = colorful_img_network_util.net(input_sketches)
            elif generator_network == 'colorful_img_mod':
                generator_output = colorful_img_network_mod_util.net(input_sketches)
            elif generator_network == 'colorful_img_connected':
                generator_output = colorful_img_network_connected_util.net(input_sketches)
            elif generator_network =='backprop':
                generator_output = tf.get_variable('backprop_input_var',
                                                   shape=[batch_size, input_shape[1], input_shape[2], COLORFUL_IMG_NUM_BIN**3],
                                                   initializer=tf.random_normal_initializer()) + 0 * input_sketches
                generator_output = tf.nn.softmax(generator_output)
            else:
                raise AssertionError("Please input a valid generator network name. Either unet or johnson")



        if not do_restore_and_generate:
            learning_rate_decayed_init = tf.constant(learning_rate)
            learning_rate_decayed = tf.get_variable(name='learning_rate_decayed', trainable=False,
                                                    initializer=learning_rate_decayed_init)
            if generator_network == 'colorful_img' or generator_network =='backprop' or generator_network == 'unet_mod':
                expected_output = tf.placeholder(tf.float32,
                                                 shape=[batch_size, input_shape[1], input_shape[2], COLORFUL_IMG_NUM_BIN**3],
                                                 name='expected_output')
                generator_loss_non_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(generator_output, expected_output))
            else:
                expected_output = tf.placeholder(tf.float32,
                                                 shape=[batch_size, input_shape[1], input_shape[2], 3],
                                                 name='expected_output')
                generator_loss_non_adv = tf.nn.l2_loss(generator_output - expected_output) / batch_size
            # tv_loss = tv_weight * total_variation(image)

            if use_adversarial_net:
                adv_net_input = tf.placeholder(tf.float32,
                                                 shape=[batch_size, input_shape[1], input_shape[2], 3], name='adv_net_input')
                adv_net_prediction_image_input = adv_net_util.net(adv_net_input)
                adv_net_prediction_generator_input = adv_net_util.net(generator_output, reuse=True)
                adv_net_all_var = adv_net_util.get_net_all_variables()

                if generator_network == 'unet':
                    generator_all_var = unet_util.get_net_all_variables()
                elif generator_network == 'johnson':
                    generator_all_var = johnson_feedforward_net_util.get_net_all_variables()
                else:
                    raise AssertionError("Please input a valid generator network name. Either unet or johnson")

                logits_from_i = adv_net_prediction_image_input
                logits_from_g = adv_net_prediction_generator_input

                # One represent labeling the image as coming from real image. Zero represent labeling it as generated.
                adv_loss_from_i = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
                adv_loss_from_g = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([batch_size], dtype=tf.int64))) * adv_net_weight

                adv_loss =  adv_loss_from_i + adv_loss_from_g
                generator_loss_through_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
                # Beta1 = 0.5 according to dcgan paper
                adv_train_step = tf.train.AdamOptimizer(learning_rate_decayed * 0.05, beta1=0.5,
                                       beta2=0.999).minimize(adv_loss, var_list=adv_net_all_var)
                adv_train_step_i = tf.train.AdamOptimizer(learning_rate_decayed * 0.05, beta1=0.5,
                                       beta2=0.999).minimize(adv_loss_from_i, var_list=adv_net_all_var)
                adv_train_step_g = tf.train.AdamOptimizer(learning_rate_decayed * 0.05, beta1=0.5,
                                       beta2=0.999).minimize(adv_loss_from_g, var_list=adv_net_all_var)
                generator_train_step_through_adv = tf.train.AdamOptimizer(learning_rate_decayed * 0.05, beta1=0.5,
                                       beta2=0.999).minimize(generator_loss_through_adv, var_list=generator_all_var)
                generator_train_step = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                       beta2=0.999).minimize(generator_loss_non_adv)

                with tf.control_dependencies([generator_train_step_through_adv, generator_train_step]):
                    generator_both_train = tf.no_op(name='generator_both_train')


                adv_loss_real_sum = scalar_summary("adv_loss_real", adv_loss_from_i)
                adv_loss_fake_sum = scalar_summary("adv_loss_fake", adv_loss_from_g)

                generator_loss_through_adv_sum = scalar_summary("g_loss_through_adv", generator_loss_through_adv)
                adv_loss_sum = scalar_summary("adv_loss", adv_loss)
                generator_loss_l2_sum = scalar_summary("generator_loss_non_adv", generator_loss_non_adv)


                g_sum = merge_summary([generator_loss_through_adv_sum, generator_loss_l2_sum])
                adv_sum = merge_summary([adv_loss_fake_sum, adv_loss_real_sum, adv_loss_sum])
            else:
                # optimizer setup
                # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
                generator_train_step = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                       beta2=0.999).minimize(generator_loss_non_adv)
                g_loss_sum = scalar_summary("generator_loss_non_adv", generator_loss_non_adv)


            def print_progress(i, feed_dict, adv_feed_dict, last=False):
                stderr.write(
                    'Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    stderr.write('Learning rate %f\n' % (learning_rate_decayed.eval()))
                    stderr.write(' generator l2 loss: %g\n' % generator_loss_non_adv.eval(feed_dict=feed_dict))
                    if use_adversarial_net:
                        stderr.write('   adv_from_i loss: %g\n' % adv_loss_from_i.eval(feed_dict=adv_feed_dict))
                        stderr.write('   adv_from_g loss: %g\n' % adv_loss_from_g.eval(feed_dict=adv_feed_dict))
                        stderr.write('generator adv loss: %g\n' % generator_loss_through_adv.eval(feed_dict=adv_feed_dict))


        # Optimization
        # It used to track and record only the best one with lowest loss. This is no longer necessary and I think
        # just recording the one generated at each round will make it easier to debug.
        best_image = None
        if restore_from_noadv_to_adv and use_adversarial_net:
            saver = tf.train.Saver(generator_all_var + [learning_rate_decayed])
        else:
            saver = tf.train.Saver()
        with tf.Session() as sess:
            if do_restore_and_generate:
                assert batch_size == 1
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
                    else:
                        content_image = imread(test_img_dir, (input_shape[1], input_shape[2]))
                    content_image = np.array([content_image])
                    image_sketches = sketches_util.image_to_sketch(content_image)
                    image_sketches = np.expand_dims(image_sketches, axis=3)

                    # Now generate an image using the style_blend_weights given.
                    feed_dict = {input_sketches:image_sketches}

                    if use_hint:
                        image_hint = imread(test_img_hint, (input_shape[1], input_shape[2]), rgba=True)
                        feed_dict[input_hint] = np.array([image_hint])

                    generated_image = generator_output.eval(feed_dict=feed_dict)
                    iterator += 1
                    # Can't return because we are in a generator.
                    # yield (iterator, vgg.unprocess(
                    #     scipy.misc.imresize(generated_image[0, :, :, :], (input_shape[1], input_shape[2])), mean_pixel))
                    # No need to unprocess it because we've preprocessed the generated image in the network. That means
                    # the generated image is before preprocessing.
                    yield (iterator, generated_image)

            else:
                # Initialize log writer
                summary_writer = SummaryWriter("./logs", sess.graph)

                # initialize pre-processsed numpy array
                if content_preprocessed_folder is not None:
                    if not os.path.isfile(content_preprocessed_folder + 'record.txt'):
                        raise AssertionError(
                            'No preprocessed content images found in %s. To use this feature, first use some '
                            'other file to call read_resize_and_save_all_imgs_in_dir.'
                            % (content_preprocessed_folder))
                    content_preprocessed_record = sketches_util.read_preprocessed_sketches_npy_record(content_preprocessed_folder)
                    if content_preprocessed_record[0][2] % batch_size != 0 \
                            or content_preprocessed_record[0][3] != height\
                            or content_preprocessed_record[0][4] != width:
                        raise AssertionError(
                            'The height, width, and batch size of the preprocessed numpy files does not '
                            'match those of the current setting.')
                    # Read the first file
                    print('Reading preprocessed content images.')
                    content_img_preprocessed = np.load(content_preprocessed_record[prev_content_preprocessed_file_i][0])
                    sketches_preprocessed = np.load(content_preprocessed_record[prev_content_preprocessed_file_i][1])

                with open(save_dir + 'loss.tsv', 'w') as loss_record_file:
                    pass  # Clear the loss file before appending to it.
                if use_adversarial_net:
                    with open(save_dir + 'adv_loss.tsv', 'w') as loss_record_file:
                        loss_record_file.write('i\tcurrent_generator_l2_loss\tcurrent_adv_loss_i\tcurrent_adv_loss_g\tcurrent_gen_loss_through_adv\n')
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
                    if restore_from_noadv_to_adv and use_adversarial_net:
                        # Simply running this doesn;t seem to work.
                        # sess.run(tf.initialize_variables(adv_net_all_var))

                        # Get all variables except the generator net and the learning rate
                        if '0.12.0' in tf.__version__:
                            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                        else:
                            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
                        var_not_saved = [item for item in all_vars if item not in (generator_all_var + [learning_rate_decayed])]
                        sess.run(tf.initialize_variables(var_not_saved))
                        # Now change the saver back to normal
                        saver = tf.train.Saver()
                else:
                    sess.run(tf.initialize_all_variables())

                # Get path to all content images.
                content_dirs = get_all_image_paths_in_dir(content_folder)
                # Ignore the ones at the end.
                if batch_size != 1 and len(content_dirs) % batch_size != 0:
                    content_dirs = content_dirs[:-(len(content_dirs) % batch_size)]
                print('The size of training dataset is %d images.' % len(content_dirs))

                # # Test training GAN differently***
                # generators_turn = True
                # # END TEST***


                for i in range(iter_start, iterations):
                    # First decay the learning rate if we need to
                    if (i % lr_decay_steps == 0 and i!= iter_start):
                        current_lr = learning_rate_decayed.eval()
                        sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))
                    if content_preprocessed_folder is not None:
                        current_content_preprocessed_file_i, index_within_preprocessed =  \
                            sketches_util.find_corresponding_sketches_npy_from_record(
                            content_preprocessed_record, i * batch_size)
                        if prev_content_preprocessed_file_i != current_content_preprocessed_file_i:
                            prev_content_preprocessed_file_i = current_content_preprocessed_file_i
                            content_img_preprocessed = np.load(content_preprocessed_record[
                                                                   current_content_preprocessed_file_i][0])
                            sketches_preprocessed = np.load(content_preprocessed_record[
                                                                   current_content_preprocessed_file_i][1])
                        content_pre_list = content_img_preprocessed[
                                           index_within_preprocessed:index_within_preprocessed+batch_size,
                                           ...].astype(np.float32)
                        image_sketches = sketches_preprocessed[
                                           index_within_preprocessed:index_within_preprocessed+batch_size,
                                           ...].astype(np.float32)
                        image_sketches = np.expand_dims(image_sketches, axis=3)
                    else:

                        current_content_dirs = get_batch_paths(content_dirs, i * batch_size, batch_size)
                        content_pre_list = read_and_resize_batch_images(current_content_dirs, input_shape[1],
                                                                        input_shape[2])

                        image_sketches = sketches_util.image_to_sketch(content_pre_list)
                        image_sketches = np.expand_dims(image_sketches, axis=3)

                    # Now generate an image using the style_blend_weights given.
                    if generator_network == 'colorful_img' or generator_network =='backprop' or generator_network == 'unet_mod':
                        # Encoding image into rgb bins takes a while. It takes about 1s to convert one batch of 12
                        # images with shape 256x256 into rgb bins. It is not really possible to preprocess this step
                        # though since it will take up too much space. (Maybe not so much space if I do it smartly.)
                        # It's not a huge issue for now...
                        current_rgb_bin = img_to_rgb_bin_encoder.img_to_rgb_bin(content_pre_list)
                        feed_dict = {expected_output: current_rgb_bin, input_sketches: image_sketches}
                    else:
                        feed_dict = {expected_output:content_pre_list, input_sketches:image_sketches}

                    if use_hint:
                        image_hint = sketches_util.generate_hint_from_image(content_pre_list)
                        feed_dict[input_hint] = image_hint

                    last_step = (i == iterations - 1)


                    if use_adversarial_net:
                        adv_feed_dict = {expected_output:content_pre_list, input_sketches:image_sketches, adv_net_input: content_pre_list}
                        if use_hint:
                            adv_feed_dict[input_hint] = image_hint

                        # TEST printing before training
                        print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, last=last_step)

                        # if generators_turn:
                        #     # generator_train_step.run(feed_dict=feed_dict)
                        #     generator_train_step_through_adv.run(feed_dict=adv_feed_dict)
                        #     adv_train_step.run(feed_dict=adv_feed_dict)

                        # generator_train_step_through_adv.run(feed_dict=adv_feed_dict)
                        # adv_train_step.run(feed_dict=adv_feed_dict)


                        # Update D network
                        _, summary_str = sess.run([adv_train_step, adv_sum],
                                                       feed_dict=adv_feed_dict)
                        summary_writer.add_summary(summary_str, i)

                        # Update G network
                        # TODO: Try to update the generator network with loss both from adversarial net and from l2.
                        _, summary_str = sess.run([generator_both_train, g_sum],
                                                       feed_dict=adv_feed_dict)
                        summary_writer.add_summary(summary_str, i)

                        # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        # _, summary_str = sess.run([generator_train_step_through_adv, g_sum],
                        #                                feed_dict=adv_feed_dict)
                        # summary_writer.add_summary(summary_str, i)

                    else:
                        adv_feed_dict = None
                        print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, last=last_step)

                        _, summary_str = sess.run([generator_train_step, g_loss_sum], feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str,i)

                    # TEST printing after training
                    print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, last=last_step)

                    # # TODO:
                    # if i%100==0:
                    #     with open(save_dir + 'loss.tsv','a') as loss_record_file:
                    #         current_generator_l2_loss = generator_loss_non_adv.eval(feed_dict=feed_dict)
                    #         loss_record_file.write('%d\t%g\n' % (i, current_generator_l2_loss))
                    #     if use_adversarial_net:
                    #         with open(save_dir + 'adv_loss.tsv', 'a') as loss_record_file:
                    #             current_adv_loss_i = adv_loss_from_i.eval(feed_dict=adv_feed_dict)
                    #             current_adv_loss_g = adv_loss_from_g.eval(feed_dict=adv_feed_dict)
                    #             current_gen_loss_through_adv = generator_loss_through_adv.eval(feed_dict=adv_feed_dict)
                    #             loss_record_file.write('%d\t%g\t%g\t%g\t%g\n' % (i, current_generator_l2_loss, current_adv_loss_i, current_adv_loss_g, current_gen_loss_through_adv))


                    if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                        saver.save(sess, save_dir + 'model.ckpt', global_step=i)
                        print('Checkpoint saved.')

                        if test_img_dir is not None:
                            test_image = imread(test_img_dir)
                            test_image_shape = test_image.shape

                        # The for loop will run once and terminate. Can't use return and yield in the same function so this is a hacky way to do it.
                        for _, generated_image in color_sketches_net(test_image_shape[0],
                                                                      test_image_shape[1],
                                                                      iterations,
                                                                      1,
                                                                      content_weight, tv_weight,
                                                                      learning_rate,
                                                                      generator_network=generator_network,
                                                                      use_adversarial_net=False,  # use_adversarial_net=use_adversarial_net,
                                                                      use_hint=use_hint,
                                                                      save_dir=save_dir,
                                                                      do_restore_and_generate=True,
                                                                      do_restore_and_train=False,
                                                                      from_screenshot=False,
                                                                      from_webcam=False,
                                                                      test_img_dir=test_img_dir,
                                                                      test_img_hint=test_img_hint):
                            pass

                            if generator_network == 'colorful_img' or generator_network =='backprop' or generator_network == 'unet_mod':
                                best_image = img_to_rgb_bin_encoder.rgb_bin_to_img(generated_image)
                            else:
                                best_image = generated_image

                        # Because we now have batch, choose the first one in the batch as our sample image.
                        yield (
                            (None if last_step else i),
                            None if test_img_dir is None else
                             best_image
                        )
