"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417 and
https://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

# import gtk.gdk
from sys import stderr

import cv2
import scipy
import tensorflow as tf

import image_to_sketches_util
import unet_util
from general_util import *

# TODO: change rtype
def color_sketches_net(height, width, iterations, batch_size, content_weight, tv_weight,
                        learning_rate, lr_decay_steps=200, min_lr=0.001, lr_decay_rate=0.7,print_iterations=None,
                        checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                        do_restore_and_train=False, content_folder=None,
                        from_screenshot=False, from_webcam=False, test_img_dir=None):
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

    input_shape = (1, height, width, 3)
    print('The input shape is: %s' % (str(input_shape)))

    # Define tensorflow placeholders and variables.
    with tf.Graph().as_default():
        input_sketches = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 1])

        output = unet_util.net(input_sketches)
        expected_output = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 3])

        if not do_restore_and_generate:
            learning_rate_decayed_init = tf.constant(learning_rate)
            learning_rate_decayed = tf.get_variable(name='learning_rate_decayed', trainable=False,
                                                    initializer=learning_rate_decayed_init)

            content_loss = tf.nn.l2_loss(output - expected_output)
            # tv_loss = tv_weight * total_variation(image)

            overall_loss = content_loss
            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            train_step = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.9,
                                   beta2=0.999).minimize(overall_loss)

            def print_progress(i, feed_dict, last=False):
                stderr.write(
                    'Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    stderr.write('Learning rate %f\n' % (learning_rate_decayed.eval()))
                    # Assume that the feed_dict is for the last content and style.
                    # stderr.write('    style loss: %g\n' % style_loss_for_each_style[-1].eval(feed_dict=feed_dict))
                    # stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                    stderr.write('    total loss: %g\n' % overall_loss.eval(feed_dict=feed_dict))

        # Optimization
        # It used to track and record only the best one with lowest loss. This is no longer necessary and I think
        # just recording the one generated at each round will make it easier to debug.
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
                    image_sketches = image_to_sketches_util.image_to_sketch(content_image)
                    image_sketches = np.expand_dims(image_sketches, axis=3)

                    # Now generate an image using the style_blend_weights given.
                    feed_dict = {expected_output:content_image, input_sketches:image_sketches}

                    generated_image = output.eval(feed_dict=feed_dict)
                    iterator += 1
                    # Can't return because we are in a generator.
                    # yield (iterator, vgg.unprocess(
                    #     scipy.misc.imresize(generated_image[0, :, :, :], (input_shape[1], input_shape[2])), mean_pixel))
                    # No need to unprocess it because we've preprocessed the generated image in the network. That means
                    # the generated image is before preprocessing.
                    yield (iterator, generated_image)

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

                for i in range(iter_start, iterations):
                    # First decay the learning rate if we need to
                    if (i % lr_decay_steps == 0):
                        current_lr = learning_rate_decayed.eval()
                        sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))

                    current_content_dirs = get_batch(content_dirs, i * batch_size, batch_size)
                    content_pre_list = read_and_resize_batch_images(current_content_dirs, input_shape[1],
                                                                    input_shape[2])

                    image_sketches = image_to_sketches_util.image_to_sketch(content_pre_list)
                    image_sketches = np.expand_dims(image_sketches, axis=3)

                    # Now generate an image using the style_blend_weights given.
                    feed_dict = {expected_output:content_pre_list, input_sketches:image_sketches}
                    last_step = (i == iterations - 1)

                    train_step.run(feed_dict=feed_dict)
                    print_progress(i, feed_dict=feed_dict, last=last_step)
                    # Record loss after each training round.
                    with open(save_dir + 'loss.tsv','a') as loss_record_file:
                        loss_record_file.write('%d\t%g\n' % (i, overall_loss.eval(feed_dict=feed_dict)))

                    if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                        saver.save(sess, save_dir + 'model.ckpt', global_step=i)

                        if test_img_dir is not None:
                            test_image = imread(test_img_dir)
                            test_image_shape = test_image.shape

                        """
                        def color_sketches_net(height, width, iterations, batch_size, content_weight, tv_weight,
                        learning_rate, lr_decay_steps=200, min_lr=0.001, lr_decay_rate=0.7,print_iterations=None,
                        checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                        do_restore_and_train=False, content_folder=None,
                        from_screenshot=False, from_webcam=False, test_img_dir=None):
                        """


                        # The for loop will run once and terminate. Can't use return and yield in the same function so this is a hacky way to do it.
                        for _, generated_image in color_sketches_net(test_image_shape[0],
                                                                      test_image_shape[1],
                                                                      iterations,
                                                                      1,
                                                                      content_weight, tv_weight,
                                                                      learning_rate,
                                                                      save_dir=save_dir,
                                                                      do_restore_and_generate=True,
                                                                      do_restore_and_train=False,
                                                                      from_screenshot=False,
                                                                      from_webcam=False,
                                                                      test_img_dir=test_img_dir):
                            pass

                            best_image = generated_image

                        # Because we now have batch, choose the first one in the batch as our sample image.
                        yield (
                            (None if last_step else i),
                            None if test_img_dir is None else
                             best_image
                        )