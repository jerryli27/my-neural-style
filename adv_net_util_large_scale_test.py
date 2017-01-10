"""
This file is for making sure adversarial network has at least the capability to tell real images from animation images.
"""

# import gtk.gdk

from argparse import ArgumentParser
from sys import stderr

import tensorflow as tf

import adv_net_util
from general_util import *


# TODO: change rtype
def color_sketches_net(height, width, iterations, batch_size, content_weight, tv_weight,
                       learning_rate,
                       lr_decay_steps=50000,
                       min_lr=0.00001, lr_decay_rate=0.7, print_iterations=None,
                       checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                       do_restore_and_train=False, real_folder=None, fake_folder=None):
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

        learning_rate_decayed_init = tf.constant(learning_rate)
        learning_rate_decayed = tf.get_variable(name='learning_rate_decayed', trainable=False,
                                                initializer=learning_rate_decayed_init)

        adv_net_input_real = tf.placeholder(tf.float32,
                                         shape=[batch_size, input_shape[1], input_shape[2], 3], name='adv_net_input_real')
        adv_net_prediction_image_input = adv_net_util.net(adv_net_input_real)

        adv_net_input_fake = tf.placeholder(tf.float32,
                                       shape=[batch_size, input_shape[1], input_shape[2], 3], name='adv_net_input_fake')
        adv_net_prediction_generator_input = adv_net_util.net(adv_net_input_fake, reuse=True)
        adv_net_all_var = adv_net_util.get_net_all_variables()

        logits_from_i = adv_net_prediction_image_input
        logits_from_g = adv_net_prediction_generator_input

        # One represent labeling the image as coming from real image. Zero represent labeling it as generated.
        adv_loss_from_i = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([batch_size], dtype=tf.int64)))
        adv_loss_from_g = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([batch_size], dtype=tf.int64)))

        adv_loss =  adv_loss_from_i + adv_loss_from_g
        adv_train_step = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.5,
                               beta2=0.999).minimize(adv_loss, var_list=adv_net_all_var)
        adv_train_step_i = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.5,
                               beta2=0.999).minimize(adv_loss_from_i, var_list=adv_net_all_var)
        adv_train_step_g = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.5,
                               beta2=0.999).minimize(adv_loss_from_g, var_list=adv_net_all_var)

        # with tf.control_dependencies([generator_train_step_through_adv, adv_train_step]):
        with tf.control_dependencies([adv_train_step_i, adv_train_step_g]):
            adv_generator_both_train = tf.no_op(name='adv_generator_both_train')

        def print_progress(i, adv_feed_dict, last=False):
            stderr.write(
                'Iteration %d/%d\n' % (i + 1, iterations))
            if last or (print_iterations and i % print_iterations == 0):
                stderr.write('Learning rate %f\n' % (learning_rate_decayed.eval()))
                stderr.write('   adv_from_i loss: %g\n' % adv_loss_from_i.eval(feed_dict=adv_feed_dict))
                stderr.write('   adv_from_g loss: %g\n' % adv_loss_from_g.eval(feed_dict=adv_feed_dict))


        # Optimization
        # It used to track and record only the best one with lowest loss. This is no longer necessary and I think
        # just recording the one generated at each round will make it easier to debug.
        best_real_image = None
        best_fake_image = None

        best_real_loss = 100.0
        best_fake_loss = 100.0

        saver = tf.train.Saver()
        with tf.Session() as sess:
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
            else:
                sess.run(tf.initialize_all_variables())

            # Get path to all content images.
            real_dirs = get_all_image_paths_in_dir(real_folder)
            # Ignore the ones at the end.
            if batch_size != 1 and len(real_dirs) % batch_size != 0:
                real_dirs = real_dirs[:-(len(real_dirs) % batch_size)]
            print('The size of real dataset is %d images.' % len(real_dirs))


            fake_dirs = get_all_image_paths_in_dir(fake_folder)
            # Ignore the ones at the end.
            if batch_size != 1 and len(fake_dirs) % batch_size != 0:
                fake_dirs = fake_dirs[:-(len(fake_dirs) % batch_size)]
            print('The size of fake dataset is %d images.' % len(fake_dirs))


            # # Test training GAN differently***
            # generators_turn = True
            # # END TEST***


            for i in range(iter_start, iterations):
                # First decay the learning rate if we need to
                if (i % lr_decay_steps == 0 and i!= iter_start):
                    current_lr = learning_rate_decayed.eval()
                    sess.run(learning_rate_decayed.assign(max(min_lr, current_lr * lr_decay_rate)))

                current_content_dirs = get_batch_paths(real_dirs, i * batch_size, batch_size)
                content_pre_list = read_and_resize_batch_images(current_content_dirs, input_shape[1],
                                                                input_shape[2])


                current_fake_dirs = get_batch_paths(fake_dirs, i * batch_size, batch_size)
                fake_pre_list = read_and_resize_batch_images(current_fake_dirs, input_shape[1],
                                                                input_shape[2])

                last_step = (i == iterations - 1)


                adv_feed_dict = {adv_net_input_real: content_pre_list, adv_net_input_fake: fake_pre_list}
                # TEST printing before training
                print_progress(i, adv_feed_dict=adv_feed_dict, last=last_step)

                # if generators_turn:
                #     # generator_train_step.run(feed_dict=feed_dict)
                #     generator_train_step_through_adv.run(feed_dict=adv_feed_dict)
                #     adv_train_step.run(feed_dict=adv_feed_dict)

                # generator_train_step_through_adv.run(feed_dict=adv_feed_dict)
                # adv_train_step.run(feed_dict=adv_feed_dict)


                adv_generator_both_train.run(feed_dict=adv_feed_dict)
                # if i < 10000:
                #     generator_train_step.run(feed_dict=feed_dict)

                print_progress(i, adv_feed_dict= adv_feed_dict, last=last_step)
                # TODO:
                if i%10==0:
                    with open(save_dir + 'adv_loss.tsv', 'a') as loss_record_file:
                        current_adv_loss_i = adv_loss_from_i.eval(feed_dict=adv_feed_dict)
                        current_adv_loss_g = adv_loss_from_g.eval(feed_dict=adv_feed_dict)
                        loss_record_file.write('%d\t%g\t%g\n' % (i, current_adv_loss_i, current_adv_loss_g))


                        if current_adv_loss_i < best_real_loss:
                            best_real_loss = current_adv_loss_i
                            best_real_image = content_pre_list
                        if current_adv_loss_g < best_fake_loss:
                            best_fake_loss = current_adv_loss_g
                            best_fake_image = fake_pre_list


                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    # saver.save(sess, save_dir + 'model.ckpt', global_step=i)

                    # Because we now have batch, choose the first one in the batch as our sample image.
                    yield (
                        (None if last_step else i),
                        best_real_image, best_fake_image
                    )


# default arguments
CONTENT_WEIGHT = 5e0
TV_WEIGHT = 2e2

LEARNING_RATE = 0.0002 # Set according to dcgan paper
ITERATIONS = 4000
BATCH_SIZE = 8
PRINT_ITERATIONS = 100

# TODO: fix comments for the color sketches net.
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_folder', dest='content_folder',
                        help='The path to the colored pixiv images for training. ',
                        metavar='CONTENT_FOLDER', default='/home/ubuntu/pixiv/pixiv_training_filtered/')
    parser.add_argument('--output', dest='output',
                        help='Output path.',
                        metavar='OUTPUT', default='output/adv_net_util_large_scale_test_%s.jpg')
    parser.add_argument('--checkpoint_output', dest='checkpoint_output',
                        help='The checkpoint output format. It must contain 2 %s, the first one for content index '
                             'and the second one for style index.',
                        metavar='OUTPUT', default='output_checkpoint/adv_net_util_large_scale_test_%d_%s.jpg')


    parser.add_argument('--iterations', type=int, dest='iterations',
                        help='Iterations (default %(default)s).',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        help='Batch size (default %(default)s).',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--height', type=int, dest='height',
                        help='Input and output height. All content images and style images should be automatically '
                             'scaled accordingly.',
                        metavar='HEIGHT', default=256)
    parser.add_argument('--width', type=int, dest='width',
                        help='Input and output width. All content images and style images should be automatically '
                             'scaled accordingly.',
                        metavar='WIDTH', default=256)
    parser.add_argument('--content_weight', type=float, dest='content_weight',
                        help='Content weight (default %(default)s).',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--tv_weight', type=float, dest='tv_weight',
                        help='Total variation regularization weight (default %(default)s).',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                        help='Learning rate (default %(default)s).',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--print_iterations', type=int, dest='print_iterations',
                        help='Statistics printing frequency.',
                        metavar='PRINT_ITERATIONS', default=PRINT_ITERATIONS)
    parser.add_argument('--checkpoint_iterations', type=int, dest='checkpoint_iterations',
                        help='Checkpoint frequency.',
                        metavar='CHECKPOINT_ITERATIONS', default=100)
    parser.add_argument('--test_img', type=str,
                        dest='test_img', help='test image path',
                        metavar='TEST_IMAGE', default='../johnson-fast-neural-style/fast-style-transfer/data/train2014/')
    parser.add_argument('--test_img_hint', type=str,
                        dest='test_img_hint', help='test image path')
    parser.add_argument('--model_save_dir', dest='model_save_dir',
                        help='The directory to save trained model and its checkpoints.',
                        metavar='MODEL_SAVE_DIR', default='model/adv_net_util_large_scale_test/')
    parser.add_argument('--do_restore_and_generate', dest='do_restore_and_generate',
                        help='If true, it generates an image from a previously trained model. '
                             'Otherwise it does training and generate a model.',
                        action='store_true')
    parser.set_defaults(do_restore_and_generate=False)
    parser.add_argument('--do_restore_and_train', dest='do_restore_and_train',
                        help='If set, we read the model at model_save_dir and start training from there. '
                             'The overall setting and structure must be the same.',
                        action='store_true')
    parser.set_defaults(do_restore_and_train=False)
    return parser


def main():
    print('Starting to run a large scale test on the capability of the adversarial network. It will probably take a '
          'while depending on the input you provided. At the end you should see that the loss for both real and fake '
          'iamges goes down to near zero (in the recorded adv_loss.tsv). If not, then the test failed.')

    parser = build_parser()
    options = parser.parse_args()

    if not os.path.exists(options.model_save_dir):
        os.makedirs(options.model_save_dir)  # TODO: add %s content_img_style_weight_mask_string to the model_save_dir

    for iteration, best_real_img, best_fake_img in color_sketches_net(
            height=options.height,
            width=options.width,
            iterations=options.iterations,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
            save_dir=options.model_save_dir,
            do_restore_and_generate=options.do_restore_and_generate,
            do_restore_and_train=options.do_restore_and_train,
            real_folder=options.content_folder,
            fake_folder=options.test_img

    ):
        if iteration is not None:
            output_file_real = options.checkpoint_output % (iteration, 'real')
            output_file_fake = options.checkpoint_output % (iteration, 'fake')
        else:
            output_file_real = options.output % ('real')
            output_file_fake = options.output % ('fake')

        # The first dimension is always going to be 1 for now... until I support multiple test image output.
        imsave(output_file_real, best_real_img[0,...])
        imsave(output_file_fake, best_fake_img[0,...])

if __name__ == '__main__':
    main()