#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file is for running a neural network (or two if you're using the adversarial net) that can generate colored
version of sketches given the black-and-white sketch image. Training takes a lot of image/paintings, preferably with
sharp edges so that the sketch generator can do its job.
"""
from argparse import ArgumentParser

import color_sketches_net
from general_util import *

# default arguments
CONTENT_WEIGHT = 5e0
TV_WEIGHT = 2e2

LEARNING_RATE = 0.0002  # Set according to the dcgan paper but it also works when we're not using adversarial net.
ITERATIONS = 160000  # Change this according to the dataset used.
num_epochs = 10
# The larger batch size the more memory required and the slower the training is, but it provides stability. Usually 4
# or 8 works well for non-adversarial condition.
BATCH_SIZE = 64
PRINT_ITERATIONS = 100

# TODO: fix comments for the color sketches net.
def build_parser():
    parser = ArgumentParser()
    # '/home/ubuntu/pixiv_full/pixiv/' or /home/ubuntu/pixiv/pixiv_training_filtered/' or
    # '/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/' -> Number of images  442701.
    parser.add_argument('--content_folder', dest='content_folder',
                        help='The path to the colored pixiv images for training. ', metavar='CONTENT_FOLDER',
                        default='/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/pixiv_downloaded/')

    parser.add_argument('--output', dest='output',
                        help='Output path.',
                        metavar='OUTPUT', required=True)
    parser.add_argument('--checkpoint_output', dest='checkpoint_output',
                        help='The checkpoint output format. It must contain 2 %s, the first one for content index '
                             'and the second one for style index.',
                        metavar='OUTPUT')
    parser.add_argument('--generator_network', dest='generator_network', type=str,
                        help='todo. which generator_network it should use.', metavar='GENERATOR_NETWORK',
                        default='johnson')
    parser.add_argument('--use_adversarial_net', dest='use_adversarial_net',
                        help='If set, we train an adversarial network to distinguish between the image generated and '
                             'the real image. This will help the generator to generate more real looking images.',
                        action='store_true')
    parser.set_defaults(use_adversarial_net=False)

    parser.add_argument('--use_hint', dest='use_hint',
                        help='If set, hints are given to the generator network about the color used at various parts '
                             'of the image.',
                        action='store_true')
    parser.set_defaults(use_hint=False)

    parser.add_argument('--num_epochs', type=int, dest='num_epochs',
                        help='num_epochs (default %(default)s).',
                        metavar='num_epochs', default=num_epochs)
    # parser.add_argument('--iterations', type=int, dest='iterations',
    #                     help='Iterations (default %(default)s).',
    #                     metavar='ITERATIONS', default=ITERATIONS)
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
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--test_img', type=str,
                        dest='test_img', help='test image path',
                        metavar='TEST_IMAGE')
    parser.add_argument('--test_img_hint', type=str,
                        dest='test_img_hint', help='test image path')
    parser.add_argument('--model_save_dir', dest='model_save_dir',
                        help='The directory to save trained model and its checkpoints.',
                        metavar='MODEL_SAVE_DIR', default='models/')
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
    parser.add_argument('--restore_from_noadv_to_adv', dest='restore_from_noadv_to_adv',
                        help='If set, it tries to load a checkpoint as if it does not have adversarial network trained. '
                             'It then train the adversarial part from scratch along with other previously trained '
                             'variables.',
                        action='store_true')
    parser.set_defaults(restore_from_noadv_to_adv=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    
    num_images  = len(get_all_image_paths_in_dir(options.content_folder))
    num_iterations = int(num_images * options.num_epochs / options.batch_size)
    print("Number of images in content folder: %d. Number of epochs to train: %d (%d iterations)."
          %(num_images, options.num_epochs ,num_iterations))

    for iteration, image in color_sketches_net.color_sketches_net(
            height=options.height,
            width=options.width,
            iterations=num_iterations,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            generator_network=options.generator_network,
            use_adversarial_net=options.use_adversarial_net,
            use_hint=options.use_hint,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
            save_dir=options.model_save_dir,
            do_restore_and_generate=options.do_restore_and_generate,
            do_restore_and_train=options.do_restore_and_train,
            restore_from_noadv_to_adv=options.restore_from_noadv_to_adv,
            content_folder=options.content_folder,
            test_img_dir=options.test_img,
            test_img_hint=options.test_img_hint
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image[0,...])
        else:
            if options.test_img:
                if iteration is not None:
                    output_file = options.checkpoint_output % (iteration)
                else:
                    output_file = options.output
                if output_file:
                    # The first dimension is always going to be 1 for now... until I support multiple test image output.
                    imsave(output_file, image[0,...])

if __name__ == '__main__':
    main()