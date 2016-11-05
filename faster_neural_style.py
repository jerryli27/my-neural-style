"""
This file uses the texture nets technique to generate an image by combining style of an input and the content of
another input.
"""

import os

import texture_nets
import n_style_feedforward_net
from general_util import *

import numpy as np
import scipy.misc

import math
from argparse import ArgumentParser

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000  # 2000 in the paper
BATCH_SIZE = 1  # 16 in the paper
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content',
            nargs='+', help='one or more content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format',
            metavar='OUTPUT')
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--batch_size', type=int,
            dest='batch_size', help='batch size (default %(default)s)',
            metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--height', type=int,
            dest='height', help='input and output height',
            metavar='HEIGHT', default=256)
    parser.add_argument('--width', type=int,
            dest='width', help='input and output width',
            metavar='WIDTH', default=256)
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--use_n_style', type=bool,
            dest='use_n_style', help='If true, it uses n style model from https://arxiv.org/abs/1610.07629',
            metavar='USE_N_STYLE', default=True)
    parser.add_argument('--model_save_dir',
            dest='model_save_dir', help='The directory to save trained model and its checkpoints.',
            metavar='MODEL_SAVE_DIR', default='models/')
    parser.add_argument('--do_restore_and_generate', type=bool,
            dest='do_restore_and_generate', help='If true, it generates an image from a previously trained model. '
                            'Otherwise it does training and generate a model.',
            metavar='DO_RESTORE_AND_GENERATE', default=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_images = [imread(content) for content in options.content]
    style_images = [imread(style) for style in options.styles]

    width = options.width
    height = options.height
    # If there is no width and height, we automatically take the first image's width and height and apply to all the
    # other ones.
    if width is not None:
        if height is not None:
            target_shape = (height, width)
        else:
            target_shape = (int(math.floor(float(content_images[0].shape[0]) /
                    content_images[0].shape[1] * width)), width)
    else:
        if height is not None:
            target_shape = (height, int(math.floor(float(content_images[0].shape[1]) /
                                        content_images[0].shape[0] * height)))
        else:
            target_shape = (content_images[0].shape[0], content_images[0].shape[1])


    for i in range(len(content_images)):
        if content_images[i].shape != target_shape:
            content_images[i] = scipy.misc.imresize(content_images[i], target_shape)
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], (int(style_scale * target_shape[0]),int(style_scale * target_shape[1])))

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # Default is equal weights. There is no need to divide weight by number of styles, because at training time,
        # for each style, we do one content loss training and one style loss training. If we do the division, then
        # it favors the content loss by a factor of number of styles.
        style_blend_weights = [1.0 for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight * len(style_blend_weights)
                               for weight in style_blend_weights]

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    if options.use_n_style:
        style_synthesis_net = n_style_feedforward_net.style_synthesis_net
    else:
        style_synthesis_net = texture_nets.style_synthesis_net


    for iteration, image in style_synthesis_net(
        path_to_network=options.network,
        content=content_images,
        styles=style_images,
        iterations=options.iterations,
        batch_size=options.batch_size,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations,
        save_dir=options.model_save_dir,
        do_restore_and_generate=options.do_restore_and_generate,
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image)
        elif options.use_n_style:
            for i, _ in enumerate(options.styles):
                output_file = None
                if iteration is not None:
                    output_file = options.checkpoint_output % (i, iteration)
                else:
                    output_file = options.output % i  # TODO: add test for legal output.
                if output_file:
                    imsave(output_file, image[i])
        else:
            output_file = None
            if iteration is not None:
                output_file = options.checkpoint_output % iteration
            else:
                output_file = options.output
            if output_file:
                imsave(output_file, image)

# TODO: There are some serious problems. The result is not nearly comparable with the results from Gatyls.
# It's not batch, it's not norm. not relu5_1, not total variation.
# I know why. The network is supposed to take in more than 1 content image.
# TODO: Check why we need to feed in different sizes of original image to the generation layer.


if __name__ == '__main__':
    main()
    # following are some lists of possible commands.
    """
    --content=source_compressed/256/19.jpg-256.jpg --styles style_compressed/256/4.jpg-256.jpg style_compressed/256/red-peppers256.o.jpg --output=output/19-blended-4-instancenorm-iter-1500-lr-10-style-50-content-5.jpg --learning-rate=10 --iterations=1500 --style-weight=50 --content-weight=5 --checkpoint-output="output_checkpoint/checkpoint_19-blended-4-instancenorm-iter-1500-lr-10-style-50-content-5-stylenum-%s_%s.jpg" --checkpoint-iterations=50
    --content=source_compressed/256/19.jpg-256.jpg --styles=style_compressed/256/4.jpg-256.jpg  --output=output/19-blended-4-instancenorm-iter-1500-lr-10-style-50-content-5.jpg --learning-rate=10 --iterations=1500 --style-weight=50 --content-weight=5 --checkpoint-output="output_checkpoint/checkpoint_19-blended-4-instancenorm-iter-1500-lr-10-style-50-content-5_%s.jpg" --checkpoint-iterations=50
    --content=source_compressed/256/19.jpg-256.jpg --styles style_compressed/256/4.jpg-256.jpg style_compressed/256/red-peppers256.o.jpg --output=output/19-blended-4-nstyle-iter-1500-lr-10-style-50-content-5.jpg --learning-rate=10 --iterations=1000 --style-weight=50 --content-weight=5 --checkpoint-output="output_checkpoint/checkpoint_19-blended-4-nstyle-iter-1500-lr-10-style-50-content-5-stylenum-%s_%s.jpg" --checkpoint-iterations=50 --do_restore_and_generate=True
    --content=source_compressed/256/19.jpg-256.jpg --styles style_compressed/claude_monet/256/1.jpg style_compressed/claude_monet/256/2.jpg --output=output/19-blended-4-nstyle-iter-1500-lr-10-style-50-content-5.jpg --learning-rate=10 --iterations=1500 --style-weight=50 --content-weight=5 --checkpoint-output="output_checkpoint/checkpoint_19-blended-4-nstyle-iter-1500-lr-10-style-50-content-5-stylenum-%s_%s.jpg" --checkpoint-iterations=300
    --content=source_compressed/512/sea_512.jpg --styles style_compressed/claude_monet/512/1.jpg style_compressed/claude_monet/512/2.jpg style_compressed/claude_monet/512/3.jpg style_compressed/claude_monet/512/4.jpg --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5.jpg --learning-rate=10 --iterations=1500 --style-weight=100 --content-weight=5 --checkpoint-output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-stylenum-%s_%s.jpg" --checkpoint-iterations=300
    --content=source_compressed/512/sea_512.jpg --styles style_compressed/claude_monet/512/1.jpg style_compressed/claude_monet/512/2.jpg style_compressed/claude_monet/512/3.jpg style_compressed/claude_monet/512/4.jpg --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5.jpg --learning-rate=10 --iterations=150000 --style-weight=100 --content-weight=5 --checkpoint-output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-stylenum-%s_%s.jpg" --checkpoint-iterations=300

    """
