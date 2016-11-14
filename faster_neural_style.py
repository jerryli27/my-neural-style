"""
This file uses the texture nets technique to generate an image by combining style of an input and the content of
another input.
Citations:
A Learned Representation For Artistic Style https://arxiv.org/abs/1610.07629
Texture Networks: Feed-forward Synthesis of Textures and Stylized Images https://arxiv.org/abs/1603.03417
Instance Normalization: The Missing Ingredient for Fast Stylization https://arxiv.org/abs/1607.08022
Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis: https://arxiv.org/abs/1601.04589
Perceptual Losses for Real-Time Style Transfer and Super-Resolution: https://arxiv.org/abs/1603.08155
Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks: https://arxiv.org/abs/1603.01768
"""

import math
import os
import scipy.misc
from argparse import ArgumentParser

import n_style_feedforward_net
from general_util import *

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
# lr = 0.001 in https://arxiv.org/abs/1610.07629.
# Higher learning rate than 0.01 may sacrifice the quality of the network.
LEARNING_RATE = 0.001
STYLE_SCALE = 1.0
ITERATIONS = 160000  # 40000 in https://arxiv.org/abs/1610.07629
BATCH_SIZE = 4  # 16 in https://arxiv.org/abs/1610.07629
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
PRINT_ITERATIONS = 100


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content', nargs='+',
                        help='One or more content images.',
                        metavar='CONTENT', required=True)
    parser.add_argument('--styles',dest='styles', nargs='+',
                        help='One or more style images.',
                        metavar='STYLE', required=True)
    parser.add_argument('--texture_synthesis_only', dest='texture_synthesis_only',
                        help='If true, we only generate the texture of the style images.'
                             ' No content image will be used.',
                        action='store_true')
    parser.set_defaults(texture_synthesis_only=False)
    parser.add_argument('--output', dest='output',
                        help='Output path.',
                        metavar='OUTPUT', required=True)
    parser.add_argument('--checkpoint_output', dest='checkpoint_output',
                        help='The checkpoint output format. It must contain 2 %s, the first one for content index '
                             'and the second one for style index.',
                        metavar='OUTPUT')
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
    parser.add_argument('--network', dest='network',
                        help='Path to network parameters (default %(default)s).',
                        metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--use_mrf', dest='use_mrf',
                        help='If true, we use Markov Random Fields loss instead of Gramian loss.'
                             ' (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_mrf=False)
    parser.add_argument('--use_johnson', dest='use_johnson',
                        help='If true, we use the johnson generator net instead of pyramid net (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_johnson=False)
    # TODO: delete content weight after we make sure we do not need tv weight.
    parser.add_argument('--content_weight', type=float, dest='content_weight',
                        help='Content weight (default %(default)s).',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style_weight', type=float, dest='style_weight',
                        help='Style weight (default %(default)s).',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style_blend_weights', type=float, dest='style_blend_weights',
                        help='How much we weigh each style during the training. The more we weigh one style, the more '
                             'loss will come from that style and the more the output image will look like that style. '
                             'During training it should not be set because the network automatically deals with '
                             'multiple styles.',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
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
    parser.add_argument('--model_save_dir', dest='model_save_dir',
                        help='The directory to save trained model and its checkpoints.',
                        metavar='MODEL_SAVE_DIR', default='models/')
    parser.add_argument('--do_restore_and_generate', type=bool, dest='do_restore_and_generate',
                        help='If true, it generates an image from a previously trained model. '
                             'Otherwise it does training and generate a model.',
                        metavar='DO_RESTORE_AND_GENERATE', default=False)
    parser.add_argument('--do_restore_and_train', dest='do_restore_and_train',
                        help='If set, we read the model at model_save_dir and start training from there. '
                             'The overall setting and structure must be the same.',
                        action='store_true')
    parser.set_defaults(do_restore_and_train=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_images, style_images = read_and_resize_images(options.content, options.styles, options.height, options.width)

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # Default is equal weights. There is no need to divide weight by number of styles, because at training time,
        # for each style, we do one content loss training and one style loss training. If we do the division, then
        # it favors the content loss by a factor of number of styles.
        style_blend_weights = [1.0 for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight * len(style_blend_weights)
                               for weight in style_blend_weights]

    if options.output and options.output.count("%s") != 2:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain at least two `%s` (e.g. `foo_content_%s_style_%s.jpg`).")
    if options.checkpoint_output and options.checkpoint_output.count("%s") != 3:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain at least three `%s` (e.g. `foo_content_%s_style_%s_iteration_%s.jpg`).")

    for iteration, image in n_style_feedforward_net.style_synthesis_net(
            path_to_network=options.network,
            contents=content_images,
            styles=style_images,
            iterations=options.iterations,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            style_weight=options.style_weight,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            style_only=options.texture_synthesis_only,
            use_mrf=options.use_mrf,
            use_johnson=options.use_johnson,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
            save_dir=options.model_save_dir,
            do_restore_and_generate=options.do_restore_and_generate,
            do_restore_and_train=options.do_restore_and_train
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image)
        else:
            for style_i, _ in enumerate(options.styles):
                for content_i, _ in enumerate(options.content):
                    if iteration is not None:
                        output_file = options.checkpoint_output % (content_i, style_i, iteration)
                    else:
                        output_file = options.output % (content_i, style_i)  # TODO: add test for legal output.
                    if output_file:
                        imsave(output_file, image[style_i][content_i])

if __name__ == '__main__':
    main()
    # following are some lists of possible commands.
    """
    --content=source_compressed/512/sea_512.jpg --styles style_compressed/claude_monet/512/1.jpg --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s.jpg --learning_rate=10 --iterations=1500 --style_weight=100 --content_weight=5 --checkpoint_output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s_%s.jpg" --checkpoint_iterations=300
    """

# TODO:
"""
MRF running on feedforward style transfer net. But it's not working. The texture synthesis was sort of working but
the style transfer is just a complete mess.
I will try to add the tv loss back to see if that improves anything. If not I will try a different generator structure.
"""