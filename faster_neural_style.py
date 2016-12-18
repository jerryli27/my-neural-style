"""
This file is the main file for feed forward style synthesis. It combined numerous techniques, including but not limited
to: texture synthesis, style synthesis, style synthesis with mrf, and fast neural doodles.
Citations:
A Learned Representation For Artistic Style https://arxiv.org/abs/1610.07629
Texture Networks: Feed-forward Synthesis of Textures and Stylized Images https://arxiv.org/abs/1603.03417
Instance Normalization: The Missing Ingredient for Fast Stylization https://arxiv.org/abs/1607.08022
Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis: https://arxiv.org/abs/1601.04589
Perceptual Losses for Real-Time Style Transfer and Super-Resolution: https://arxiv.org/abs/1603.08155
Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks: https://arxiv.org/abs/1603.01768
"""

from argparse import ArgumentParser

import n_style_feedforward_net
from general_util import *

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

# Higher learning rate than 0.01 may sacrifice the quality of the network.
LEARNING_RATE = 0.001  # Set according to  https://arxiv.org/abs/1610.07629.
STYLE_SCALE = 1.0
ITERATIONS = 160000  # 40000 in https://arxiv.org/abs/1610.07629
BATCH_SIZE = 4  # 16 in https://arxiv.org/abs/1610.07629, but higher value requires more memory.
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
PRINT_ITERATIONS = 100
MASK_FOLDER = 'random_masks/'
SEMANTIC_MASKS_WEIGHT = 1.0
SEMANTIC_MASKS_NUM_LAYERS = 1


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_folder', dest='content_folder',
                        help='The path to the content images for training. In the papers they use the Microsoft COCO dataset.',
                        metavar='CONTENT_FOLDER', default='../johnson-fast-neural-style/fast-style-transfer/data/train2014/')

    parser.add_argument('--style_folder', dest='style_folder',
                        help='The path to the style images for training.',
                        metavar='STYLE_FOLDER',
                        default='/home/jerryli27/shirobako01pic/')

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
    parser.add_argument('--use_skip_noise_4', dest='use_skip_noise_4',
                        help='If true, we use the skip_noise_4 generator network (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_skip_noise_4=False)
    parser.add_argument('--multiple_styles_train_scale_offset_only', dest='multiple_styles_train_scale_offset_only',
                        help='If true, TODO (default %(default)s).',
                        action='store_true')
    parser.set_defaults(multiple_styles_train_scale_offset_only=False)


    parser.add_argument('--use_semantic_masks', dest='use_semantic_masks',
                        help='If true, we use semantic masks to help training (default %(default)s).',
                        action='store_true')
    parser.set_defaults(use_semantic_masks=False)

    parser.add_argument('--mask_folder', dest='mask_folder',
                        help='Folder to a directory containing random mask images for training.',
                        metavar='MASK_FOLDER', default=MASK_FOLDER)
    parser.add_argument('--mask_resize_as_feature', dest='mask_resize_as_feature',
                        help='If true, instead of feeding mask through the network, we resize it and treat that '
                             'resized image as the feature for a given feature layer. (default %(default)s).',
                        action='store_true')
    parser.set_defaults(mask_resize_as_feature=True)
    parser.add_argument('--style_semantic_mask_dirs',dest='style_semantic_mask_dirs', nargs='+',
                        help='One or more style mask images.')
    parser.add_argument('--semantic_masks_weight', type=float, dest='semantic_masks_weight',
                        help='Semantic masks weight (default %(default)s).',
                        metavar='SEMANTIC_MASKS_WEIGHT', default=SEMANTIC_MASKS_WEIGHT)
    parser.add_argument('--semantic_masks_num_layers', type=int, dest='semantic_masks_num_layers',
                        help='number of semantic masks (default %(default)s).',
                        metavar='SEMANTIC_MASKS_NUM_LAYERS', default=SEMANTIC_MASKS_NUM_LAYERS)
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
    parser.add_argument('--test_img', type=str,
                        dest='test_img', help='test image path',
                        metavar='TEST_IMAGE')
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
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    style_images = read_and_resize_images(options.styles, options.height, options.width)

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

    style_semantic_masks = read_and_resize_bw_mask_images(options.style_semantic_mask_dirs, options.height, options.width, len(options.styles), options.semantic_masks_num_layers) if options.use_semantic_masks else []

    if options.output and options.output.count("%s") != 1:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain only one `%s` (e.g. `foo_style_%s.jpg`).")
    if options.checkpoint_output and options.checkpoint_output.count("%s") != 2:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain only two `%s` (e.g. `foo_style_%s_iteration_%s.jpg`).")
    if options.use_johnson and options.use_skip_noise_4:
        parser.error("use_johnson and use_skip_noise_4 can't both be true. Please choose only one generator network.")

    for iteration, image in n_style_feedforward_net.style_synthesis_net(
            path_to_network=options.network,
            height=options.height,
            width=options.width,
            styles=style_images,
            iterations=options.iterations,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            style_weight=options.style_weight,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=options.learning_rate,
            style_only=options.texture_synthesis_only,
            multiple_styles_train_scale_offset_only= options.multiple_styles_train_scale_offset_only,
            use_mrf=options.use_mrf,
            use_johnson=options.use_johnson,
            use_skip_noise_4=options.use_skip_noise_4,
            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
            save_dir=options.model_save_dir,
            do_restore_and_generate=options.do_restore_and_generate,
            do_restore_and_train=options.do_restore_and_train,
            content_folder=options.content_folder,
            use_semantic_masks=options.use_semantic_masks,
            mask_folder=options.mask_folder,
            mask_resize_as_feature=options.mask_resize_as_feature,
            style_semantic_masks=style_semantic_masks,
            semantic_masks_weight=options.semantic_masks_weight,
            semantic_masks_num_layers=options.semantic_masks_num_layers,
            test_img_dir=options.test_img
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image)
        else:
            for style_i, _ in enumerate(options.styles):
                if options.test_img:
                    if iteration is not None:
                        output_file = options.checkpoint_output % (style_i, iteration)
                    else:
                        output_file = options.output % (style_i)  # TODO: add test for legal output.
                    if output_file:
                        imsave(output_file, image[style_i])

if __name__ == '__main__':
    main()
    # following are some lists of possible commands.
    """
    --content=source_compressed/512/sea_512.jpg --styles style_compressed/claude_monet/512/1.jpg --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s.jpg --learning_rate=10 --iterations=1500 --style_weight=100 --content_weight=5 --checkpoint_output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-contentnum-%s-stylenum-%s_%s.jpg" --checkpoint_iterations=300
    """
    """
    --styles
    style_compressed/claude_monet/512/1.jpg
    --output=output/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-stylenum-%s.jpg
    --width=256
    --height=256
    --learning_rate=10
    --iterations=1500
    --style_weight=10vcxxxxxccx0
    --content_weight=5
    --checkpoint_output="output_checkpoint/sea-512-nstyle-iter-1500-lr-10-style-100-content-5-stylenum-%s_%s.jpg"
    --checkpoint_iterations=300
    --use_johnson
    --print_iterations=10
    --test_img=source_compressed/chicago.jpg
    --model_save_dir=model/tempmodel/
    --style_folder=./style_compressed/claude_monet_2/
    """