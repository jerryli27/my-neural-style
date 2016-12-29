from argparse import ArgumentParser

import scipy.misc

from general_util import *
from stylize import stylize

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
SEMANTIC_MASKS_WEIGHT = 1.0
SEMANTIC_MASKS_NUM_LAYERS = 4
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
            dest='content', help='content image. If left blank, it will switch to texture/style genration mode.',
            metavar='CONTENT', default='', required=False)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)

    parser.add_argument('--use_semantic_masks',
                        dest='use_semantic_masks', help='If true, we accept some additional image inputs. They '
                                                        'represent the semantic masks of the content and style images.'
                                                        '(default %(default)s).', action='store_true')
    parser.set_defaults(use_semantic_masks=False)
    parser.add_argument('--semantic_masks_weight',
                        dest='semantic_masks_weight', help='The weight we give to matching semantic masks',
                        metavar='WIDTH', required=False, type=float, default=SEMANTIC_MASKS_WEIGHT)
    parser.add_argument('--output_semantic_mask',
            dest='output_semantic_mask', help='one content image semantic mask', required=False)
    parser.add_argument('--style_semantic_masks',
            dest='style_semantic_masks',
            nargs='+', help='one or more style image semantic masks', required=False)
    parser.add_argument('--semantic_masks_num_layers', type=int, dest='semantic_masks_num_layers',
                        help='number of semantic masks (default %(default)s).',
                        metavar='SEMANTIC_MASKS_NUM_LAYERS', default=SEMANTIC_MASKS_NUM_LAYERS) # TODO: FOR FUTURE USE.


    parser.add_argument('--new_gram',
                        dest='new_gram', help='If true, it uses the new loss function instead of the gram loss. '
                                                        '(FOR TESTING)(default %(default)s).', action='store_true')
    parser.set_defaults(new_gram=False)
    parser.add_argument('--new_gram_shift_size', type=int,
            dest='new_gram_shift_size', help='TODO', default=4)
    parser.add_argument('--new_gram_stride', type=int,
            dest='new_gram_stride', help='TODO', default=1)

    parser.add_argument('--content_img_style_weight_mask',
            dest='content_img_style_weight_mask', help='one style weight masks for the content image.', required=False)


    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format',
            metavar='OUTPUT')
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH', default=256)
    parser.add_argument('--height', type=int, dest='height',
                        help='Input and output height. All content images and style images should be automatically '
                             'scaled accordingly.',
                        metavar='HEIGHT', default=256)
    parser.add_argument('--style-scales', type=float,
            dest='style_scales',
            nargs='+', help='one or more style scales',
            metavar='STYLE_SCALE')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--use_mrf',
                        dest='use_mrf', help='If true, we use Markov Random Fields loss instead of Gramian loss.'
                                             ' (default %(default)s).', action='store_true')
    parser.set_defaults(use_mrf=False)
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
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    # content_image = imread(options.content)
    # style_images = [imread(style) for style in options.styles]
    #
    # width = options.width
    # if width is not None:
    #     new_shape = (int(math.floor(float(content_image.shape[0]) /
    #             content_image.shape[1] * width)), width)
    #     content_image = scipy.misc.imresize(content_image, new_shape)

    content_image = None
    if options.content != '':
        print('reading content image %s' %options.content)
        content_image =read_and_resize_images(options.content, options.height, options.width)
    # style_images = read_and_resize_images(options.styles, options.height, options.width)
    style_images = read_and_resize_images(options.styles, None, None) # We don't need to resize style images... Testing...

    target_shape = (1, options.height, options.width, 3)
    # Again.. probably no need to resize style images... unless we're doing mrf loss?? Dunno.
    # for i in range(len(style_images)):
    #     style_scale = STYLE_SCALE
    #     if options.style_scales is not None:
    #         style_scale = options.style_scales[i]
    #     style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
    #             target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])

    output_semantic_mask = None
    style_semantic_masks = None
    if options.use_semantic_masks:
        assert (len(options.style_semantic_masks) == len(options.styles))
        # output_semantic_mask = imread(options.output_semantic_mask)
        # width = options.width
        # if width is not None:
        #     new_shape = (int(math.floor(float(output_semantic_mask.shape[0]) /
        #                                 output_semantic_mask.shape[1] * width)), width)
        #     output_semantic_mask = scipy.misc.imresize(output_semantic_mask, new_shape)
        # style_semantic_masks = [imread(style) for style in options.style_semantic_masks]
        # for i in range(len(style_semantic_masks)):
        #     style_scale = STYLE_SCALE
        #     if options.style_scales is not None:
        #         style_scale = options.style_scales[i]
        #     style_semantic_masks[i] = scipy.misc.imresize(style_semantic_masks[i], style_scale *
        #             target_shape[1] / style_semantic_masks[i].shape[1])
        # output_semantic_mask = read_and_resize_images(options.output_semantic_mask, options.height, options.width)
        # style_semantic_masks = read_and_resize_images(options.style_semantic_masks, options.height, options.width)

        output_semantic_mask_paths = get_all_image_paths_in_dir(options.output_semantic_mask)
        output_semantic_mask = read_and_resize_bw_mask_images(output_semantic_mask_paths, options.height, options.width, 1,
                                                              options.semantic_masks_num_layers)
        style_semantic_masks = []
        for style_i, style_semantic_mask_dir in enumerate(options.style_semantic_masks):
            style_semantic_mask_paths = get_all_image_paths_in_dir(style_semantic_mask_dir)
            style_semantic_masks.append(read_and_resize_bw_mask_images(style_semantic_mask_paths, style_images[style_i].shape[0], style_images[style_i].shape[1], 1,
                                                           options.semantic_masks_num_layers))

    content_img_style_weight_mask = None
    if options.content_img_style_weight_mask:
        content_img_style_weight_mask = (read_and_resize_bw_mask_images([options.content_img_style_weight_mask], options.height, options.width, 1, 1))



    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    for iteration, image in stylize(
        network=options.network,
        initial=initial,
        content=content_image,
        styles=style_images,
        shape=target_shape,
        iterations=options.iterations,
        content_weight=options.content_weight,
        style_weight=options.style_weight,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        use_mrf=options.use_mrf,
        use_semantic_masks = options.use_semantic_masks,
        output_semantic_mask = output_semantic_mask,
        style_semantic_masks= style_semantic_masks,
        semantic_masks_weight= options.semantic_masks_weight,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations,
        new_gram=options.new_gram,
        new_gram_shift_size=options.new_gram_shift_size,
        new_gram_stride=options.new_gram_stride,
        semantic_masks_num_layers=options.semantic_masks_num_layers,
        content_img_style_weight_mask = content_img_style_weight_mask
    ):
        output_file = None
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            imsave(output_file, image)

# TODO: add loss for deviation from original mask layer


if __name__ == '__main__':
    main()

"""
--content=style_compressed/semantic_masks/Freddie.jpg
--styles=style_compressed/semantic_masks/Mia.jpg
--output=output/test.jpg
--learning-rate=10
--iterations=1000
--style-weight=50
--content-weight=5
--checkpoint-output="output_checkpoint/checkpoint_test_%s.jpg"
--checkpoint-iterations=50
--output_semantic_mask=style_compressed/semantic_masks/Freddie_sem.png
--style_semantic_masks=style_compressed/semantic_masks/Mia_sem.png
--width=256
--use_mrf
--use_semantic_masks
--semantic_masks_weight=0.00000001"""

"""
--content=style_compressed/semantic_masks/Freddie.jpg
--styles=style_compressed/shirobako_mask/shirobako_01_0025.png
--output=output/shirobako.jpg
--learning-rate=10
--iterations=1000
--style-weight=200
--content-weight=5
--checkpoint-output="output_checkpoint/checkpoint_shirobako_%s.jpg"
--checkpoint-iterations=50
--output_semantic_mask=style_compressed/semantic_masks/Freddie_sem_mod.png
--style_semantic_masks=style_compressed/shirobako_mask/shirobako_01_0025_mask.jpg
--width=256
--use_mrf
--use_semantic_masks
--semantic_masks_weight=0.1
--print-iterations=10"""

"""

--content=style_compressed/semantic_masks/Freddie.jpg
--styles=style_compressed/shirobako_mask/shirobako_01_0025.png
--output=output/shirobako.jpg
--learning-rate=10
--iterations=1000
--style-weight=500
--content-weight=5
--checkpoint-output="output_checkpoint/checkpoint_shirobako_%s.jpg"
--checkpoint-iterations=50
--output_semantic_mask=style_compressed/semantic_masks/Freddie_sem_mod.png
--style_semantic_masks=style_compressed/shirobako_mask/shirobako_01_0025_mask.jpg
--width=256
--height=256
--semantic_masks_weight=0.1
--print-iterations=100
"""

"""
--content=style_compressed/semantic_masks/Freddie.jpg
--styles=style_compressed/shirobako_mask/shirobako_01_0025.png
--output=output/shirobako.jpg
--learning-rate=10
--iterations=1000
--style-weight=20
--content-weight=5
--checkpoint-output="output_checkpoint/checkpoint_shirobako_%s.jpg"
--checkpoint-iterations=50
--output_semantic_mask=style_compressed/semantic_masks/Freddie_sem_mod.png
--style_semantic_masks=style_compressed/shirobako_mask/shirobako_01_0025_mask.jpg
--width=256
--use_mrf
--use_semantic_masks
--semantic_masks_weight=100.0
--print-iterations=10

"""


"""
--content=source_compressed/todo/6.jpg
--styles=source_compressed/todo/4.jpg
--output=output/light_bulb.jpg
--learning-rate=10
--iterations=1000
--style-weight=25
--content-weight=5
--checkpoint-output="output_checkpoint/checkpoint_lightbulb_%s.jpg"
--checkpoint-iterations=50
--output_semantic_mask=test_masks/
--style_semantic_masks=test_masks/
--width=256
--semantic_masks_weight=100.0
--print-iterations=10
--new_gram
--use_semantic_masks"""