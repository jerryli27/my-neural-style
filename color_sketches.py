"""
TODO: Add description
"""
from argparse import ArgumentParser

import color_sketches_net
from general_util import *

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

# Higher learning rate than 0.01 may sacrifice the quality of the network.
LEARNING_RATE = 0.001  # Set according to  https://arxiv.org/abs/1610.07629.
ITERATIONS = 160000  # 40000 in https://arxiv.org/abs/1610.07629
BATCH_SIZE = 4  # 16 in https://arxiv.org/abs/1610.07629, but higher value requires more memory.
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
STYLE_WEIGHT_MASKS_FOR_TRAINING = 'style_weight_train_masks.npy'
PRINT_ITERATIONS = 100
MASK_FOLDER = 'random_masks/'
SEMANTIC_MASKS_WEIGHT = 1.0
SEMANTIC_MASKS_NUM_LAYERS = 1


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_folder', dest='content_folder',
                        help='The path to the content images for training. In the papers they use the Microsoft COCO dataset.',
                        metavar='CONTENT_FOLDER', default='../johnson-fast-neural-style/fast-style-transfer/data/train2014/')
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

    for iteration, image in color_sketches_net.color_sketches_net(
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
            content_folder=options.content_folder,
            test_img_dir=options.test_img
    ):
        if options.do_restore_and_generate:
            imsave(options.output, image)
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