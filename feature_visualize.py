"""
This file will take two images, one style and one content image. It can help visualize how the vgg network layers are
activated for each one. This is mainly for learning and debugging purposes.
"""

from argparse import ArgumentParser

from matplotlib import pyplot as plt

import feature_visualize_util
from general_util import *

# default arguments
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_IMG = '/home/jerryli27/PycharmProjects/my-neural-style/source_compressed/chicago.jpg'
STYLE_IMG = '/home/jerryli27/PycharmProjects/my-neural-style/output/mirror-nstyle-van_gogh_starry_sky-iter-80000-batchsize-8-lr-0.001000-use_mrf-False-johnson-style-200-content-5-stylenum-0_67500.jpg'
OUTPUT_PATH=''

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content_img', type=str,
                        dest='content_img', help='test image path',
                        metavar='CONTENT_IMG', default = CONTENT_IMG)
    parser.add_argument('--style_img',
            dest='style_img', help='one style image',
            metavar='STYLE_IMG', default=STYLE_IMG)

    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', default=OUTPUT_PATH)
    parser.add_argument('--height', type=int,
            dest='height', help='output height',
            metavar='HEIGHT', default=256)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH', default=256)
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--use_mrf',
                        dest='use_mrf', help='If true, we use Markov Random Fields loss instead of Gramian loss.'
                                             ' (default %(default)s).', action='store_true')
    parser.set_defaults(use_mrf=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    target_shape = (options.height, options.width)
    content_image = imread(options.content_img, shape=target_shape)
    style_image = imread(options.style_img, shape=target_shape)
    # layers = (
    #     'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    #
    #     'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    #
    #     'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    #     'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    #
    #     'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    #     'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    #
    #     'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    #     'relu5_3', 'conv5_4', 'relu5_4'
    # )
    layers = (
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )


    plt.ion()
    fig, axes = plt.subplots(nrows=2,ncols=2)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    axes[0,0].set_title("Content image")
    axes[1,0].set_title("Content Activation")
    axes[0,1].set_title("Style image")
    axes[1,1].set_title("Style Activation")

    for i in range(2):
        for j in range(2):
            axes[i, j].axis("off")

    content_im = axes[0, 0].imshow(content_image.astype(np.uint8), vmin=0, vmax=255)  # Blank starting image
    content_im.axes.figure.canvas.draw()
    style_im = axes[0, 1].imshow(style_image.astype(np.uint8), vmin=0, vmax=255)  # Blank starting image
    style_im.axes.figure.canvas.draw()
    content_feature_im = axes[1, 0].imshow(np.zeros((options.height, options.width, 3)) + 128,vmin=0,vmax=255)  # Blank starting image
    content_feature_im.axes.figure.canvas.draw()
    style_feature_im = axes[1, 1].imshow(np.zeros((options.height, options.width, 3)) + 128,vmin=0,vmax=255)  # Blank starting image
    style_feature_im.axes.figure.canvas.draw()
    plt.show()
    plt.pause(0.001)

    content_features, style_features =  feature_visualize_util.style_synthesis_net(content_image, style_image,layers, options.network)
    
    
    for layer in layers:
        # We must do this clip step before we display the image. Otherwise the color will be off.
        content_feature_image = np.clip(content_features[layer], 0, 255).astype(np.uint8)
        style_feature_image = np.clip(style_features[layer], 0, 255).astype(np.uint8)
        # Change the data in place instead of create a new window.
        for feature_i in range(content_feature_image.shape[3]):
            axes[1, 0].set_title(str('Content Activation %s, %d'% (layer, feature_i)))
            content_feature = content_feature_image[0,:,:, feature_i]
            content_feature = np.dstack((content_feature,content_feature,content_feature))
            content_feature_im.set_data(content_feature)  # Blank starting image
            content_feature_im.axes.figure.canvas.draw()

            axes[1, 1].set_title(str('Style Activation %s, %d'% (layer, feature_i)))
            style_feature = style_feature_image[0,:,:, feature_i]
            style_feature = np.dstack((style_feature,style_feature,style_feature))
            style_feature_im.set_data(style_feature)  # Blank starting image
            style_feature_im.axes.figure.canvas.draw()

            plt.show()
            plt.pause(0.1)
            # plt.ioff()
            print('Showing layer %s, press enter to continue.' %layer)
            plt.waitforbuttonpress()
            # plt.ion()


if __name__ == '__main__':
    main()