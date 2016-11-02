"""
This file uses the texture nets technique to generate an image by combining style of an input and the content of
another input.
"""

import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import time

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
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--batch_size', type=int,
            dest='batch_size', help='batch size (default %(default)s)',
            metavar='BATCH_SIZE', default=BATCH_SIZE)
    parser.add_argument('--height', type=int,
            dest='height', help='output height',
            metavar='HEIGHT', default=256)
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
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
    parser.add_argument('--model_save_dir',
            dest='model_save_dir', help='The directory to save trained model and its checkpoints.',
            metavar='MODEL_SAVE_DIR', default='models/')
    parser.add_argument('--from_screenshot', type=bool,
            dest='from_screenshot', help='If true, the content image is the screen shot',
            metavar='FROM_SCREENSHOT', default=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    dummy_content = np.zeros((options.height, options.width, 3))

    # Todo: use camera.
    # cap = cv2.VideoCapture(0)
    # while (True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # content_image = imread(options.content)
    style_images = [imread(style) for style in options.styles]

    target_shape = (options.height, options.width)
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # Default is equal weights.
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    plt.ion()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title("Real Time Neural Style")
    im = ax.imshow(np.zeros((256, 256, 3)) + 128,vmin=0,vmax=255)  # Blank starting image
    axcolor = 'lightgoldenrodyellow'
    slider_axes = [plt.axes([0.25, 0.24 - i * 0.03, 0.65, 0.02], axisbg=axcolor) for i, style in enumerate(options.styles)]
    sliders = [Slider(slider_axes[i], style, 0.0, 1.0, valinit=style_blend_weights[i]) for i, style in enumerate(options.styles)]
    fig.show()
    im.axes.figure.canvas.draw()
    plt.pause(0.001)
    tstart = None

    def update(val):
        # TODO: Test if we need to normalize the style weights.
        for i, slider in enumerate(sliders):
            style_blend_weights[i] = slider.val

    for slider in sliders:
        slider.on_changed(update)

    for iteration, image in n_style_feedforward_net.style_synthesis_net(
            path_to_network=options.network,
            content=dummy_content,
            styles=style_images,
            iterations=None,
            batch_size=options.batch_size,
            content_weight=options.content_weight,
            style_weight=options.style_weight,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,
            learning_rate=1,  # Dummy learning rate.
            print_iterations=None,
            checkpoint_iterations=None,
            save_dir=options.model_save_dir,
            do_restore_and_generate=True,
            from_screenshot = options.from_screenshot
        ):
        # We must do this clip step before we display the image. Otherwise the color will be off.
        image = np.clip(image, 0, 255).astype(np.uint8)
        if tstart is None:
            tstart = time.time()
        # Change the data in place instead of create a new window.
        ax.set_title(str(iteration))
        im.set_data(image)
        im.axes.figure.canvas.draw()
        plt.pause(0.001)
        print ('FPS:', iteration / (time.time() - tstart + 0.001))


if __name__ == '__main__':
    main()
    # following are some lists of possible commands.
    """
    --styles style_compressed/claude_monet/256/1.jpg style_compressed/claude_monet/256/2.jpg --output=output/19-blended-4-nstyle-iter-1500-lr-10-style-50-content-5.jpg --style-weight=50 --content-weight=5 --from_screenshot=True
    """
