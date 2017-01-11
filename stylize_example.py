"""
This file contains code to run neural_style.py It shows a few usages of the function and output the
results.
"""

import os

from general_util import *

if __name__=='__main__':
    # First download the required files.
    download_if_not_exist('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', 'imagenet-vgg-verydeep-19.mat', 'Pretrained vgg 19')
    download_if_not_exist('https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/1-content.jpg',
                          'stylize_examples/1-content.jpg', 'Example content image')
    download_if_not_exist('https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/1-style.jpg',
                          'stylize_examples/1-style.jpg', 'Example style image')

    # The first example: Generate an image styled with Van Gogh's Starry Sky.
    content = 'stylize_examples/1-content.jpg'
    styles = ['stylize_examples/1-style.jpg']
    learning_rate = 10.0
    iterations = 1000
    width = 400
    height = 533

    checkpoint_output_str = 'stylize_examples/output_checkpoint/1_iter_%%s.jpg'
    output_str = 'stylize_examples/output/1_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --content=%s --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d'
              %(content, ' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height))
