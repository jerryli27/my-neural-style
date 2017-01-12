"""
This file contains code to run neural_style.py It shows a few usages of the function and output the
results.
"""

from general_util import *

if __name__=='__main__':
    # First download the required files.
    download_if_not_exist('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', 'imagenet-vgg-verydeep-19.mat', 'Pretrained vgg 19')
    download_if_not_exist('https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/1-content.jpg',
                          'stylize_examples/1-content.jpg', 'Example content image')
    download_if_not_exist('https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/1-style.jpg',
                          'stylize_examples/1-style.jpg', 'Example style image')
    download_if_not_exist('https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/2-style1.jpg',
                          'stylize_examples/2-style.jpg', 'Example style image No.2')
    download_if_not_exist('https://raw.githubusercontent.com/alexjc/neural-doodle/master/samples/Mia.jpg',
                          'stylize_examples/5-style.jpg', 'Style image for mrf loss with semantic masks')
    download_if_not_exist('https://raw.githubusercontent.com/alexjc/neural-doodle/master/samples/Freddie.jpg',
                          'stylize_examples/5-content.jpg', 'Content image for mrf loss with semantic masks')


    'https://raw.githubusercontent.com/alexjc/neural-doodle/master/samples/Freddie.jpg'


    # The first example: Generate an image styled with Van Gogh's Starry Sky and with content as 1-content.jpg
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

    # The second example: Generate an image textured with Van Gogh's Starry Sky without any content.
    styles = ['stylize_examples/1-style.jpg']
    learning_rate = 10.0
    iterations = 500
    width = 256
    height = 256

    checkpoint_output_str = 'stylize_examples/output_checkpoint/2_iter_%%s.jpg'
    output_str = 'stylize_examples/output/2_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d'
              %(' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height))

    # The third example is to use multiple styles with the first weighted more than the second one.
    content = 'stylize_examples/1-content.jpg'
    styles = ['stylize_examples/1-style.jpg', 'stylize_examples/2-style.jpg']
    style_blend_weights = [0.7, 0.3]
    learning_rate = 10.0
    iterations = 1000
    width = 400
    height = 533

    checkpoint_output_str = 'stylize_examples/output_checkpoint/3_iter_%%s.jpg'
    output_str = 'stylize_examples/output/3_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --content=%s --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d --style-blend-weights %s'
              % (content, ' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height, ' '.join(map(str, style_blend_weights))))


    # The fourth example: use mrf loss instead of gramian loss.
    content = 'stylize_examples/4-content.jpg'
    styles = ['stylize_examples/4-style.jpg']
    learning_rate = 10.0
    iterations = 1000
    width = 512
    height = 384
    style_weight = 5

    checkpoint_output_str = 'stylize_examples/output_checkpoint/4_iter_%%s.jpg'
    output_str = 'stylize_examples/output/4_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --content=%s --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d --style-weight=%f --use_mrf'
              %(content, ' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height, style_weight))

    # The fifth example: use mrf loss with semantic masks.
    content = 'stylize_examples/5-content.jpg'
    styles = ['stylize_examples/5-style.jpg']
    learning_rate = 10.0
    iterations = 1000
    width = 512
    height = 512
    style_weight = 5
    output_semantic_mask = 'stylize_examples/semantic_masks/Freddie_sem_masks/'
    style_semantic_masks = ['stylize_examples/semantic_masks/Mia_sem_masks/']
    semantic_masks_num_layers = 10

    checkpoint_output_str = 'stylize_examples/output_checkpoint/5_iter_%%s.jpg'
    output_str = 'stylize_examples/output/5_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --content=%s --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d --style-weight=%f --use_mrf --output_semantic_mask=%s --style_semantic_masks %s --semantic_masks_num_layers=%d'
              %(content, ' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height, style_weight, output_semantic_mask, ' '.join(style_semantic_masks),semantic_masks_num_layers))

    # The sixth example: use the 'content_img_style_weight_mask" to control the degree of stylization for each pixel.

    content = 'stylize_examples/6-content.jpg'
    styles = ['stylize_examples/1-style.jpg']
    learning_rate = 10.0
    iterations = 1000
    width = 712
    height = 474
    content_img_style_weight_mask = 'stylize_examples/6-mask.jpg'

    checkpoint_output_str = 'stylize_examples/output_checkpoint/6_iter_%%s.jpg'
    output_str = 'stylize_examples/output/6_result.jpg'

    os.system('/home/xor/anaconda2/bin/python neural_style.py --content=%s --styles %s --learning-rate=%f '
              '--iterations=%d --checkpoint-output=%s --output=%s --width=%d --height=%d --content_img_style_weight_mask=%s'
              %(content, ' '.join(styles), learning_rate, iterations, checkpoint_output_str, output_str, width, height, content_img_style_weight_mask))
