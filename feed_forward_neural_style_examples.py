"""
This file contains code to run neural_style.py It shows a few usages of the function and output the
results.
"""

from general_util import *

if __name__=='__main__':
    # First download the required files.
    download_if_not_exist(
        'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
        'imagenet-vgg-verydeep-19.mat', 'Pretrained vgg 19')
    download_if_not_exist(
        'https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/1-style.jpg',
        'feed_forward_examples/1-style.jpg', 'Example style image')
    download_if_not_exist(
        'https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/2-style1.jpg',
        'feed_forward_examples/2-style.jpg', 'Example style image No.2')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/02_composition_vii.jpg',
        'feed_forward_examples/3-style.jpg', 'Example style image No.3')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/03_feathers.jpg',
        'feed_forward_examples/4-style.jpg', 'Example style image No.4')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/04_la_muse.jpg',
        'feed_forward_examples/5-style.jpg', 'Example style image No.5')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/05_mosaic.jpg',
        'feed_forward_examples/6-style.jpg', 'Example style image No.6')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/06_the_scream.jpg',
        'feed_forward_examples/7-style.jpg', 'Example style image No.7')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/07_udnie.jpg',
        'feed_forward_examples/8-style.jpg', 'Example style image No.8')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/08_wave.jpg',
        'feed_forward_examples/9-style.jpg', 'Example style image No.9')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/09_cubist_style.jpg',
        'feed_forward_examples/10-style.jpg', 'Example style image No.10')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/10_fur-style.jpg',
        'feed_forward_examples/11-style.jpg', 'Example style image No.11')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/01_candy.jpg',
        'feed_forward_examples/12-style.jpg', 'Example style image No.12')
    download_if_not_exist(
        'https://raw.githubusercontent.com/junrushao1994/fast-neural-style.tf/master/examples/inputs/content.jpg',
        'feed_forward_examples/1-content.jpg', 'Example content image No.1')

    # The first example is to use all the style images above to train one single feed forward neural network that is
    # capable of reproducing any of the provided styles and their combinations
    end_iterations_dict = {128: 10000, 256: 15000, 512: 20000}

    for hw in [128, 256, 512]:
        styles = ['feed_forward_examples/%d-style.jpg' % (i) for i in range(1, 13)]
        style_name = 'multi_style_feed_forward'

        learning_rate = 0.001
        final_goal_iterations = 20000  # Trying to train the model multiple times on different scales.
        current_end_iterations = end_iterations_dict[hw]
        batch_size = 1  # Optimally 16, but it ran out of memory.
        style_weight = 100
        content_weight = 5
        checkpoint_iterations = 500
        height = hw
        width = hw
        print_iteration = 100
        use_mrf = False
        use_johnson = True
        use_skip_noise_4 = False
        texture_synthesis_only = False
        do_restore_and_train = False if hw == 128 else True
        do_restore_and_generate = False
        multi_style_offset_only = False
        use_semantic_masks = False

        test_img = 'feed_forward_examples/1-content.jpg'

        style_or_texture_string = 'texture' if texture_synthesis_only else 'nstyle'
        texture_synthesis_only_string = '--texture_synthesis_only' if texture_synthesis_only else ''
        use_mrf_string = '--use_mrf' if use_mrf else ''
        johnson_or_pyramid_string = 'johnson' if use_johnson else ('skip_noise_4' if use_skip_noise_4 else 'pyramid')
        use_johnson_string = '--use_johnson' if use_johnson else ''
        use_skip_noise_4_string = '--use_skip_noise_4' if use_skip_noise_4 else ''
        do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
        do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
        multi_style_offset_only_string = '--multiple_styles_train_scale_offset_only' if multi_style_offset_only else ''
        use_semantic_masks_string = '--use_semantic_masks' if use_semantic_masks else ''


        # TODO: change this dir
        checkpoint_output = 'feed_forward_examples/output_checkpoint/feed_forward_example_1-stylenum-%s-iter-%s.jpg'
        output = 'feed_forward_examples/output/feed_forward_example_1-stylenum-%s.jpg'
        model_save_dir = 'model/feed_forward_example_1/'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)  # TODO: add %s content_img_style_weight_mask_string to the model_save_dir

        os.system(
            'python feed_forward_neural_style.py --styles %s %s --learning_rate=%f '
            '--iterations=%d --batch_size=%d %s %s %s %s --style_weight=%d --content_weight=%d '
            '--checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s '
            '--output=%s --model_save_dir=%s --print_iterations=%d %s %s %s'
            % (' '.join(styles), texture_synthesis_only_string, learning_rate, current_end_iterations,
               batch_size, use_mrf_string, use_johnson_string, use_skip_noise_4_string, multi_style_offset_only_string,
               style_weight, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output,
               model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string,
               use_semantic_masks_string))
