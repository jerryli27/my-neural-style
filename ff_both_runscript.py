# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

from general_util import *
styles = ['van_gogh/style.png']
style_folder = '/home/ubuntu/pixiv/pixiv_training_filtered/'
style_name = 'pixiv_training_filtered'
style_semantic_mask_dirs=['van_gogh/bw_masks/style_mask_segmented_0.jpg', 'van_gogh/bw_masks/style_mask_segmented_1.jpg', 'van_gogh/bw_masks/style_mask_segmented_2.jpg', 'van_gogh/bw_masks/style_mask_segmented_3.jpg']
mask_folder= 'random_masks/' # '4_color_masks/'

learning_rate=0.001
iterations=80000
batch_size=4 # Optimally 16, but it ran out of memory. #TODO: change it to 8.
style_weight=50
content_weight=5
checkpoint_iterations=500
height = 256
width = 256
print_iteration = 100
use_mrf = False
use_johnson = True
use_skip_noise_4 = False
texture_synthesis_only = False # True
do_restore_and_train = False
do_restore_and_generate = False
multi_style_offset_only = False
use_semantic_masks = False # False
semantic_masks_num_layers = 4

test_img = 'test_masks/' if use_semantic_masks else 'source_compressed/256/sea.jpg' #'source_compressed/chicago.jpg'
content_img_style_weight_mask = '' #'source_compressed/sea_test/sea_style_weight_mask_2.jpg'


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
content_img_style_weight_mask_string = 'swmask-test_mask-' if content_img_style_weight_mask!='' else ''

checkpoint_output='output_checkpoint/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-mask-%s-multi_style_offset_only-%s-%sstyle-%d-content-%d-stylenum-%%s_%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(use_semantic_masks), str(multi_style_offset_only), content_img_style_weight_mask_string, style_weight, content_weight)
output='output/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-mask-%s-multi_style_offset_only-%s-%sstyle-%d-content-%d-stylenum-%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(use_semantic_masks), str(multi_style_offset_only), content_img_style_weight_mask_string, style_weight, content_weight)
model_save_dir='model/genstyle-%s-%s-iter-batchsize-%d-%d-lr-%f-use_mrf-%s-%s-mask-%s-multi_style_offset_only-%s-style-%d-content-%d/' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(use_semantic_masks), str(multi_style_offset_only), style_weight, content_weight)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/ff_both_neural_style_experimental.py --style_folder=%s --styles %s %s --learning_rate=%f --iterations=%d --batch_size=%d %s %s %s %s --style_weight=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s --style_semantic_mask_dirs %s --semantic_masks_num_layers=%d --mask_folder=%s --content_img_style_weight_mask=%s'
          % (style_folder, ' '.join(styles), texture_synthesis_only_string, learning_rate, iterations, batch_size, use_mrf_string, use_johnson_string, use_skip_noise_4_string, multi_style_offset_only_string, style_weight, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_semantic_masks_string, ' '.join(style_semantic_mask_dirs), semantic_masks_num_layers, mask_folder, content_img_style_weight_mask))
