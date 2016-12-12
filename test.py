# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

from general_util import *
styles = ['van_gogh/style.png']
style_folder = 'style_compressed/claude_monet_2/' # DUMMY
style_name = 'van_gogh'
style_semantic_mask_dirs=['van_gogh/style_mask_segmented_0.jpg', 'van_gogh/style_mask_segmented_1.jpg', 'van_gogh/style_mask_segmented_2.jpg', 'van_gogh/style_mask_segmented_3.jpg']

learning_rate=0.001
iterations=80000
batch_size=1 # Optimally 16, but it ran out of memory. #TODO: change it to 8.
style_weight=200
content_weight=5
checkpoint_iterations=100
width = 256
height = 256
print_iteration = 100
use_mrf = False
use_johnson = True
texture_synthesis_only = False
do_restore_and_train = False
multi_style_offset_only = False
use_semantic_masks = True
semantic_masks_num_layers = 4

test_img = 'test_masks/' if use_semantic_masks else 'source_compressed/chicago.jpg'


style_or_texture_string = 'texture' if texture_synthesis_only else 'nstyle'
texture_synthesis_only_string = '--texture_synthesis_only' if texture_synthesis_only else ''
use_mrf_string = '--use_mrf' if use_mrf else ''
johnson_or_pyramid_string = 'johnson' if use_johnson else 'pyramid'
use_johnson_string = '--use_johnson' if use_johnson else ''
do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
multi_style_offset_only_string = '--multiple_styles_train_scale_offset_only' if multi_style_offset_only else ''
use_semantic_masks_string = '--use_semantic_masks' if use_semantic_masks else ''

checkpoint_output='output_checkpoint/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-multi_style_offset_only-%s-style-%d-content-%d-stylenum-%%s_%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(multi_style_offset_only), style_weight, content_weight)
output='output/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-multi_style_offset_only-%s-style-%d-content-%d-stylenum-%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(multi_style_offset_only), style_weight, content_weight)
model_save_dir='model/genstyle-%s-%s-iter-batchsize-%d-%d-lr-%f-use_mrf-%s-%s-multi_style_offset_only-%s-style-%d-content-%d/' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, str(multi_style_offset_only), style_weight, content_weight)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/faster_neural_style.py --style_folder=%s --styles %s %s --learning_rate=%f --iterations=%d --batch_size=%d %s %s %s --style_weight=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s --style_semantic_mask_dirs %s --semantic_masks_num_layers=%d'
          % (style_folder, ' '.join(styles), texture_synthesis_only_string, learning_rate, iterations, batch_size, use_mrf_string, use_johnson_string, multi_style_offset_only_string, style_weight, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output, model_save_dir, print_iteration, do_restore_and_train_string, use_semantic_masks_string, ' '.join(style_semantic_mask_dirs), semantic_masks_num_layers))
