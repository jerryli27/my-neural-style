#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is for running a sanity check that make sure the model can overfit one single image and generate a
# perfectly satisfactory result.

from general_util import *

learning_rate=0.1 # if colorful_img model, the learning rate the paper was using was 3 * 10^-5. They did not
# saytheir batch size.
# iterations=800000# 800000
epochs = 1000
batch_size=1  # The larger the better.
content_weight=5
checkpoint_iterations=20
height = 256
width = 256
generator_network='unet_both'
output_mode = 'lab'
print_iteration = 1
do_restore_and_train = False  # True
do_restore_and_generate = False
use_adversarial_net = False
# use_adversarial_net_real = True
use_hint = False

test_img = '378688_256.jpg'#u'/home/ubuntu/pixiv/pixiv_testing/骨董屋・三千世界の女主人_12746957.jpg'
# #'source_compressed/chicago.jpg'
test_img_hint = '378688_256_hint.png'

do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
use_hint_string = '--use_hint' if use_hint else ''
# restore_from_noadv_to_adv_string = '--restore_from_noadv_to_adv' if use_adversarial_net_real != use_adversarial_net else ''

content_folder = "'sanity_check_folder/'"

checkpoint_output='output_checkpoint/colorsketches-sanity_check-content_%s.jpg'
output='output/colorsketches-sanity_check-content.jpg'
model_save_dir='model/colorsketches-sanity_check/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir

# For utf 8
# os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('/home/xor/anaconda2/bin/python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --num_epochs=%d '
          '--batch_size=%d --generator_network=%s --output_mode=%s --content_folder=%s '
          '--content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
          % (learning_rate, epochs, batch_size, generator_network, output_mode, content_folder, content_weight,
             checkpoint_iterations, width, height,
             checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))

"""
Trained directories:
colorsketches-adv_net-False-hint-True-iter-batchsize-800000-8-lr-0.000200-content-5
"""