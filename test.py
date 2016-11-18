# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

import os
from general_util import *

# content= 'source_compressed/512/sea_512.jpg'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']



# contents= ['source_compressed/256/sea.jpg']
# #contents= ['source_compressed/my256/%d.jpg' % i for i in range(1,17)]
# contents_name = '256'
# styles = ['style_compressed/van_gogh/self_portrait.jpg', 'style_compressed/van_gogh/starry_sky.jpg']
# styles = ['van_gogh/starry_sky256.jpg']
# style_name = 'van_gogh_starry_sky'
# styles = ['style_compressed/claude_monet/512/1.jpg','style_compressed/claude_monet/512/2.jpg','style_compressed/claude_monet/512/3.jpg','style_compressed/claude_monet/512/4.jpg']

styles = ['van_gogh/starry_sky256.jpg'] # DUMMY
style_name = 'shirobako'

learning_rate=0.001
iterations=80000
batch_size=8 # Optimally 16, but it ran out of memory.
style_weight=200
content_weight=5
checkpoint_iterations=500
width = 256
height = 256
print_iteration = 100
use_mrf = False
use_johnson = True
texture_synthesis_only = False

style_or_texture_string = 'texture' if texture_synthesis_only else 'nstyle'
texture_synthesis_only_string = '--texture_synthesis_only' if texture_synthesis_only else ''
use_mrf_string = '--use_mrf' if use_mrf else ''
johnson_or_pyramid_string = 'johnson' if use_johnson else 'pyramid'
use_johnson_string = '--use_johnson' if use_johnson else ''
do_restore_and_train = False
do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''

# contents= ['source_compressed/my256/%d.jpg' % i for i in range(1,17)]
# contents_name = 'my256'
test_img = 'source_compressed/chicago.jpg'
texture_synthesis_only = False
style_or_texture_string = 'texture' if texture_synthesis_only else 'nstyle'
texture_synthesis_only_string = '--texture_synthesis_only' if texture_synthesis_only else ''
use_mrf = False
use_mrf_string = '--use_mrf' if use_mrf else ''
do_restore_and_train = False
do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''

# TODO: Don't forget to delete mirror.
checkpoint_output='output_checkpoint/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-style-%d-content-%d-stylenum-%%s_%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, style_weight, content_weight)
output='output/genstyle-%s-%s-iter-%d-batchsize-%d-lr-%f-use_mrf-%s-%s-style-%d-content-%d-stylenum-%%s.jpg' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, style_weight, content_weight)
model_save_dir='model/genstyle-%s-%s-iter-batchsize-%d-%d-lr-%f-use_mrf-%s-%s-style-%d-content-%d/' % (style_or_texture_string, style_name, iterations, batch_size, learning_rate, str(use_mrf), johnson_or_pyramid_string, style_weight, content_weight)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/faster_neural_style.py --styles %s %s --learning_rate=%f --iterations=%d --batch_size=%d %s %s --style_weight=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --output=%s --model_save_dir=%s --print_iterations=%d %s'
          % (' '.join(styles), texture_synthesis_only_string, learning_rate, iterations, batch_size, use_mrf_string, use_johnson_string, style_weight, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output, model_save_dir, print_iteration, do_restore_and_train_string))





# import cv2
# cap = cv2.VideoCapture(0)
# ret = cap.set(3, 1280)
# ret = cap.set(4, 960)
# ret, frame = cap.read()
# print('The dimension of this camera is : %d x %d' % (frame.shape[0], frame.shape[1]))