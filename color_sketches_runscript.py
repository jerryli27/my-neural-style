# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

from general_util import *

learning_rate=0.001
iterations=80000
batch_size=1 # Optimally 16, but it ran out of memory. #TODO: change it to 8.
content_weight=5
checkpoint_iterations=500
height = 128
width = 128
print_iteration = 100
do_restore_and_train = False
do_restore_and_generate = False

test_img = 'source_compressed/256/sea.jpg' #'source_compressed/chicago.jpg'

do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''

checkpoint_output='output_checkpoint/colorsketches-iter-%d-batchsize-%d-lr-%f-content-%d-stylenum-%%s_%%s.jpg' % (iterations, batch_size, learning_rate, content_weight)
output='output/colorsketches-iter-%d-batchsize-%d-lr-%f-content-%d-stylenum-%%s.jpg' % (iterations, batch_size, learning_rate, content_weight)
model_save_dir='model/colorsketches-iter-batchsize-%d-%d-lr-%f-content-%d/' % (iterations, batch_size, learning_rate, content_weight)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir

# NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
os.system('python ~/PycharmProjects/my-neural-style/faster_neural_style.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s'
          % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string))
