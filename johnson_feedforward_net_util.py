# The code mainly comes from https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
from conv_util import *


# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding = True, one_hot_style_vector = None, reuse = False):
    conv1 = conv_layer(image, 32, 9, 1, mirror_padding = mirror_padding, name ='conv1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv2 = conv_layer(conv1, 64, 3, 2, mirror_padding = mirror_padding, name ='conv2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv3 = conv_layer(conv2, 128, 3, 2, mirror_padding = mirror_padding, name ='conv3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid1 = residual_block(conv3, 3, mirror_padding = mirror_padding, name ='resid1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid2 = residual_block(resid1, 3, mirror_padding = mirror_padding, name ='resid2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid3 = residual_block(resid2, 3, mirror_padding = mirror_padding, name ='resid3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid4 = residual_block(resid3, 3, mirror_padding = mirror_padding, name ='resid4', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid5 = residual_block(resid4, 3, mirror_padding = mirror_padding, name ='resid5', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    # There are some errors in the gradient calculation due to mirror padding in conv transpose.
    conv_t1 = conv_tranpose_layer(resid5, 64, 3, 2, mirror_padding = False, name ='conv_t1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv_t2 = conv_tranpose_layer(conv_t1, 32, 3, 2, mirror_padding = False, name ='conv_t2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv_t3 = conv_layer(conv_t2, 3, 9, 1, relu=False, mirror_padding = mirror_padding, name ='conv_t3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    # assert image.get_shape().as_list() == preds.get_shape().as_list()
    return preds

def get_johnson_scale_offset_var():
    all_var = tf.all_variables()
    scale_offset_variables = []
    for var in all_var:
        if 'scale' in var.name or 'shift' in var.name:
            scale_offset_variables.append(var)
    if len(scale_offset_variables) !=  3 * 2 + 5 * 2 * 2 + 3 * 2:
        print('The number of scale offset variables is wrong. ')
        raise AssertionError
    return scale_offset_variables
