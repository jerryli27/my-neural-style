"""
This file implements the unet generator network according to https://arxiv.org/pdf/1505.04597.pdf
"""

from conv_util import *

WEIGHTS_INIT_STDEV = .1
CONV_DOWN_NUM_FILTERS=[32, 64, 64, 128, 128, 256, 256, 512, 512]
CONV_DOWN_KERNEL_SIZES=[3, 4, 3, 4, 3, 4, 3, 4, 3]
CONV_DOWN_STRIDES=[1, 2, 1, 2, 1, 2, 1, 2, 1]

CONV_UP_NUM_FILTERS=[512, 256, 256, 128, 128, 64, 64, 32, 3]
CONV_UP_KERNEL_SIZES=[4, 3, 4, 3, 4, 3, 4, 3, 3]
CONV_UP_STRIDES=[2, 1, 2, 1, 2, 1, 2, 1, 1]


# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding=False, reuse=False):
    # TODO: maybe delete mirror padding because it's causing back prop to complain.
    image_shape = image.get_shape().as_list()
    assert len(image_shape) == 4 and image_shape[1] >= 32 and image_shape[2] >= 32  # Otherwise the conv fails.
    prev_layer = image
    prev_layer_list = [image]

    with tf.variable_scope('unet', reuse=reuse):
        for i in range(len(CONV_DOWN_NUM_FILTERS)):
            current_layer = conv_layer(prev_layer, num_filters=CONV_DOWN_NUM_FILTERS[i],
                                       filter_size=CONV_DOWN_KERNEL_SIZES[i], strides=CONV_DOWN_STRIDES[i],
                                       mirror_padding=mirror_padding, name='conv_down_%d' %i, reuse=reuse)
            prev_layer = current_layer
            prev_layer_list.append(current_layer)


        for i in range(len(CONV_UP_NUM_FILTERS) - 1):

            if i % 2 == 0:
                concat_layer = tf.concat(3, [prev_layer_list[-i-1], prev_layer])
                current_layer = conv_tranpose_layer(concat_layer, num_filters=CONV_UP_NUM_FILTERS[i],
                                                    filter_size=CONV_UP_KERNEL_SIZES[i], strides=CONV_UP_STRIDES[i],
                                                    mirror_padding=mirror_padding, name='conv_up_%d' %i, reuse=reuse)
                prev_layer = current_layer
            else:
                current_layer = conv_layer(prev_layer, num_filters=CONV_UP_NUM_FILTERS[i],
                                                    filter_size=CONV_UP_KERNEL_SIZES[i], strides=CONV_UP_STRIDES[i],
                                                    mirror_padding=mirror_padding, name='conv_up_%d' %i, reuse=reuse)
                prev_layer = current_layer


        # Do a final convolution with output dimension = 3 and stride 1.
        weights_init = conv_init_vars(prev_layer, CONV_UP_NUM_FILTERS[-1], CONV_UP_KERNEL_SIZES[-1], name='final_conv', reuse=reuse)
        strides_shape = [1, CONV_UP_STRIDES[-1], CONV_UP_STRIDES[-1], 1]
        final = tf.nn.conv2d(prev_layer, weights_init, strides_shape, padding='SAME')

        # Do sanity check.
        final_shape = final.get_shape().as_list()
        if not (image_shape[0] == final_shape[0] and image_shape[1] == final_shape[1] and image_shape[2] == final_shape[2]):
            print('image_shape and final_shape are different. image_shape = %s and final_shape = %s' %(str(image_shape), str(final_shape)))
            raise AssertionError

    return final