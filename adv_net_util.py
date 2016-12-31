"""
This is the advisor net that give hints about the colors of the original image. It is used with the unet. For more
information, please check http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d.
# TODO: is this really necessary to have a deep neural net just to give hints? It looks like it's working well, but
is there other ways that requires less training perhaps?
"""

from conv_util import *

WEIGHTS_INIT_STDEV = .1
CONV_DOWN_NUM_FILTERS=[32, 32, 64, 64, 128, 128, 256]
CONV_DOWN_KERNEL_SIZES=[4, 3, 4, 3, 4, 3, 4]
CONV_DOWN_STRIDES=[2, 1, 2, 1, 2, 1, 2]


# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding=True, reuse=False):
    # TODO: check if we need mirror padding
    image_shape = image.get_shape().as_list()
    prev_layer = image
    prev_layer_list = [image]

    with tf.variable_scope('unet', reuse=reuse):
        for i in range(len(CONV_DOWN_NUM_FILTERS)):
            current_layer = conv_layer(prev_layer, num_filters=CONV_DOWN_NUM_FILTERS[i],
                                       filter_size=CONV_DOWN_KERNEL_SIZES[i], strides=CONV_DOWN_STRIDES[i],
                                       mirror_padding=mirror_padding, name='conv_down_%d' %i, reuse=reuse)
            prev_layer = current_layer
            prev_layer_list.append(current_layer)


        final = fully_connected(prev_layer, 2, name='fc', reuse=reuse) # The initial weight here might be a little bit tricky. The original specified the wscale.

        # # Do sanity check.
        # final_shape = final.get_shape().as_list()
        # if not (image_shape[0] == final_shape[0] and image_shape[1] == final_shape[1] and image_shape[2] == final_shape[2]):
        #     print('image_shape and final_shape are different. image_shape = %s and final_shape = %s' %(str(image_shape), str(final_shape)))
        #     raise AssertionError

    return final


