"""
This is the adversarial net that tries to distinguish the original image from the generated one. It is used with the
unet. For more information, please check http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d.
"""

from conv_util import *

WEIGHTS_INIT_STDEV = .1
CONV_DOWN_NUM_FILTERS=[32, 32, 64, 64, 128, 128, 256]
CONV_DOWN_KERNEL_SIZES=[4, 3, 4, 3, 4, 3, 4]
CONV_DOWN_STRIDES=[2, 1, 2, 1, 2, 1, 2]


# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding=False, reuse=False):
    # TODO: check if we need mirror padding
    prev_layer = image
    prev_layer_list = [image]

    with tf.variable_scope('adv_net', reuse=reuse):
        for i in range(len(CONV_DOWN_NUM_FILTERS)):
            current_layer = conv_layer(prev_layer, num_filters=CONV_DOWN_NUM_FILTERS[i],
                                       filter_size=CONV_DOWN_KERNEL_SIZES[i], strides=CONV_DOWN_STRIDES[i],
                                       mirror_padding=mirror_padding, name='conv_down_%d' %i, reuse=reuse)
            prev_layer = current_layer
            prev_layer_list.append(current_layer)


        fc = fully_connected(prev_layer, 2, name='fc', reuse=reuse) # The initial weight here might be a little bit tricky. The original specified the wscale.
        # TODO: here's one layer I added versus the original version described. As of my understanding, the output
        # is an indicator vector indicating whether it is original or fake. So I added a softmax layer.
        final = tf.nn.softmax(fc)
    return final


def get_net_all_variables():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adv_net')