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
def net(image, reuse=False):
    prev_layer = image
    prev_layer_list = [image]

    with tf.variable_scope('adv_net', reuse=reuse):
        for i in range(len(CONV_DOWN_NUM_FILTERS)):
            # Do not normalize the first layer. According to https://arxiv.org/abs/1511.06434.
            current_layer = conv_layer(prev_layer, num_filters=CONV_DOWN_NUM_FILTERS[i],
                                       filter_size=CONV_DOWN_KERNEL_SIZES[i], strides=CONV_DOWN_STRIDES[i],
                                       mirror_padding=False, norm='batch_norm' if i != 0 else '', name='conv_down_%d' %i, reuse=reuse)
            prev_layer = current_layer
            prev_layer_list.append(current_layer)


        # The output is the unnormalized probability for each class (we have two here). the loss should be
        # sparse_softmax_cross_entropy_with_logits.
        final = fully_connected(prev_layer, 2, name='fc', reuse=reuse)
    return final


def get_net_all_variables():
    if '0.12.0' in tf.__version__:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adv_net')
    else:
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope='adv_net')