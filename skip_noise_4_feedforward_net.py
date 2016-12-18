# The code is the tensorflow implementation of https://github.com/DmitryUlyanov/online-neural-doodle/blob/master/models/skip_noise_4.lua
# Decoder-encoder like model with skip-connections
# and additional noise inputs.

import tensorflow as tf

from neural_util import conv2d_mirror_padding, conv2d_transpose_mirror_padding

WEIGHTS_INIT_STDEV = .1
nums_3x3down = [4, 4, 4, 4,4]
nums_1x1 = [4, 4, 4, 4, 4]
nums_noise=[16, 16, 16, 16, 16]
nums_3x3up = [16, 32, 64, 128,128]


# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding=True, reuse=False):
    # TODO: check if we need mirror padding
    prev_layer = image
    prev_layer_list = [image]
    skip_noise_list = []
    skip_concat_list = []

    with tf.variable_scope('skip_noise_4', reuse=reuse):
        for i in range(len(nums_3x3down)):
            # deeper first applies a conv with kernel = 3 and stride = 2
            deeper_1 = _conv_layer(prev_layer, nums_3x3down[i], 3, 2, mirror_padding=mirror_padding, name='deeper_1_%d' %i, reuse=reuse)
            deeper_2 = _conv_layer(deeper_1, nums_3x3down[i], 3, 1, mirror_padding=mirror_padding, name='deeper_2_%d' %i, reuse=reuse)
            prev_layer_list.append(deeper_2)
            prev_layer = deeper_2

        for i in range(len(nums_3x3down)):

            # Skip just applies a 1d conv.
            skip_conv_prev = _conv_layer(prev_layer_list[i], nums_1x1[i], 1, 1, relu=False, mirror_padding=mirror_padding, name='skip_conv_prev_%d' %i, reuse=reuse)
            skip_conv_prev_shape = map(lambda s: s.value, skip_conv_prev.get_shape())
            # Then add a noise layer
            skip_noise = tf.placeholder(tf.float32, shape=(skip_conv_prev_shape[0], skip_conv_prev_shape[1], skip_conv_prev_shape[2], nums_noise[i]), name = 'skip_noise_%d' %i)
            skip_noise_list.append(skip_noise)
            skip_concat = tf.concat(3,[skip_conv_prev,skip_noise], name='skip_concat_%d' %i)
            skip_concat_list.append(skip_concat)

        # Next is deconv.
        prev_layer = prev_layer_list[-1]
        for i in range(len(nums_3x3up)):
            skip_concat_height, skip_concat_width = skip_concat_list[-i - 1].get_shape().as_list()[1:3]
            upsampled = tf.image.resize_nearest_neighbor(prev_layer, [skip_concat_height, skip_concat_width])
            deconv_1 = _conv_layer(upsampled, nums_3x3up[i], 3, 1, mirror_padding=mirror_padding, name='deconv_1_%d' %i, reuse=reuse)
            deconv_2 = _conv_layer(deconv_1, nums_3x3up[i], 1, 1, mirror_padding=mirror_padding, name='deconv_2_%d' %i, reuse=reuse)
            skip_deeper_concat = tf.concat(3, [skip_concat_list[-i - 1], deconv_2], name='skip_deeper_concat_%d' % i)
            prev_layer = skip_deeper_concat

        # Do a final convolution with output dimension = 3 and stride 1.
        weights_init = _conv_init_vars(prev_layer, 3, 1, name='final_conv', reuse=reuse)
        strides_shape = [1, 1, 1, 1]
        final = tf.nn.conv2d(prev_layer, weights_init, strides_shape, padding='SAME')

        # Do sanity check.
        image_shape = image.get_shape().as_list()
        final_shape = final.get_shape().as_list()
        assert len(image_shape) == len(final_shape) and sorted(image_shape) == sorted(final_shape)

    return final, skip_noise_list


def _conv_layer(net, num_filters, filter_size, strides, relu=True, mirror_padding=True, name='', reuse=False):
    with tf.variable_scope('conv_layer' + name, reuse=reuse):
        weights_init = _conv_init_vars(net, num_filters, filter_size, name=name, reuse=reuse)
        strides_shape = [1, strides, strides, 1]
        if mirror_padding:
            net = conv2d_mirror_padding(net, weights_init, None, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = _instance_norm(net, reuse=reuse, name=name)
        if relu:
            net = tf.nn.relu(net) # In original file they're using leaky relu

        return net


def _conv_tranpose_layer(net, num_filters, filter_size, strides, mirror_padding=True, name='', reuse=False):
    with tf.variable_scope('conv_tranpose_layer' + name, reuse=reuse):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True, name=name, reuse=reuse)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.pack(new_shape)
        strides_shape = [1, strides, strides, 1]

        if mirror_padding:
            net = conv2d_transpose_mirror_padding(net, weights_init, None, tf_shape, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net, name=name, reuse=reuse)
        return tf.nn.relu(net)


def _residual_block(net, filter_size=3, mirror_padding=True, name='', reuse=False):
    tmp = _conv_layer(net, 128, filter_size, 1, mirror_padding=mirror_padding, name=name + '_first', reuse=reuse)
    return net + _conv_layer(tmp, 128, filter_size, 1, mirror_padding=mirror_padding, name=name + '_second', relu=False,
                             reuse=reuse)


def _instance_norm(net, name='', reuse=False):
    with tf.variable_scope('instance_norm' + name, reuse=reuse):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
        shift_init = tf.zeros(var_shape)
        shift = tf.get_variable('shift', initializer=shift_init)
        scale_init = tf.ones(var_shape)
        scale = tf.get_variable('scale', initializer=scale_init)
        epsilon = 1e-3
        normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
        return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False, name='', reuse=False):
    with tf.variable_scope('conv_init_vars' + name, reuse=reuse):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
        weights_initializer = tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
        weights_init = tf.get_variable('weights_init', dtype=tf.float32, initializer=weights_initializer)
        return weights_init