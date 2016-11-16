# The code mainly comes from https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
import tensorflow as tf

from neural_util import conv2d_mirror_padding, conv2d_transpose_mirror_padding

WEIGHTS_INIT_STDEV = .1

# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding = True, reuse = False):
    conv1 = _conv_layer(image, 32, 9, 1, mirror_padding = mirror_padding, name = 'conv1', reuse = reuse)
    conv2 = _conv_layer(conv1, 64, 3, 2, mirror_padding = mirror_padding, name = 'conv2', reuse = reuse)
    conv3 = _conv_layer(conv2, 128, 3, 2, mirror_padding = mirror_padding, name = 'conv3', reuse = reuse)
    resid1 = _residual_block(conv3, 3, mirror_padding = mirror_padding, name = 'resid1', reuse = reuse)
    resid2 = _residual_block(resid1, 3, mirror_padding = mirror_padding, name = 'resid2', reuse = reuse)
    resid3 = _residual_block(resid2, 3, mirror_padding = mirror_padding, name = 'resid3', reuse = reuse)
    resid4 = _residual_block(resid3, 3, mirror_padding = mirror_padding, name = 'resid4', reuse = reuse)
    resid5 = _residual_block(resid4, 3, mirror_padding = mirror_padding, name = 'resid5', reuse = reuse)
    # There are some errors in the gradient calculation due to mirror padding in conv transpose.
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, mirror_padding = False, name = 'conv_t1', reuse = reuse)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, mirror_padding = False, name = 'conv_t2',reuse = reuse)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, mirror_padding = mirror_padding, name = 'conv_t3', reuse = reuse)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    # assert image.get_shape().as_list() == preds.get_shape().as_list()
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True, mirror_padding = True, name = '', reuse = False):
    with tf.variable_scope('conv_layer' + name, reuse=reuse):
        weights_init = _conv_init_vars(net, num_filters, filter_size, name=name, reuse = reuse)
        strides_shape = [1, strides, strides, 1]
        if mirror_padding:
            net = conv2d_mirror_padding(net, weights_init, None, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = _instance_norm(net, reuse = reuse, name=name)
        if relu:
            net = tf.nn.relu(net)

        return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides, mirror_padding = True, name = '', reuse = False):
    with tf.variable_scope('conv_tranpose_layer' + name, reuse=reuse):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True,name=name, reuse = reuse)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.pack(new_shape)
        strides_shape = [1,strides,strides,1]

        if mirror_padding:
            net = conv2d_transpose_mirror_padding(net, weights_init, None, tf_shape, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net,name=name, reuse = reuse)
        return tf.nn.relu(net)

def _residual_block(net, filter_size=3, mirror_padding = True, name = '', reuse = False):
    tmp = _conv_layer(net, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_first', reuse = reuse)
    return net + _conv_layer(tmp, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_second', relu=False, reuse = reuse)

def _instance_norm(net, name = '',reuse = False):
    with tf.variable_scope('instance_norm' + name, reuse=reuse):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift_init = tf.zeros(var_shape)
        shift = tf.get_variable('shift', initializer= shift_init)
        scale_init = tf.ones(var_shape)
        scale = tf.get_variable('scale', initializer= scale_init)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False,name = '', reuse = False):
    with tf.variable_scope('conv_init_vars' + name, reuse=reuse):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
        weights_initializer = tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
        weights_init = tf.get_variable('weights_init', dtype=tf.float32, initializer=weights_initializer)
        return weights_init
