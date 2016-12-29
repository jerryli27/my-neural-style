# The code mainly comes from https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py
import tensorflow as tf

from neural_util import conv2d_mirror_padding, conv2d_transpose_mirror_padding

WEIGHTS_INIT_STDEV = .1

# NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly by 4.
def net(image, mirror_padding = True, one_hot_style_vector = None, reuse = False):
    conv1 = _conv_layer(image, 32, 9, 1, mirror_padding = mirror_padding, name = 'conv1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv2 = _conv_layer(conv1, 64, 3, 2, mirror_padding = mirror_padding, name = 'conv2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv3 = _conv_layer(conv2, 128, 3, 2, mirror_padding = mirror_padding, name = 'conv3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid1 = _residual_block(conv3, 3, mirror_padding = mirror_padding, name = 'resid1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid2 = _residual_block(resid1, 3, mirror_padding = mirror_padding, name = 'resid2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid3 = _residual_block(resid2, 3, mirror_padding = mirror_padding, name = 'resid3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid4 = _residual_block(resid3, 3, mirror_padding = mirror_padding, name = 'resid4', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    resid5 = _residual_block(resid4, 3, mirror_padding = mirror_padding, name = 'resid5', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    # There are some errors in the gradient calculation due to mirror padding in conv transpose.
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, mirror_padding = False, name = 'conv_t1', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, mirror_padding = False, name = 'conv_t2', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, mirror_padding = mirror_padding, name = 'conv_t3', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    # assert image.get_shape().as_list() == preds.get_shape().as_list()
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True, mirror_padding = True, one_hot_style_vector = None, name = '', reuse = False):
    with tf.variable_scope('conv_layer' + name, reuse=reuse):
        weights_init = _conv_init_vars(net, num_filters, filter_size, name=name, reuse = reuse)
        strides_shape = [1, strides, strides, 1]
        if mirror_padding:
            net = conv2d_mirror_padding(net, weights_init, None, filter_size, stride=strides)
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = _instance_norm(net,name=name, one_hot_style_vector = one_hot_style_vector, reuse = reuse)
        if relu:
            # net = tf.nn.relu(net)
            net = tf.nn.elu(net)

        return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides, mirror_padding = True, one_hot_style_vector = None, name = '', reuse = False):
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
        net = _instance_norm(net,name=name, one_hot_style_vector = one_hot_style_vector, reuse = reuse)
        return tf.nn.elu(net)  # tf.nn.relu(net)

def _residual_block(net, filter_size=3, mirror_padding = True, name = '', one_hot_style_vector = None, reuse = False):
    tmp = _conv_layer(net, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_first', one_hot_style_vector = one_hot_style_vector, reuse = reuse)
    return net + _conv_layer(tmp, 128, filter_size, 1, mirror_padding = mirror_padding, name=name + '_second', relu=False, one_hot_style_vector = one_hot_style_vector, reuse = reuse)

def _instance_norm(net, name = '', one_hot_style_vector = None, reuse = False):
    with tf.variable_scope('instance_norm' + name, reuse=reuse):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        if one_hot_style_vector is None:
            var_shape = [channels]
        else:
            num_styles = one_hot_style_vector.get_shape().as_list()[1]
            var_shape = [num_styles, channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        # Try applying an abs on the sigma_sq. in theory it should always be positive but in practice due to inaccuracy in float calculation, it may be negative when the actual sigma is very small, which causes the output to be NaN sometimes.
        sigma_sq = tf.abs(sigma_sq)
        shift_init = tf.zeros(var_shape)
        shift = tf.get_variable('shift', initializer= shift_init)
        scale_init = tf.ones(var_shape)
        scale = tf.get_variable('scale', initializer= scale_init)
        if one_hot_style_vector is not None:
            shift = tf.matmul(one_hot_style_vector, shift)
            scale = tf.matmul(one_hot_style_vector, scale)
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
