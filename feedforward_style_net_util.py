import scipy.misc
import tensorflow as tf
from tensorflow.python.ops import math_ops

import neural_util
import vgg
from general_util import *


# TODO: don't forget to change the normalization back to instantce norm.

def generator_net_n_styles(input_noise_z, input_style_placeholder, reuse=False):
    """
    This function is a generator. It takes a list of tensors as input, and outputs a tensor with a given width and
    height. The output should be similar in content to the content image and style to the style image.
    :param input_noise_z: A list of tensors seeded with noise.
    :param input_style_placeholder: A one-hot tensor indicating which style we chose.
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return: the full sized generated image/texture.
    """

    # Different from Ulyanov et el, in the original paper z_k is the smallest input and z_1 is the largest.
    # Here we reverse the order
    with tf.variable_scope('texture_nets', reuse=reuse):
        noise_joined = input_noise_z[0]
        current_channels = 8
        channel_step_size = 8
        for noise_layer in input_noise_z[1:]:
            low_res = conv_block('block_low_%d' % current_channels, noise_joined, input_style_placeholder,
                                 current_channels, reuse=reuse)
            high_res = conv_block('block_high_%d' % current_channels, noise_layer, input_style_placeholder,
                                  channel_step_size, reuse=reuse)
            current_channels += channel_step_size
            noise_joined = join_block('join_%d' % current_channels, low_res, high_res)
        final_chain = conv_block("output_chain", noise_joined, input_style_placeholder, current_channels, reuse=reuse)
        return conv_relu_layers("output", final_chain, input_style_placeholder, kernel_size=1, out_channels=3,
                                reuse=reuse)


def conv_block(name, input_layer, input_style_placeholder, out_channels, reuse=False):
    """
    Each convolution block in Figure 2 contains three convo-
    lutional layers, each of which is followed by a ReLU acti-
    vation layer. The convolutional layers contain respectively
    3 x 3, 3 x 3 and 1 x 1 filters. Filers are computed densely
    (stride one) and applied using circular convolution to re-
    move boundary effects, which is appropriate for textures.
    :param name:
    :param input_layer:
    :param out_channels:
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        block1 = conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3,
                                  out_channels=out_channels, reuse=reuse)
        block2 = conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels,
                                  reuse=reuse)
        block3 = conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels,
                                  reuse=reuse)
    return block3


def conv_relu_layers(name, input_layer, input_style_placeholder, kernel_size, out_channels, relu_leak=0.01,
                     reuse=False):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Instance Norm
        - LeakyReLu
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    """

    with tf.variable_scope(name, reuse=reuse):
        in_channels = input_layer.get_shape().as_list()[-1]
        # per https://arxiv.org/abs/1610.07629
        weights = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels], tf.float32,
                                  tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [out_channels], tf.float32,
                                 tf.constant_initializer(dtype=tf.float32))
        conv = neural_util.conv2d_mirror_padding(input_layer, weights, biases, kernel_size)
        norm = spatial_batch_norm(conv, input_style_placeholder, reuse=reuse)
        relu = tf.nn.elu(norm, 'elu')  # Note: original paper uses leaky ReLU. ELU also seem to work.
        return relu


def join_block(name, lower_res_layer, higher_res_layer):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    A block that combines two resolutions by upsampling the lower, batchnorming both, and concatting.
    """
    with tf.variable_scope(name):
        upsampled = tf.image.resize_nearest_neighbor(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        # No need to normalize here. According to https://arxiv.org/abs/1610.07629  normalize only after convolution.
        # return tf.concat(3, [upsampled, higher_res_layer])

        # According to https://arxiv.org/abs/1603.03417 figure 8, we need to normalize after join block.
        # batch_norm_lower = spatial_batch_norm(upsampled, 'normLower')
        # batch_norm_higher = spatial_batch_norm(higher_res_layer, 'normHigher')
        # return tf.concat(3, [batch_norm_lower, batch_norm_higher])




        """
        We found that training benefited signif-
        icantly from inserting batch normalization layers (Ioffe
        & Szegedy, 2015) right after each convolutional layer
        and, most importantly, right before the concatenation lay-
        ers, since this balances gradients travelling along different
        branches of the network.
        """
        # According to https://arxiv.org/abs/1603.03417 figure 8, we need to normalize after join block.
        batch_norm_lower = spatial_batch_norm(upsampled, None, 'normLower')
        batch_norm_higher = spatial_batch_norm(higher_res_layer, None, 'normHigher')
        return tf.concat(3, [batch_norm_lower, batch_norm_higher])


def get_all_layers_generator_net_n_styles(input_noise_z, input_style_placeholder):
    """
    This function takes a list of tensors as input, and outputs a tensor with a given width and height. The output
    should be similar in style to the target image.
    :param input_noise_z: A list of tensors seeded with noise.
    :param input_style_placeholder: A one-hot tensor indicating which style we chose.
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return: the full sized generated image/texture.
    """

    # Different from Ulyanov et el, in the original paper z_k is the smallest input and z_1 is the largest.
    # Here we reverse the order
    ret = {}
    with tf.variable_scope('texture_nets', reuse=True):
        noise_joined_layers = [input_noise_z[0]]
        current_channels = 8
        channel_step_size = 8
        for noise_layer in input_noise_z[1:]:
            low_res = conv_block('block_low_%d' % current_channels, noise_joined_layers[-1], input_style_placeholder,
                                 current_channels, reuse=True)
            high_res = conv_block('block_high_%d' % current_channels, noise_layer, input_style_placeholder,
                                  channel_step_size, reuse=True)
            low_res_all_layers = get_all_layers_conv_block('block_low_%d' % current_channels, noise_joined_layers[-1],
                                                           input_style_placeholder, current_channels)
            high_res_all_layers = get_all_layers_conv_block('block_high_%d' % current_channels, noise_layer,
                                                            input_style_placeholder, channel_step_size)

            current_channels += channel_step_size
            noise_joined_layers.append(join_block('join_%d' % current_channels, low_res, high_res))
            ret['block_low_%d' % current_channels] = low_res
            ret['block_high_%d' % current_channels] = high_res
            ret['join_%d' % current_channels] = noise_joined_layers[-1]
            for key, val in low_res_all_layers.iteritems():
                ret[key] = val
            for key, val in high_res_all_layers.iteritems():
                ret[key] = val
        final_chain = conv_block("output_chain", noise_joined_layers[-1], input_style_placeholder, current_channels,
                                 reuse=True)
        final_layer = conv_relu_layers("output", final_chain, input_style_placeholder, kernel_size=1, out_channels=3,
                                       reuse=True)

    return ret


def get_all_layers_conv_block(name, input_layer, input_style_placeholder, out_channels):
    """
    Each convolution block in Figure 2 contains three convo-
    lutional layers, each of which is followed by a ReLU acti-
    vation layer. The convolutional layers contain respectively
    3 x 3, 3 x 3 and 1 x 1 filters. Filers are computed densely
    (stride one) and applied using circular convolution to re-
    move boundary effects, which is appropriate for textures.
    :param name:
    :param input_layer:
    :param out_channels:
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    :return:
    """
    with tf.variable_scope(name, reuse=True):
        block1 = conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3,
                                  out_channels=out_channels, reuse=True)
        block2 = conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3, out_channels=out_channels,
                                  reuse=True)
        block3 = conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1, out_channels=out_channels,
                                  reuse=True)
        layer1_layers = get_all_layers_conv_relu_layers("layer1", input_layer, input_style_placeholder, kernel_size=3,
                                                        out_channels=out_channels, reuse=True)
        layer2_layers = get_all_layers_conv_relu_layers("layer2", block1, input_style_placeholder, kernel_size=3,
                                                        out_channels=out_channels, reuse=True)
        layer3_layers = get_all_layers_conv_relu_layers("layer3", block2, input_style_placeholder, kernel_size=1,
                                                        out_channels=out_channels, reuse=True)
    ret = {}
    for key, val in layer1_layers.iteritems():
        ret[name + 'layer1' + key] = val
    for key, val in layer2_layers.iteritems():
        ret[name + 'layer2' + key] = val
    for key, val in layer3_layers.iteritems():
        ret[name + 'layer3' + key] = val
    return ret


def get_all_layers_conv_relu_layers(name, input_layer, input_style_placeholder, kernel_size, out_channels,
                                    relu_leak=0.01, reuse=False):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Instance Norm
        - LeakyReLu
    :param reuse: If true, instead of creating new variables, we reuse the variables previously trained.
    """

    with tf.variable_scope(name, reuse=reuse):
        in_channels = input_layer.get_shape().as_list()[-1]
        weights = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, out_channels], tf.float32,
                                  tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', [out_channels], tf.float32,
                                 tf.constant_initializer(dtype=tf.float32))
        conv = neural_util.conv2d_mirror_padding(input_layer, weights, biases, kernel_size)
        norm = conditional_instance_norm(conv, input_style_placeholder, reuse=reuse)
        relu = tf.nn.elu(norm, 'elu')

        num_channels = conv.get_shape().as_list()[3]

        num_styles = input_style_placeholder.get_shape().as_list()[1]
        scale = tf.get_variable('scale', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())

        mean, variance = tf.nn.moments(conv, [0, 1, 2])
        variance = tf.abs(variance)
        variance_epsilon = 0.001
        inv = math_ops.rsqrt(variance + variance_epsilon)
        return {'conv': conv, 'norm': norm, 'relu': relu, 'scale': scale, 'offset': offset, 'mean': mean,
                'variance': variance, 'inv': inv}


def input_pyramid(name, height, width, batch_size, k=5, with_content_image=False, content_image_num_features=3):
    """
    Generates k inputs at different scales, with height x width being the largest.
    If with_content_image is true, add 3 to the last rgb dim because we are appending the resized content image to the
    noise generated.
    """
    if height % (2 ** (k - 1)) != 0 or width % (2 ** (k - 1)) != 0:
        print ('Warning: Input width or height cannot be divided by 2^(k-1). Width: %d. Height: %d This might cause problems when generating images.' %(width, height))
    with tf.get_default_graph().name_scope(name):
        return_val = [tf.placeholder(tf.float32, [batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)),
                                                  content_image_num_features+3 if with_content_image else 3], name=str(x)) for x in range(k)]
        return_val.reverse()
    return return_val


def noise_pyramid(height, width, batch_size, k=5, ablation_layer=None):
    """
    :param height: Height of the largest noise image.
    :param width: Width of the largest noise image.
    :param batch_size: Number of images we generate per batch.
    :param k: Number of inputs generated at different scales. If k = 3 for example, it will generate images with
    h x w, h/2 x w/2, h/4 x w/4.
    :param ablation_layer: If not none, every layer except for this one will be zeros. 0 <= ablation_layer < k-1.
    :return: A list of numpy arrays with size (batch_size, h/2^?, w/2^?, 3)
    """
    if height % (2 ** (k - 1)) != 0 or width % (2 ** (k - 1)) != 0:
        print('Warning: Input width or height cannot be divided by 2^(k-1). This might cause problems when generating '
               'images.')
    # return [np.random.uniform(size=(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3))
    #         if (ablation_layer is None or ablation_layer < 0 or (k-1-ablation_layer) != x) else
    #         np.zeros((batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3), dtype=np.float64) + 0.5
    #         for x in range(k)][::-1]

    return [np.random.uniform(size=(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3))
            if (ablation_layer is None or ablation_layer < 0 or (k - 1 - ablation_layer) == x) else
            np.random.uniform(size=(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3)) * 0.0
            for x in range(k)][::-1]

def noise_pyramid_w_content_img(height, width, batch_size, content_image_pyramid, k=5, ablation_layer=None):
    """
    :param height: Height of the largest noise image.
    :param width: Width of the largest noise image.
    :param batch_size: Number of images we generate per batch.
    :param k: Number of inputs generated at different scales. If k = 3 for example, it will generate images with
    h x w, h/2 x w/2, h/4 x w/4.
    :param ablation_layer: If not none, every layer except for this one will be zeros. 0 <= ablation_layer < k-1.
    :return: A list of numpy arrays with size (batch_size, h/2^?, w/2^?, 3)
    """
    """If an additional input tensor, the content image tensor, is
    provided, then we concatenate the downgraded versions of that tensor to the noise tensors."""
    return [np.concatenate((np.random.uniform(size=(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3))
                            if (ablation_layer is None or ablation_layer < 0 or (k - 1 - ablation_layer) == x) else
                            np.random.uniform(size=(batch_size, max(1, height // (2 ** x)), max(1, width // (2 ** x)), 3)) * 0.0,
                            content_image_pyramid[x]), axis=3) for x in range(k)][::-1]


def generate_image_pyramid(height, width, batch_size, content_image, k=5, num_features = 3):
    return [np.array([scipy.misc.imresize(content_image[batch],
                                          (max(1, height // (2 ** x)), max(1, width // (2 ** x)), num_features))
                      for batch in range(batch_size)]) for x in range(k)]

def generate_image_pyramid_from_content_list(height, width, content_image, k=5, num_features = 3):
    return [np.array([scipy.misc.imresize(content_image[i],
                                          (max(1, height // (2 ** x)), max(1, width // (2 ** x)), num_features))
                      for i in range(content_image.shape[0])]) for x in range(k)]


# TODO: add support for reuse.
def spatial_batch_norm(input_layer, input_style_placeholder, name='spatial_batch_norm', reuse=False):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    with tf.variable_scope(name, reuse=reuse):
        mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
        # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
        variance = tf.abs(variance)
        variance_epsilon = 0.001
        num_channels = input_layer.get_shape().as_list()[3]
        scale = tf.get_variable('scale', [num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_channels], tf.float32, tf.constant_initializer())
        return_val = tf.nn.batch_normalization(input_layer, mean, variance, offset, scale, variance_epsilon, name=name)
        return return_val


# TODO: add support for reuse.
def instance_norm(input_layer, name='instance_norm', reuse=False):
    """
    Instance-normalize the layer as in https://arxiv.org/abs/1607.08022
    """
    # calculate the mean and variance for width and height axises.
    with tf.variable_scope(name, reuse=reuse):
        input_layers = tf.unpack(input_layer)
        return_val = []
        num_channels = input_layer.get_shape().as_list()[3]
        # The scale and offset variable is reused for all batches in this norm.
        # NOTE: it is ok to use a different scale and offset for each batch. The meaning of doing so is not so clear but
        # it will still work. The resulting coloring of the image is different from the current implementation.
        scale = tf.get_variable('scale', [num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_channels], tf.float32, tf.constant_initializer())
        for l in input_layers:
            l = tf.expand_dims(l, 0)
            # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
            mean, variance = tf.nn.moments(l, [0, 1, 2])
            variance = tf.abs(variance)
            variance_epsilon = 0.001
            return_val.append(
                tf.squeeze(tf.nn.batch_normalization(l, mean, variance, offset, scale, variance_epsilon, name=name), [0]))
        return_val = tf.pack(return_val)
        return return_val


def conditional_instance_norm(input_layer, input_style_placeholder, name='conditional_instance_norm', reuse=False):
    """
    Instance-normalize the layer conditioned on the style as in https://arxiv.org/abs/1610.07629
    input_style_placeholder is a one hot vector (1 x N tensor) with length N where N is the number of different style
    images.
    """
    # calculate the mean and variance for width and height axises.
    with tf.variable_scope(name, reuse=reuse):
        input_layers = tf.unpack(input_layer)
        return_val = []
        num_styles = input_style_placeholder.get_shape().as_list()[1]
        num_channels = input_layer.get_shape().as_list()[3]
        scale = tf.get_variable('scale', [num_styles, num_channels], tf.float32, tf.random_uniform_initializer())
        offset = tf.get_variable('offset', [num_styles, num_channels], tf.float32, tf.constant_initializer())
        scale_for_current_style = tf.matmul(input_style_placeholder, scale)
        offset_for_current_style = tf.matmul(input_style_placeholder, offset)
        for l in input_layers:
            l = tf.expand_dims(l, 0)
            # NOTE: Tensorflow norm has some issues when the actual variance is near zero. I have to apply abs on it.
            mean, variance = tf.nn.moments(l, [0, 1, 2])
            variance = tf.abs(variance)
            variance_epsilon = 0.001
            return_val.append(tf.squeeze(tf.nn.batch_normalization(
                l, mean, variance, offset_for_current_style, scale_for_current_style, variance_epsilon, name=name),
                [0]))
        return_val = tf.pack(return_val)
        return return_val


def gramian(layer):
    # Takes (batches, height, width, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    _, height, width, number = map(lambda i: i.value, layer.get_shape())
    size = height * width * number
    layer_unpacked = tf.unpack(layer)
    grams = []
    for single_layer in layer_unpacked:
        feats = tf.reshape(single_layer, (-1, number))
        grams.append(tf.matmul(tf.transpose(feats), feats) / size) # TODO: find out the right normalization. This might be wrong.
    return tf.pack(grams)


def get_pyramid_scale_offset_var():
    scale_offset_variables = []
    for d in range(8, 48, 8):
        for layer in range(1, 4):
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                        scope='texture_nets/block_low_%d/layer%d/conditional_instance_norm/scale' % (
                                                        d, layer))
            scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                        scope='texture_nets/block_high_%d/layer%d/conditional_instance_norm/scale' % (
                                                        d, layer))

    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                scope='texture_nets/output_chain/layer1/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                scope='texture_nets/output_chain/layer2/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                scope='texture_nets/output_chain/layer3/conditional_instance_norm/scale')
    scale_offset_variables += tf.get_collection(tf.GraphKeys.VARIABLES,
                                                scope='texture_nets/output/conditional_instance_norm/scale')
    return scale_offset_variables


def total_variation(image_batch):
    """
    :param image_batch: A 4D tensor of shape [batch_size, height, width, channels]
    """
    batch_shape = image_batch.get_shape().as_list()
    batch_size = batch_shape[0]
    height = batch_shape[1]
    left = tf.slice(image_batch, [0, 0, 0, 0], [-1, height - 1, -1, -1])
    right = tf.slice(image_batch, [0, 1, 0, 0], [-1, -1, -1, -1])

    width = batch_shape[2]
    top = tf.slice(image_batch, [0, 0, 0, 0], [-1, -1, width - 1, -1])
    bottom = tf.slice(image_batch, [0, 0, 1, 0], [-1, -1, -1, -1])

    # left and right are 1 less wide than the original, top and bottom 1 less tall
    # In order to combine them, we take 1 off the height of left-right, and 1 off width of top-bottom
    vertical_diff = tf.slice(tf.sub(left, right), [0, 0, 0, 0], [-1, -1, width - 1, -1])
    horizontal_diff = tf.slice(tf.sub(top, bottom), [0, 0, 0, 0], [-1, height - 1, -1, -1])

    vertical_diff_shape = vertical_diff.get_shape().as_list()
    num_pixels_in_vertical_diff = vertical_diff_shape[0] * vertical_diff_shape[1] * vertical_diff_shape[2] * \
                                  vertical_diff_shape[3]
    horizontal_diff_shape = horizontal_diff.get_shape().as_list()
    num_pixels_in_horizontal_diff = horizontal_diff_shape[0] * horizontal_diff_shape[1] * horizontal_diff_shape[2] * \
                                    horizontal_diff_shape[3]

    # Why there's a 2 here? I added it according to https://github.com/antlerros/tensorflow-fast-neuralstyle and
    # https://github.com/anishathalye/neural-style
    total_variation = 2 * (tf.nn.l2_loss(horizontal_diff) / num_pixels_in_horizontal_diff + tf.nn.l2_loss(vertical_diff)/ num_pixels_in_vertical_diff) / batch_size


    return total_variation


def compute_image_features(image_path,layers,shape,vgg_data,mean_pixel, use_mrf, use_semantic_masks):
    features_dict = {}
    g = tf.Graph()
    # If using gpu, uncomment the following line.
    # with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    with g.as_default(), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.pre_read_net(vgg_data, image)
        style_pre = np.array([vgg.preprocess(image_path, mean_pixel)])
        for layer in layers:
            if use_mrf or use_semantic_masks:
                features = net[layer].eval(feed_dict={image: style_pre})
                features_dict[layer] = features
            else:
                # Calculate and store gramian.
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                features_dict[layer] = gram
    return features_dict