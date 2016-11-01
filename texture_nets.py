"""
This file implements the feed-forward texture networks as described in http://arxiv.org/abs/1603.03417.
(For more background, see http://arxiv.org/abs/1508.06576)
"""

import tensorflow as tf
import numpy as np
from sys import stderr
import scipy.misc

import neural_util
import vgg
from general_util import *

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')


def style_synthesis_net(path_to_network, content, styles, iterations, batch_size,
                        content_weight, style_weight, style_blend_weights, tv_weight,
                        learning_rate, print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """

    m = 256

    shape = (1,) + content.shape
    # Append a (1,) in front of the shapes of the style images. So the style_shapes contains (1, height, width , 3).
    # 3 corresponds to rgb.
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(path_to_network, image)
        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
            feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net, _ = vgg.net(path_to_network, image)
            style_pre = np.array([vgg.preprocess(styles[i], mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():

        noise_inputs = input_pyramid("noise", m, batch_size, with_content_image=True)
        image = generator_net(noise_inputs)
        net, _ = vgg.net(path_to_network, image)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) /
                                         content_features[CONTENT_LAYER].size)
        # style loss
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                gram = gramian(layer)
                style_gram = style_features[i][style_layer]
                style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)

            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # TODO: THIS PART WAS GIVING ME NAN WHEN I ADDED THE CONTENT IMAGE.
        # # total variation denoising
        # tv_y_size = tensor_size(image[:, 1:, :, :])
        # tv_x_size = tensor_size(image[:, :, 1:, :])
        # tv_loss = tv_weight * 2 * (
        #     (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
        #      tv_y_size) +
        #     (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
        #      tv_x_size))
        # # overall loss
        # loss = content_loss + style_loss + tv_loss

        # overall loss
        loss = content_loss + style_loss

        # optimizer setup
        # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
        # They used learning rate = 0.001
        train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        def print_progress(i, feed_dict, last=False):
            stderr.write('Iteration %d/%d\n' % (i + 1, iterations))
            if last or (print_iterations and i % print_iterations == 0):
                stderr.write('  content loss: %g\n' % content_loss.eval(feed_dict=feed_dict))
                stderr.write('    style loss: %g\n' % style_loss.eval(feed_dict=feed_dict))
                # stderr.write('       tv loss: %g\n' % tv_loss.eval(feed_dict=feed_dict))
                stderr.write('    total loss: %g\n' % loss.eval(feed_dict=feed_dict))

        # Optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            content_image_pyramid = content_img_pyramid(m, batch_size, content_pre)

            for i in range(iterations):
                last_step = (i == iterations - 1)
                feed_dict = {}
                noise = noise_pyramid_w_content_img(m, batch_size, content_image_pyramid)
                for index, noise_frame in enumerate(noise_inputs):
                    feed_dict[noise_frame] = noise[index]

                print_progress(i, feed_dict=feed_dict, last=last_step)
                train_step.run(feed_dict=feed_dict)

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval(feed_dict=feed_dict)
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval(feed_dict=feed_dict)
                    yield (
                        (None if last_step else i),
                        vgg.unprocess(best[0, :, :, :].reshape(shape[1:]), mean_pixel)
                    )  # Because we now have batch, choose the first one in the batch as our sample image.


def generator_net(input_noise_z):
    """
    This function takes a list of tensors as input, and outputs a tensor with a given width and height. The output
    should be similar in style to the target image.
    :return: the full sized generated image/texture.
    """

    # For now, we assume that the input_noise_z consists of K = 5 random tensors as described in the paper.
    # Different from Ulyanov et el, in the original paper z_k is the smallest input and z_1 is the largest.
    # Here we reverse the order
    assert len(input_noise_z) == 5
    with tf.get_default_graph().name_scope('texture_nets'):
        noise_joined = input_noise_z[0]
        current_channels = 8
        channel_step_size = 8
        for noise_layer in input_noise_z[1:]:
            low_res = conv_block('block_low_%d' % current_channels, noise_joined, current_channels)
            high_res = conv_block('block_high', noise_layer, channel_step_size)
            current_channels += channel_step_size
            noise_joined = join_block('join_%d' % current_channels, low_res, high_res)
        final_chain = conv_block("output_chain", noise_joined, current_channels)
        return conv_relu_layers("output", final_chain, kernel_size=1, out_channels=3)


def conv_block(name, input_layer, out_channels):
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
    :return:
    """
    with tf.get_default_graph().name_scope(name):
        block1 = conv_relu_layers("layer1", input_layer, kernel_size=3, out_channels=out_channels)
        block2 = conv_relu_layers("layer1", block1, kernel_size=3, out_channels=out_channels)
        block3 = conv_relu_layers("layer1", block2, kernel_size=1, out_channels=out_channels)
    return block3


def conv_relu_layers(name, input_layer, kernel_size, out_channels, relu_leak=0.01):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    Per Ulyanov et el, this is a convolution layer followed by a ReLU layer consisting of
        - Mirror pad (TODO)
        - Number of maps from a convolutional layer equal to out_channels (multiples of 8)
        - Instance Norm
        - LeakyReLu
    """

    with tf.get_default_graph().name_scope(name):
        in_channels = input_layer.get_shape().as_list()[-1]

        # per https://arxiv.org/abs/1610.07629
        # initialize weights and bias using isotropic gaussian.
        weights = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_channels, out_channels],
                                               mean=0.0, stddev=0.01, name='weights'))
        biases = tf.Variable(tf.random_normal([out_channels], mean=0.0, stddev=0.01, name='biases'))
        # Do mirror padding in the future.
        conv = neural_util.conv2d(input_layer, weights, biases)
        batch_norm = instance_norm(conv)
        relu = neural_util.leaky_relu(batch_norm, relu_leak)
        return relu


def join_block(name, lower_res_layer, higher_res_layer):
    """
    This code is mostly taken from github.com/ProofByConstruction/texture-networks/blob/master/texture_network.py
    A block that combines two resolutions by upsampling the lower, batchnorming both, and concatting.
    """
    with tf.get_default_graph().name_scope(name):
        upsampled = tf.image.resize_nearest_neighbor(lower_res_layer, higher_res_layer.get_shape().as_list()[1:3])
        # No need to normalize here. According to https://arxiv.org/abs/1610.07629  normalize only after convolution.
    return tf.concat(3, [upsampled, higher_res_layer])


def input_pyramid(name, M, batch_size, k=5, with_content_image=False):
    """
    Generates k inputs at different scales, with MxM being the largest.
    If with_content_image is true, add 3 to the last rgb dim because we are appending the content image to the noise
    generated.
    """
    with tf.get_default_graph().name_scope(name):
        return_val = [
            tf.placeholder(tf.float32, [batch_size, M // (2 ** x), M // (2 ** x), 6 if with_content_image else 3],
                           name=str(x)) for x in range(k)]
        return_val.reverse()
    return return_val


def noise_pyramid(M, batch_size, k=5):
    return [np.random.rand(batch_size, M // (2 ** x), M // (2 ** x), 3) for x in range(k)][::-1]


def noise_pyramid_w_content_img(M, batch_size, content_image_pyramid, k=5):
    """If an additional input tensor, the content image tensor, is
    provided, then we concatenate the downgraded versions of that tensor to the noise tensors."""
    return [np.concatenate((np.random.rand(batch_size, M // (2 ** x), M // (2 ** x), 3),
                            content_image_pyramid[x]), axis=3) for x in range(k)][::-1]


def content_img_pyramid(M, batch_size, content_image, k=5):
    return [np.array(
        [scipy.misc.imresize(content_image[0], (M // (2 ** x), M // (2 ** x), 3)) for batch in range(batch_size)]) for x
            in range(k)]


def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def spatial_batch_norm(input_layer, name='spatial_batch_norm'):
    """
    Batch-normalizes the layer as in http://arxiv.org/abs/1502.03167
    This is important since it allows the different scales to talk to each other when they get joined.
    """
    mean, variance = tf.nn.moments(input_layer, [0, 1, 2])
    variance_epsilon = 0.00001
    inv = tf.rsqrt(variance + variance_epsilon)
    num_channels = input_layer.get_shape().as_list()[3]
    scale = tf.Variable(tf.random_uniform([num_channels]), name='scale')
    offset = tf.Variable(tf.random_uniform([num_channels]), name='offset')
    # return_val = tf.sub(tf.mul(tf.mul(scale, inv), tf.sub(input_layer, mean)), offset, name=name)
    # return return_val
    return_val = tf.nn.batch_normalization(input_layer, mean, variance, offset, scale, variance_epsilon, name=name)
    return return_val


# it works when batch_size = 1.
def instance_norm(input_layer, name='spatial_batch_norm'):
    """
    Instance-normalize the layer as in https://arxiv.org/abs/1607.08022
    """
    # calculate the mean and variance for width and height axises.
    input_layers = tf.unpack(input_layer)
    return_val = []
    for l in input_layers:
        return_val.append(tf.squeeze(spatial_batch_norm(tf.expand_dims(l, 0)), [0]))
    return_val = tf.pack(return_val)
    return return_val


def gramian(layer):
    # Takes (batches, width, height, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    # vectorized_activations = tf.reshape(tf.transpose(activations, perm=[0, 3, 1, 2]),
    #                                     [activations_shape[0], activations_shape[3], -1])
    # transposed_vectorized_activations = tf.transpose(vectorized_activations, perm=[0, 2, 1])
    # mult = tf.batch_matmul(vectorized_activations, transposed_vectorized_activations)
    # return mult

    # The old way of calculating gramian only works for batch = 1
    _, height, width, number = map(lambda i: i.value, layer.get_shape())
    size = height * width * number
    layer_unpacked = tf.unpack(layer)
    grams = []
    for single_layer in layer_unpacked:
        feats = tf.reshape(single_layer, (-1, number))
        grams.append(tf.matmul(tf.transpose(feats), feats) / size)
    return tf.pack(grams)
