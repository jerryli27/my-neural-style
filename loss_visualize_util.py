"""
This file implements functions to visualize some potential loss functions.
"""

import vgg
from feedforward_style_net_util import *

CONTENT_LAYER = 'relu4_2'  # Same setting as in the paper.
# STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
STYLE_LAYERS = (
'relu1_2', 'relu2_2', 'relu3_2', 'relu4_2')  # Set according to https://github.com/DmitryUlyanov/texture_nets
STYLE_LAYERS_MRF = ('relu3_1', 'relu4_1')  # According to https://arxiv.org/abs/1601.04589.


# TODO: change rtype
def style_synthesis_net(content, style, layers, per_pixel_loss_func, path_to_network = 'imagenet-vgg-verydeep-19.mat'):
    height = content.shape[0]
    width = content.shape[1]
    input_shape = (1,height, width, 3)

    # Read the vgg net
    vgg_data, mean_pixel = vgg.read_net(path_to_network)
    content_pre = np.array([vgg.preprocess(content, mean_pixel)])
    style_pre = np.array([vgg.preprocess(style, mean_pixel)])
    # vgg_data_dict = loadWeightsData('./vgg19.npy')
    print('Finished loading VGG.')

    content_features = {}
    style_features = {}
    losses = {}


    g = tf.Graph()
    with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=input_shape)
        # This mean pixel variable is unique to the input trained vgg network. It is independent of the input image.
        net = vgg.pre_read_net(vgg_data, image)
        for layer in layers:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})
            style_features[layer] = net[layer].eval(feed_dict={image: style_pre})
            losses[layer] =  per_pixel_loss_func(content_features[layer], style_features[layer])
        print('Finished loading content and style image features.')

    return losses

def per_pixel_gram_loss(content_feature, style_feature):
    gram = np_gramian(content_feature)
    style_gram = np_gramian(style_feature)
    return np.abs(gram - style_gram)

def np_gramian(layer):
    # Takes (batches, height, width, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    batch, height, width, number = layer.shape
    size = height * width * number
    grams = []
    for i in range(batch):
        single_layer = layer[i,:,:,:]
        feats = np.reshape(single_layer, (-1, number))
        grams.append(np.matmul(np.transpose(feats), feats) / size)
    return np.array(grams)

# Calculate the per pixel loss for a single feature.
def gramian_loss_per_pix(layer, feature_x, feature_y):
    # Takes (batches, height, width, channels) and computes gramians of dimension (batches, channels, channels)
    # activations_shape = activations.get_shape().as_list()
    # """
    # Instead of iterating over #channels width by height matrices and computing similarity, we vectorize and compute
    # the entire gramian in a single matrix multiplication.
    # """
    batch, height, width, number = layer.shape
    size = height * width * number
    grams = []
    for i in range(batch):
        single_layer = layer[i,:,:,:]
        feats = np.reshape(single_layer, (-1, number))
        feats_gram = []
        for j in range(number):
            feats_gram.append(np.matmul(np.transpose(feats[:,j]), feats[:,j])  / size)
        # grams.append(np.matmul(np.transpose(feats), feats))
        grams.append(np.array(feats_gram))
    return np.array(grams)