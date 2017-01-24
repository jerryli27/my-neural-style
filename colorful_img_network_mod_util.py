
"""
This file implements the network in "Colorful Image Colorization" paper (https://github.com/richzhang/colorization)
It is a generator network that is specifically created for coloring images. The usage of the network is slightly
different from the original paper however. It is now used to color sketches instead of bw images.
Some code comes directly from https://github.com/richzhang/colorization/blob/master/resources/caffe_traininglayers.py
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from conv_util import *

# Note: I modified the structure of the network in two ways:
#  1. The network now will return an image with the same size as the input through convolution instead of direct
# nearest neighbor image resizing. This is because coloring bw image and sketches are fundamentally different in that
# bw images contains more information than sketches and it is ok for it to be a little bit lazier in the last few
# layers.
#  2. The network no longer does deconvolution but instead uses nearest neighbor resizing followed by a convolution
# layer to achieve the same effect without checkerboard artifacts. Please read the excellent blog post at
# http://distill.pub/2016/deconv-checkerboard/ for more information.
NAMES = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3',
         'conv5_1','conv5_2','conv5_3','conv6_1','conv6_2','conv6_3','conv7_1','conv7_2','conv7_3',
         'conv8_1','conv8_2','conv8_3','conv9_1','conv9_2','conv10_1','conv10_2']
# the last few layers has to be large enough, or it won:t contain enough info to encode 216 channels
NUM_OUTPUTS = [64,64,128,128,256,256,256,512,512,512,512,512,512,512,512,512,512,512,512,256,256,256,128,128,64,
               64]
KERNEL_SIZES = [3] * 19 + [3,3,3,3,3,3,3]
STRIDES = [1,2,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1] + [1,1,1,1,1,1,1]
NORMS = ['','batch_norm','','batch_norm','','','batch_norm','','','batch_norm','','','batch_norm','','','batch_norm',
         '', '', 'batch_norm','', '', 'batch_norm','', 'batch_norm','', 'batch_norm']
DILATIONS = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1]
CONV_TRANSPOSE_LAYERS = {'conv8_1','conv9_1','conv10_1'}



# The following is the original setting in the author's git repo.
# NAMES = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3',
#          'conv5_1','conv5_2','conv5_3','conv6_1','conv6_2','conv6_3','conv7_1','conv7_2','conv7_3',
#          'conv8_1','conv8_2','conv8_3']
# NUM_OUTPUTS = [64,64,128,128,256,256,256,512,512,512,512,512,512,512,512,512,512,512,512,256,256,256]
# KERNEL_SIZES = [3] * 19 + [4,3,3]
# STRIDES = [1,2,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1] + [2,1,1]
# NORMS = ['','batch_norm','','batch_norm','','','batch_norm','','','batch_norm','','','batch_norm','','','batch_norm',
#          '', '', 'batch_norm','', '', 'batch_norm']
# DILATIONS = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1]
# CONV_TRANSPOSE_LAYERS = {'conv8_1'}

assert len(NAMES) == len(NUM_OUTPUTS) and len(NAMES) == len(KERNEL_SIZES) and len(NAMES) == len(STRIDES) and \
       len(NAMES) == len(NORMS) and len(NAMES) == len(DILATIONS)


def net(image, mirror_padding = False, num_bin = 6 , reuse = False):
    # type: (tf.Tensor, bool, bool) -> tf.Tensor
    """
    The network is a generator network that takes a sketch and tries to apply color to it.
    Note: there is a slight modification in the network with respect to the one in the paper. For the one in the
    paper, they directly upsampled an iamge that is 4 times small. This is fine for them since they have the black
    and white image with the original resolution that can provide the detailed luminance information. In our case we
    do not have such information and therefore it should be better to do deconvolution until we get an output image
    with the same resolution as the input.
    If you'd like to use this for the reimplementation of the original paper, just delete all layers after "conv8_3"
    and do a nearest neighbor upsampling. You'd also need to modify the rgb bin into the lab bin.
    :param image: tensor with shape (batch_size, height, width, num_features)
    :param mirror_padding: If true it uses mirror padding. Otherwise it uses zero padding. Note that there's a bug
    here if I use mirror padding in the conv-transpose layers, I will get errors during gradient computation.
    :param num_bin: The number of bins to divide each r,g,b channel into. If numbin = 6 then we will have 6 **3 = 216
    different color encodings, with each color's possible value as: 0, 51, 102, ..., 255
    :param reuse: If true, it tries to reuse the variable previously defined by the same network.
    :return: tensor with shape (batch_size, height, width, num_bin ** 3)
    """

    # NOTE: There might be a small change in the dimension of the input vs. output if the size cannot be divided evenly
    # by 4.
    with tf.variable_scope('johnson', reuse=reuse):

        prev_layer = image
        prev_layer_list = [image]

        with tf.variable_scope('unet', reuse=reuse):
            for i in range(len(NAMES)):
                if NAMES[i] not in CONV_TRANSPOSE_LAYERS:
                    current_layer = conv_layer(prev_layer, num_filters=NUM_OUTPUTS[i],
                                               filter_size=KERNEL_SIZES[i], strides=STRIDES[i],
                                               mirror_padding=mirror_padding, norm=NORMS[i], dilation=DILATIONS[i],
                                               name=NAMES[i], reuse=reuse)
                else:
                    # current_layer =  conv_tranpose_layer(prev_layer, num_filters=NUM_OUTPUTS[i],
                    #                            filter_size=KERNEL_SIZES[i], strides=STRIDES[i],
                    #                            mirror_padding=mirror_padding, norm=NORMS[i], name=NAMES[i],
                    #                            reuse=reuse)
                    prev_layer_shape = prev_layer.get_shape().as_list()
                    current_layer = tf.image.resize_nearest_neighbor(prev_layer, (prev_layer_shape[1] * 2,
                                                                                  prev_layer_shape[2] * 2))
                    current_layer =  conv_layer(current_layer, num_filters=NUM_OUTPUTS[i],
                                               filter_size=KERNEL_SIZES[i], strides=STRIDES[i],
                                               mirror_padding=mirror_padding, norm=NORMS[i], name=NAMES[i],
                                               reuse=reuse)
                prev_layer = current_layer
                prev_layer_list.append(current_layer)


        # # conv10_rgb_bin = conv_layer(prev_layer, num_bin ** 3, 1, 1, mirror_padding = mirror_padding,
        # #                            name ='conv10_rgb_bin', reuse = reuse)
        # conv8_rgb_bin = conv_layer(prev_layer, num_bin ** 3, 1, 1, with_bias = True, mirror_padding = mirror_padding,
        #                            name ='conv8_rgb_bin', reuse = reuse)
        # image_shape = image.get_shape().as_list()
        # conv8_rgb_bin_shape = conv8_rgb_bin.get_shape().as_list()
        # if not (image_shape[1] == conv8_rgb_bin_shape[1] and image_shape[2] == conv8_rgb_bin_shape[2]):
        #     conv8_rgb_bin = tf.image.resize_nearest_neighbor(conv8_rgb_bin, [image_shape[1], image_shape[2]])
        # final = conv8_rgb_bin

        final = conv_layer(prev_layer, 3, 1, 1, mirror_padding=mirror_padding, norm='', name ='conv10_rgb', \
                                                                                             reuse = reuse)
        image_shape = image.get_shape().as_list()
        final_shape = final.get_shape().as_list()
        if not (image_shape[1] == final_shape[1] and image_shape[2] == final_shape[2]):
            if not (abs(image_shape[1] - final_shape[1]) <= 3 and abs(
                        image_shape[2] - final_shape[2]) <= 3):
                raise AssertionError('The layers to be concatenated differ too much in shape. Something is '
                                     'wrong. Their shapes are: %s and %s'
                                     % (str(image_shape), str(final_shape)))
            final = tf.image.resize_nearest_neighbor(final, [image_shape[1], image_shape[2]])

        # Do sanity check.
        final_shape = final.get_shape().as_list()
        if not (image_shape[0] == final_shape[0] and image_shape[1] == final_shape[1] and image_shape[2] == final_shape[2]):
            print('image_shape and final_shape are different. image_shape = %s and final_shape = %s' % (
            str(image_shape), str(final_shape)))
            raise AssertionError
        # final_shape = final.get_shape().as_list()
        # if not (image_shape[0] == final_shape[0] and image_shape[1] == final_shape[1] * 4 and image_shape[2] ==
        #     final_shape[2] * 4):
        #     print('image_shape and final_shape are different. image_shape = %s and final_shape = %s' % (
        #     str(image_shape), str(final_shape)))
        #     raise AssertionError
        return final

def get_net_all_variables():
    if '0.12.0' in tf.__version__:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='johnson')
    else:
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope='johnson')



class ImgToRgbBinEncoder():
    def __init__(self,bin_num=6):
        self.bin_num=bin_num
        index_matrix=  []
        step_size = 255.0 / (bin_num - 1)
        r, g, b = 0, 0, 0
        while r <= 255:
            r_rounded = round(r)
            g = 0.0
            while g <= 255:
                g_rounded = round(g)
                b = 0.0
                while b <= 255:
                    b_rounded = round(b)
                    index_matrix.append([r_rounded, g_rounded, b_rounded])
                    b += step_size
                g += step_size
            r += step_size
        self.index_matrix = np.array(index_matrix, dtype=np.uint8)
        self.nnencode = NNEncode(5,5,cc=self.index_matrix)
    def img_to_rgb_bin(self,img, return_sparse = False):

        """

        :param img:  An image represented in numpy array with shape (batch, height, width, 3)
        :param bin_num: number of bins for each color dimension
        :return:  An image represented in numpy array with shape (batch, height, width, bin_num ** 3)
        """
        if len(img.shape) != 4:
            raise AssertionError("The image must have shape (batch, height, width, 3), not %s" %str(img.shape))
        batch, height, width, num_channel = img.shape
        if num_channel != 3:
            raise AssertionError("The image must have 3 channels representing rgb. not %d." %num_channel)

        # nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(index_matrix)
        #
        # img_resized = np.reshape(img, (batch * height * width, num_channel))
        #
        # # Shape batch * height * width, 5
        # distances, indices = nbrs.kneighbors(img_resized)
        #
        #
        #
        # # In the original paper they used a gaussian kernel with delta = 5.
        # distances = gaussian_kernel(distances, std=5.0)
        #
        # rgb_bin = np.zeros((batch * height * width, bin_num ** 3))
        #
        # for bhw in range(batch * height * width):
        #     for i in range(5):
        #         rgb_bin[bhw,indices[bhw,i]] = distances[bhw, i]
        #
        #
        # return rgb_bin

        return self.nnencode.encode_points_mtx_nd(img,axis=3, return_sparse=return_sparse)


    def rgb_bin_to_img(self,rgb_bin,t = 0.38):
        """
        This function uses annealed-mean technique in the paper.
        :param rgb_bin:
        :param t:
        """


        if len(rgb_bin.shape) != 4:
            raise AssertionError("The rgb_bin must have shape (batch, height, width, 3), not %s" % str(rgb_bin.shape))
        batch, height, width, num_channel = rgb_bin.shape
        if num_channel != self.bin_num**3:
            raise AssertionError("The rgb_bin must have bin_num**3 channels, not %d." % num_channel)
        rgb_bin_normalized = rgb_bin/np.sum(rgb_bin,axis=3,keepdims=True)

        exp_log_z_div_t = np.exp(np.divide(np.log(rgb_bin_normalized),t))
        annealed_mean = exp_log_z_div_t / np.sum(exp_log_z_div_t, axis=3, keepdims=True)
        return self.nnencode.decode_points_mtx_nd(annealed_mean, axis=3)

    def gaussian_kernel(self,arr,std):
        return np.exp(np.square(arr) / (-2 * std**2)) / (std * np.sqrt(2 * np.pi))



class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
        self.closest_neighbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1,return_sparse=False,sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        P = pts_flt.shape[0]

        if return_sparse:
            (dists, inds) = self.nbrs.closest_neighbor(pts_flt)
        else:
            (dists,inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        self.pts_enc_flt[self.p_inds,inds] = wts
        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    # def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
    #     pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
    #     pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
    #     if(returnEncode):
    #         return (pts_dec_nd,pts_1hot_nd)
    #     else:
    #         return pts_dec_nd

# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out