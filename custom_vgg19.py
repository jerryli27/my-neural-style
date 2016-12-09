import inspect
import os

import numpy as np
import tensorflow as tf

import vgg19

VGG_MEAN = [103.939, 116.779, 123.68]

def loadWeightsData(vgg19_npy_path=None):
    if vgg19_npy_path is None:
        path = inspect.getfile(Vgg19)
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "vgg19.npy")
        vgg19_npy_path = path
        print (vgg19_npy_path)
    return np.load(vgg19_npy_path, encoding='latin1').item()

class custom_Vgg19(vgg19.Vgg19):
    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]

    def __init__(self, rgb, data_dict, train=False):
        # It's a shared weights data and used in various
        # member functions.
        self.data_dict = data_dict

        # start_time = time.time()
        print ("build model started")
        # rgb_scaled = rgb * 255.0
        rgb_scaled = rgb
        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # We only need the conv layers.
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # self.data_dict = None
        # print ("build model finished: %ds" % (time.time() - start_time))

    def debug(self):
        pass



