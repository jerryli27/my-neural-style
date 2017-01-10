"""
This file contains functions for experimental purposes. They probably won't have full documentations and you can safely
ignore them for most of the time... I hope.
"""
import tensorflow as tf

from neural_util import get_tensor_num_elements


def gram_experiment(features, horizontal_shift = 0, vertical_shift = 0):
    _, height, width, number = map(lambda i: i.value, features.get_shape())
    size = height * width * number

    features_unpacked = tf.unpack(features)

    grams = []
    for current_feature in features_unpacked:
        current_feature = tf.expand_dims(current_feature, 0)
        original = tf.slice(current_feature, [0, 0, 0, 0], [-1, height - vertical_shift, width - horizontal_shift, -1])
        shifted = tf.slice(current_feature, [0, vertical_shift, horizontal_shift, 0], [-1, -1, -1, -1])
        left_reshaped = tf.reshape(original, (-1, number))
        right_reshaped = tf.reshape(shifted, (-1, number))
        gram = tf.matmul(tf.transpose(left_reshaped), right_reshaped) / (size)
        grams.append(gram)
    grams = tf.pack(grams)
    return grams


def gram_stacks(features, shift_size=2, stride = 1):
    # This is the second attempt. It shifts the gram in a m x n range and calculate gram for each shift.
    batch_size, height, width, number = map(lambda i: i.value, features.get_shape())
    gram = []
    assert shift_size * stride < height and shift_size * stride < width

    for vertical_shift in range(shift_size):
        for horizontal_shift in range(shift_size):
            shifted_gram = gram_experiment(features, horizontal_shift * stride, vertical_shift * stride)
            gram.append(shifted_gram)
    gram_stack = tf.pack(gram)
    gram_stack = tf.transpose(gram_stack, (1,2,3,0)) # Shape = [batch_size, number, number, num_shifts]

    gram_stack_num_elements = get_tensor_num_elements(gram_stack)
    assert gram_stack_num_elements == (batch_size * number * number * shift_size * shift_size)
    return gram_stack