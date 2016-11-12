# This file contains utility functions to implement support for mrf.
# Please refer to https://arxiv.org/abs/1601.04589 for more details.

import tensorflow as tf

# TODO: I should make style_layer static to improve speed.
def mrf_loss(style_layer, generated_layer, patch_size = 3):
    generated_layer_patches = create_local_patches(generated_layer, patch_size)
    style_layer_patches = create_local_patches(style_layer, patch_size)
    generated_layer_nn_matched_patches = patch_matching(generated_layer_patches, style_layer_patches, patch_size)


    _, height, width, number = map(lambda i: i.value, generated_layer.get_shape())
    size = height * width * number
    # Normalize by the size of the image as well as the patch area.
    loss = tf.reduce_sum(tf.square(tf.sub(generated_layer_patches, generated_layer_nn_matched_patches))) / size / (patch_size ** 2)

    return loss


def create_local_patches(layer, patch_size,padding= 'VALID'):
    """

    :param layer: Feature layer tensor with dimension (1, height, width, feature)
    :param patch_size: The width and height of the patch. It is set to 3 in the paper https://arxiv.org/abs/1601.04589
    :return: Patches with dimension (cardinality, patch_size, patch_size, feature)
    """
    return tf.extract_image_patches(layer, ksizes=[1,patch_size,patch_size,1],
                                    strides=[1,1,1,1], rates=[1,1,1,1], padding=padding)



def patch_matching(generated_layer_patches, style_layer_patches, patch_size):
    """
    The patch match-
    ing (Equation 3) is implemented as an additional convolu-
    tional layer for fast computation. In this case patches sam-
    pled from the style image are treated as the filters.
    :param generated_layer_patches: Size (batch, height, width, patch_size * patch_size * feature)
    :param style_layer_patches:Size (1, height, width, patch_size * patch_size * feature)
    :param patch_size:Size (1, height, width, patch_size * patch_size * feature)
    :return: Best matching patch with size (batch, height, width, patch_size * patch_size * feature)
    """
    normalized_generated_layer_patches = tf.nn.l2_normalize(generated_layer_patches, dim = [3])
    normalized_style_layer_patches = tf.nn.l2_normalize(style_layer_patches, dim = [3])
    #
    # # For each batch
    # normalized_generated_layer_patches_per_batch = tf.unpack(normalized_generated_layer_patches, axis=0)
    # width = normalized_generated_layer_patches.get_shape().as_list()[2]
    # nn_list = []
    # debug_i = 0
    # for batch in normalized_generated_layer_patches_per_batch:
    #     heights = tf.unpack(batch, axis=0)  # Unpack the height axis
    #     height_layers_nn_list = []
    #     for height_layer in heights:
    #         widths = tf.unpack(height_layer, axis=0)
    #         width_layers_nn_list = []
    #         for width_layer in widths:
    #             normalized_cross_correlations = tf.reduce_sum(
    #                 tf.mul(width_layer, normalized_style_layer_patches), reduction_indices=[3])
    #             reshaped_normalized_cross_correlations = tf.reshape(normalized_cross_correlations, shape=[-1])
    #             index = tf.argmax(reshaped_normalized_cross_correlations, dimension=0)
    #             height_index = tf.to_int32(index / width)
    #             width_index = tf.to_int32(index % width)
    #             width_layers_nn_list.append(style_layer_patches[0, height_index, width_index,:])
    #             debug_i += 1
    #             print('%d' % debug_i)
    #         width_layers_nn =  tf.pack(width_layers_nn_list)
    #         height_layers_nn_list.append(width_layers_nn)
    #     height_layers_nn = tf.pack(height_layers_nn_list)
    #     nn_list.append(height_layers_nn)
    # ret = tf.pack(nn_list)
    # return ret


    # A better way to do this is to treat them as convolutions.
    # They have to be in dimension
    # (height * width, patch_size, patch_size, feature) <=> (batch, in_height, in_width, in_channels)
    # (patch_size, patch_size, feature, height * width) <= > (filter_height, filter_width, in_channels, out_channels)
    # Initially they are in [batch, out_rows, out_cols, patch_size * patch_size * depth]
    original_shape = normalized_style_layer_patches.get_shape().as_list()
    height = original_shape[1]
    width = original_shape[2]
    depth = original_shape[3] / patch_size / patch_size
    normalized_style_layer_patches = tf.squeeze(normalized_style_layer_patches)

    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches, [height, width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches, [height*width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.transpose(normalized_style_layer_patches, perm=[1,2,3,0])

    style_layer_patches_reshaped = tf.reshape(style_layer_patches, [height, width, patch_size, patch_size, depth])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches_reshaped, [height*width, patch_size, patch_size, depth])

    normalized_generated_layer_patches_per_batch = tf.unpack(normalized_generated_layer_patches, axis=0)
    ret = []
    for batch in normalized_generated_layer_patches_per_batch:
        original_shape = batch.get_shape().as_list()
        height = original_shape[0]
        width = original_shape[1]
        depth = original_shape[2] / patch_size / patch_size
        batch = tf.squeeze(batch)

        batch = tf.reshape(batch,
                                                    [height, width, patch_size, patch_size, depth])
        batch = tf.reshape(batch,
                                                    [height * width, patch_size, patch_size, depth])
        # batch = tf.transpose(batch, perm=[1,2,3,0])
        # TODO: according to images-analogies github, for cross-correlation, we should flip the kernels
        # That is normalized_style_layer_patches should be [:, ::-1, ::-1, :]
        # Now I didn't see that in any other source.
        convs = tf.nn.conv2d(batch, normalized_style_layer_patches, strides=[1,1,1,1], padding='VALID')
        argmax = tf.squeeze(tf.argmax(convs,dimension=3))
        best_match = tf.gather(style_layer_patches_reshaped, indices = argmax)  # [height * width, patch_size, patch_size, depth]
        best_match = tf.reshape(best_match,[height, width, patch_size, patch_size, depth])
        best_match = tf.reshape(best_match,[height, width, patch_size*patch_size*depth])
        ret.append(best_match)
    ret = tf.pack(ret)
    return ret


