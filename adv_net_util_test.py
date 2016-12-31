from adv_net_util import *


class AdvNetTest(tf.test.TestCase):
    def test_adv_net(self):
        with self.test_session():
            batch_size = 1
            height = 32
            width = 32
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            unet_output = net(input_layer)

            image_shape = input_layer.get_shape().as_list()
            final_shape = unet_output.get_shape().as_list()

            #
            # if not (image_shape[0] == final_shape[0] and image_shape[1] == final_shape[1] and image_shape[2] == final_shape[2]):
            #     print('image_shape and final_shape are different. image_shape = %s and final_shape = %s' %(str(image_shape), str(final_shape)))
            #     raise AssertionError

            self.assertAllEqual(image_shape, final_shape)

if __name__ == '__main__':
    tf.test.main()
