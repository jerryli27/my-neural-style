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

            self.assertEqual(image_shape[0], final_shape[0])
            self.assertEqual(2, final_shape[1])

            all_var = get_net_all_variables()
            expected_var_number = 3*7+2
            self.assertEqual(len(all_var), expected_var_number )

if __name__ == '__main__':
    tf.test.main()
