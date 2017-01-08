import numpy as np

from unet_util import *


class UnetTest(tf.test.TestCase):
    def test_unet(self):
        with self.test_session() as sess:
            batch_size = 1
            height = 53
            width = 67
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            unet_output = net(input_layer, mirror_padding=False)

            image_shape = input_layer.get_shape().as_list()
            final_shape = unet_output.get_shape().as_list()

            self.assertAllEqual(image_shape, final_shape)

            sess.run(tf.initialize_all_variables())

            feed_input = np.ones((batch_size, height, width, num_features))

            feed_dict = {input_layer:feed_input}
            actual_output = unet_output.eval(feed_dict)
            self.assertTrue(actual_output is not None, 'The unet failed to produce an output.')


if __name__ == '__main__':
    tf.test.main()
