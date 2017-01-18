from colorful_img_network_util import *


class ColorfulImgNetTest(tf.test.TestCase):
    def test_get_net_all_variables(self):
        with self.test_session():
            batch_size = 1
            height = 32
            width = 32
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            network = net(input_layer)
            allvar = get_net_all_variables()
            print(len(allvar))
            # self.assertEqual(len(allvar), 32)
    def test_ImgToRgbBinEncoder(self):
        enc = ImgToRgbBinEncoder(bin_num=2)
        img = np.array([[[[0, 0, 0], [0, 0, 254]]]])
        rgb_bin = enc.img_to_rgb_bin(img)
        img_reconstructed = enc.rgb_bin_to_img(rgb_bin)
        img_reconstructed_expected = np.array([[[[0, 0, 0], [0, 0, 255]]]])
        np.testing.assert_array_equal(img_reconstructed_expected,img_reconstructed)

if __name__ == '__main__':
    tf.test.main()
