import shutil
import tempfile
import unittest

from general_util import *


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class TestDataUtilMethods(unittest.TestCase):
    def test_get_global_step_from_save_dir(self):
        save_dir = u'model/my256-nstyle-van_gogh_starry_sky-iter-batchsize-160000-4-lr-0.001000-use_mrf-False-style-5-content-5/model.ckpt-5000'
        expected_output = 5000
        actual_output = get_global_step_from_save_dir(save_dir)
        self.assertEqual(expected_output,actual_output)

    def test_get_all_image_paths_in_dir(self):
        dirpath = tempfile.mkdtemp()
        image_path = dirpath + '/image.jpg'
        f = open(image_path, 'w')
        f.close()
        actual_answer = get_all_image_paths_in_dir(dirpath + '/')
        expected_answer = [image_path]
        shutil.rmtree(dirpath)
        self.assertEqual(expected_answer,actual_answer)

    def test_read_and_resize_batch_images(self):
        height = 256
        width = 256
        batch_size = 8

        random_images = []

        content_folder = tempfile.mkdtemp()
        for i in range(batch_size):
            image_path = content_folder + ('/image_%d.jpg' %i)
            current_image = np.random.rand(height, width, 3)
            current_image = current_image / np.linalg.norm(current_image)
            random_images.append(np.ndarray.astype(current_image, np.int32))
            scipy.misc.imsave(image_path, random_images[-1])

        # Get path to all content images.
        content_dirs = get_all_image_paths_in_dir(content_folder + '/')

        content_pre_list = read_and_resize_batch_images(
            get_batch(content_dirs,0,batch_size),
            height, width)

        expected_answer = np.array(random_images)
        np.testing.assert_almost_equal(expected_answer, content_pre_list)

        shutil.rmtree(content_folder)

    # def test_read_and_resize_bw_mask_images(self):
    #
    #     height = 256
    #     width = 256
    #     batch_size = 8
    #     semantic_masks_num_layers = 3
    #     num_images_per_batch = batch_size * semantic_masks_num_layers
    #
    #     random_images = []
    #
    #     content_folder = tempfile.mkdtemp()
    #     for i in range(batch_size):
    #         random_images.append([])
    #         for j in range(semantic_masks_num_layers):
    #             image_path = content_folder + ('/image_%d_%d.jpg' % (i,j))
    #             random_images[-1].append(np.random.rand(height, width, 3))
    #             scipy.misc.imsave(image_path, random_images[-1][-1])
    #
    #     # Get path to all content images.
    #     content_dirs = get_all_image_paths_in_dir(content_folder + '/')
    #
    #     content_pre_list = read_and_resize_bw_mask_images(
    #             get_batch(content_dirs, 0, num_images_per_batch),
    #             height, width, batch_size, semantic_masks_num_layers)
    #
    #     temp = np.array(random_images)
    #     expected_answer = np.ndarray.astype(np.transpose(rgb2gray(temp), (0, 2, 3, 1)) * 255.0, np.int32)
    #     np.testing.assert_almost_equal(expected_answer, content_pre_list)
    #
    #     shutil.rmtree(content_folder)

if __name__ == '__main__':
    unittest.main()
