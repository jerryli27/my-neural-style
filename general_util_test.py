import os
import tempfile
import unittest
import shutil

from general_util import *


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
        dirs = ['/home/jerryli27/PycharmProjects/johnson-fast-neural-style/fast-style-transfer/train2014/COCO_train2014_000000000009.jpg']
        # dirs = ['source_compressed/512/1.jpg']


        height = 256
        width = 256
        batch_size = 8
        content_folder = '/home/jerryli27/shirobako01pic/'# '/home/jerryli27/PycharmProjects/johnson-fast-neural-style/fast-style-transfer/train2014/'

        # Get path to all content images.
        content_dirs = get_all_image_paths_in_dir(content_folder)
        # Ignore the ones at the end to avoid
        content_dirs = content_dirs[:-(len(content_dirs) % batch_size)]

        for i in range(10, 100):
            if i % 10 == 0:
                print(i)
            content_pre_list = read_and_resize_batch_images(
                get_batch(content_dirs,i * batch_size,batch_size),
                height, width)
            self.assertEqual(content_pre_list.shape,(batch_size,height,width,3))

if __name__ == '__main__':
    unittest.main()
