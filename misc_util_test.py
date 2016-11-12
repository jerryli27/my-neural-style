import os
import tempfile
import unittest

from misc_util import *


class TestDataUtilMethods(unittest.TestCase):
    def test_get_global_step_from_save_dir(self):
        save_dir = u'model/my256-nstyle-van_gogh_starry_sky-iter-batchsize-160000-4-lr-0.001000-use_mrf-False-style-5-content-5/model.ckpt-5000'
        expected_output = 5000
        actual_output = get_global_step_from_save_dir(save_dir)
        self.assertEqual(expected_output,actual_output)

if __name__ == '__main__':
    unittest.main()
