# from feedforward_style_net_util import *
#
# import unittest
# import tempfile
# import os
# import tensorflow as tf
# import numpy as np
#
# class TestCoOccurrenceMethods(unittest.TestCase):
#
#     def test_input_pyramid(self):
#
#         g = tf.Graph()
#         with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
#             name = 'test_name'
#             height = 512
#             width = 256
#             batch_size = 4
#             k = 5
#             with_content_image = False
#             actual_value = input_pyramid(name, height, width, batch_size, k, with_content_image)
#             self.assertEqual(len(actual_value), k)
#             for i, tensor in enumerate(actual_value):
#                 expected_shape = (batch_size, max(1, height // (2 ** (k-i - 1))), max(1, width // (2 ** (k-i- 1))),
#                                                   6 if with_content_image else 3)
#                 self.assertEqual(tensor.get_shape(), expected_shape)
#                 self.assertEqual(tensor.dtype, tf.float32)
#
#
#
# # def test_input_pyramid(self):
# #     with tempfile.NamedTemporaryFile(delete=False) as f:
# #         g = tf.Graph()
# #         with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
# #             name = 'test_name'
# #             height = 512
# #             width = 256
# #             batch_size = 4
# #             k = 5
# #             actual_value = input_pyramid(name, height, width, batch_size, k, with_content_image=False)
# #
# #         os.unlink(f.name)  # manually delete.
# #         self.assertEqual(os.path.exists(f.name), False)
#
#
# if __name__ == '__main__':
#     unittest.main()