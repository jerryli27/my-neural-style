import unittest

import general_util
from image_to_sketches_util import *

class TestImageToSketchesUtil(unittest.TestCase):
    def test_image_to_sketch(self):
        img = general_util.imread('/home/xor/animeface-character-dataset/face_36_566_115.png')
        sketch = image_to_sketch(img)
        cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        cv2.imshow('Sketch', sketch.astype(np.uint8))

        cv2.waitKey(0)

if __name__ == '__main__':
    unittest.main()
