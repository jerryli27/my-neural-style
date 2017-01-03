import unittest

import general_util
from sketches_util import *

class TestImageToSketchesUtil(unittest.TestCase):
    def test_image_to_sketch(self):
        img = general_util.imread('12746957.jpg')
        sketch = image_to_sketch(img)
        cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        cv2.imshow('Sketch', sketch.astype(np.uint8))

        cv2.waitKey(0)

    def test_generate_hint_from_image(self):
        img = general_util.imread('face_36_566_115.png')
        sketch = generate_hint_from_image(img)
        cv2.imshow('Input', cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        cv2.imshow('Hint', cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_RGBA2BGR))
        cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
