import pytest

import unittest
import itertools
from .. import naive_template
from autocnet.transformation import roi 
import numpy as np
import cv2

class TestNaiveTemplateAutoReg(unittest.TestCase):

    def setUp(self):
        self._test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 1, 1, 1, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 1, 1, 1, 0, 0, 0),
                                     (0, 0, 0, 1, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 1, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

        self._shape = np.array(((1, 1, 1),
                                (1, 0, 1),
                                (1, 1, 1)), dtype=np.uint8)


    def test_subpixel_shift(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match_autoreg(self._shape,
                                                                                      self._test_image,
                                                                                      cv2.TM_CCORR_NORMED)
        print(result_x, result_y)
        np.testing.assert_almost_equal(result_x, 0.167124, decimal=5)
        np.testing.assert_almost_equal(result_y, -1.170976, decimal=5)

class TestNaiveTemplate(unittest.TestCase):

    def setUp(self):
        # Center is (5, 6)
        self._test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 1, 0),
                                     (1, 1, 1, 0, 0, 0, 0, 1, 0),
                                     (0, 1, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 1, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 1, 1, 1),
                                     (0, 1, 1, 1, 0, 0, 1, 0, 1),
                                     (0, 1, 0, 1, 0, 0, 1, 0, 1),
                                     (0, 1, 1, 1, 0, 0, 1, 0, 1),
                                     (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

        # Should yield (-3, 3) offset from image center
        self._t_shape = np.array(((1, 1, 1),
                               (0, 1, 0),
                               (0, 1, 0)), dtype=np.uint8)

        # Should be (3, -4)
        self._rect_shape = np.array(((1, 1, 1),
                                  (1, 0, 1),
                                  (1, 0, 1),
                                  (1, 0, 1),
                                  (1, 1, 1)), dtype=np.uint8)

        # Should be (-2, -4)
        self._square_shape = np.array(((1, 1, 1),
                                    (1, 0, 1),
                                    (1, 1, 1)), dtype=np.uint8)

        # Should be (3, 5)
        self._vertical_line = np.array(((0, 1, 0),
                                     (0, 1, 0),
                                     (0, 1, 0)), dtype=np.uint8)

    def test_t_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._t_shape,
                                                                              self._test_image, 
                                                                              upsampling=4)
        # Test offsets
        self.assertEqual(result_x, -3)
        self.assertEqual(result_y, -3)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_rect_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._rect_shape,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 3)
        self.assertEqual(result_y, 4)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.99, "Returned Correlation Strength of %d" % result_strength)

    def test_square_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._square_shape,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, -2)
        self.assertEqual(result_y, 4)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_line_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._vertical_line,
                                                                           self._test_image, upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 3)
        self.assertEqual(result_y, -5)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_imposed_shift(self):
        # specifically testing whole numbers, partial numbers (either in image location or
        # tested shifts) would cause the ROI class to pixel lock (reintorpolate) the ROI
        # and this is not the place to test that logic.
        x, y = (5, 6)

        delta_xs = [1, 0, -1] 
        delta_ys = [-2, -1, 0, 1, 2]
        for delta_x, delta_y in itertools.product(delta_xs, delta_ys):
            xi = x + delta_x
            yi = y + delta_y

            ref_roi = roi.Roi(self._test_image, x, y, 4, 5, buffer=1)
            moving_roi = roi.Roi(self._test_image, xi, yi, 3, 3, buffer=0)

            shift_x, shift_y, metrics, corrmap = naive_template.pattern_match(moving_roi.clip(), ref_roi.clip(), upsampling=1)
            
            self.assertEqual(shift_x, float(delta_x))
            self.assertEqual(shift_y, float(delta_y))
    
    def tearDown(self):
        pass

