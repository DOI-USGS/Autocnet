import pytest

import unittest
from .. import naive_template
import numpy as np
import cv2

class TestNaiveTemplateAutoReg(unittest.TestCase):

    def setUp(self):
        self._test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 1, 0, 0, 0, 1, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 1, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                     (0, 1, 1, 1, 0, 0, 0, 0, 0),
                                     (0, 1, 0, 1, 0, 0, 0, 0, 0),
                                     (0, 1, 1, 1, 0, 0, 0, 0, 0),
                                     (0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

        self._square = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 1, 1, 1, 0, 0, 0),
                                 (0, 0, 0, 1, 0, 1, 0, 0, 0),
                                 (0, 0, 0, 1, 1, 1, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                 (0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

        self._vertical_line = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0),
                                        (0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)
        
    def test_subpixel_square(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match_chi2(self._test_image, self._square, usfac='auto')
        np.testing.assert_almost_equal(result_x, 2.001953125, decimal=5)
        np.testing.assert_almost_equal(result_y, 9.955078125, decimal=5)
    
    def test_subpixel_line(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match_chi2(self._test_image,self._vertical_line, usfac='auto')
        # Test offsets
        self.assertEqual(result_x, 5.03515625)
        self.assertEqual(result_y, 4.98828125)

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
                                        (0, 1, 0),
                                        (0, 0, 0)), dtype=np.uint8)

    def test_t_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._test_image,
                                                                              self._t_shape, 
                                                                              upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 1)
        self.assertEqual(result_y, 3)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_rect_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._test_image,
                                                                              self._rect_shape,
                                                                              upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 7)
        self.assertEqual(result_y, 10)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_square_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._test_image,
                                                                              self._square_shape,
                                                                              upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 2)
        self.assertEqual(result_y, 10)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def test_line_shape(self):
        result_x, result_y, result_strength, _ = naive_template.pattern_match(self._test_image, 
                                                                              self._vertical_line,
                                                                              upsampling=1)
        # Test offsets
        self.assertEqual(result_x, 7)
        self.assertEqual(result_y, 2)
        # Test Correlation Strength: At least 0.8
        self.assertGreaterEqual(result_strength, 0.8, "Returned Correlation Strength of %d" % result_strength)

    def tearDown(self):
        pass
