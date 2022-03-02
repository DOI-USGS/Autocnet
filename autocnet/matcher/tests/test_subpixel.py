import math
import os
from re import A
import sys
import unittest
from unittest.mock import patch

from skimage import transform as tf
from skimage.util import img_as_float   
from skimage import color 
from skimage import data

import pytest

import numpy as np
from imageio import imread

from autocnet.examples import get_path
import autocnet.matcher.subpixel as sp
from autocnet.transformation import roi

@pytest.fixture
def iris_pair(): 
    angle = 200
    scale = 1.4
    shiftr = 30
    shiftc = 10

    image = color.rgb2gray(data.retina())
    translated = image[shiftr:, shiftc:]
    rotated = tf.rotate(translated, angle)
    rescaled = tf.rescale(rotated, scale)
    sizer, sizec = image.shape
    rts_image = rescaled[:sizer, :sizec] 
    return image, rts_image

@pytest.fixture
def apollo_subsets():
    # These need to be geodata sets or just use mocks...
    arr1 = imread(get_path('AS15-M-0295_SML(1).png'))[100:201, 123:224]
    arr2 = imread(get_path('AS15-M-0295_SML(2).png'))[235:336, 95:196]
    return arr1, arr2

@pytest.mark.parametrize("nmatches, nstrengths", [(10,1), (10,2)])
def test_prep_subpixel(nmatches, nstrengths):
    arrs = sp._prep_subpixel(nmatches, nstrengths=nstrengths)
    assert len(arrs) == 5
    assert arrs[2].shape == (nmatches, nstrengths)
    assert arrs[0][0] == 0

@pytest.mark.parametrize("center_x, center_y, size, expected", [(4, 4, 9, 404),
                                                          (55.4, 63.1, 27, 6355)])
def test_clip_roi(center_x, center_y, size, expected):
    img = np.arange(10000).reshape(100, 100)

    clip, axr, ayr = sp.clip_roi(img, center_x, center_y, size)

    assert clip.mean() == expected


def test_subpixel_template(apollo_subsets):
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    
    ref_roi = roi.Roi(a, a.shape[0]/2, a.shape[1]/2, 10, 10)
    moving_roi = roi.Roi(b, math.floor(b.shape[0]/2), math.floor(b.shape[1]/2), 51, 51)

    affine, strength, corrmap = sp.subpixel_template(ref_roi, moving_roi, upsampling=16)
    print(corrmap)
    print(affine)

    assert strength >= 0.99
    assert affine.translation[0] == 80.0625
    assert affine.translation[1] == 82



def test_subpixel_transformed_template(apollo_subsets):
    a = apollo_subsets[0]
    b = apollo_subsets[1]

    moving_center = math.floor(b.shape[0]/2), math.floor(b.shape[1]/2)
    transform = tf.AffineTransform(rotation=math.radians(1), scale=(1.1,1.1))
    ref_roi = roi.Roi(a, a.shape[0]/2, a.shape[1]/2, 10, 10)
    moving_roi = roi.Roi(b, *moving_center, 51, 51)

    # with patch('autocnet.transformation.roi.Roi.clip', side_effect=clip_side_effect):
    affine, strength, corrmap = sp.subpixel_template(ref_roi, moving_roi, transform, upsampling=16)
    
    print(corrmap)
    print(affine)
    assert strength >= 0.83
    assert affine.translation[0] == pytest.approx(70.68980522)
    assert affine.translation[1] == pytest.approx(68.20849946)


def test_estimate_logpolar_transform(iris_pair):
    img1, img2 = iris_pair 
    affine = sp.estimate_logpolar_transform(img1, img2) 

    assert pytest.approx(affine.scale, 0.1) == 0.71
    assert pytest.approx(affine.rotation, 0.1) == 0.34 
    assert pytest.approx(affine.translation[0], 0.1) == 283.68
    assert pytest.approx(affine.translation[1], 0.1) == -198.62 


def test_fourier_mellen(iris_pair):
    img1, img2 = iris_pair 
    nx, ny, error = sp.fourier_mellen(img1, img2, phase_kwargs = {"reduction" : 11, "size":(401, 401), "convergence_threshold" : 1, "max_dist":100}) 
    
    assert pytest.approx(nx, 0.01) == 996.39 
    assert pytest.approx(ny, 0.01) ==  984.912 
    assert pytest.approx(error, 0.01) == 0.0422 


@pytest.mark.parametrize("convergence_threshold, expected", [(2.0, (50.49, 52.44, -9.5e-20))])
def test_iterative_phase(apollo_subsets, convergence_threshold, expected):
    reference_image = apollo_subsets[0]
    walking_image = apollo_subsets[1]
    image_size = (51, 51)

    ref_x, ref_y = reference_image.shape[0]/2, reference_image.shape[1]/2
    walk_x, walk_y = walking_image.shape[0]/2, walking_image.shape[1]/2

    reference_roi = roi.Roi(reference_image, ref_x, ref_y, size_x=image_size[0], size_y=image_size[1])
    walking_roi = roi.Roi(walking_image, walk_x, walk_y, size_x=image_size[0], size_y=image_size[1])
    affine, error, diffphase = sp.iterative_phase(reference_roi,
                                                  walking_roi,
                                                  convergence_threshold=convergence_threshold,
                                                  upsample_factor=100)
    ref_to_walk = affine.inverse((ref_x, ref_y))[0]
    assert ref_to_walk[0] == expected[0]
    assert ref_to_walk[1] == expected[1]
    if expected[2] is not None:
        assert pytest.approx(error,6) == expected[2]

@pytest.mark.parametrize("data, expected", [
    ((21,21), (10, 10)),
    ((20,20), (10,10))
])
def test_check_image_size(data, expected):
    assert sp.check_image_size(data) == expected

@pytest.mark.parametrize("x, y, x1, y1, image_size, template_size, expected",[
    (4, 3, 4, 2, (5,5), (3,3), (0,-1)),
    (4, 3, 4, 2, (7,7), (3,3), (0,-2)),  # Increase the search image size
    (4, 3, 4, 2, (7,7), (5,5), (0,-2)), # Increase the template size
    (4, 3, 3, 2, (7,7), (3,3), (-1,-2)), # Move point in the x-axis
    (4, 3, 5, 3, (7,7), (3,3), (1,-2)), # Move point in the other x-direction
    (4, 3, 4, 1, (7,7), (3,3), (0, -2)), # Move point negative in the y-axis
    (4, 3, 4, 3, (7,7), (3,3), (0,-2))  # Move point positive in the y-axis

])
def test_subpixel_template_cooked(x, y, x1, y1, image_size, template_size, expected):
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 1, 1, 1, 0, 1, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 1, 0, 1, 0, 0, 1, 0, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

    # Should yield (-3, 3) offset from image center
    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 1, 1, 1, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

    ref_roi = roi.Roi(test_image, x, y, *image_size)
    moving_roi = roi.Roi(t_shape, x1, y1, *template_size)
    new_affine, corr, corrmap = sp.subpixel_template(ref_roi, moving_roi, upsampling=1)
    assert corr >= 0.8  # geq because sometime returning weird float > 1 from OpenCV
    assert new_affine.translation[0] == expected[0]
    assert new_affine.translation[1] == expected[1]

@pytest.mark.parametrize("x, y, x1, y1, image_size, expected",[
    (4, 3, 3, 2, (1,1), (3,2)),
    (4, 3, 3, 2, (2,2), (3,2)),  # Increase the search image size
    (4, 3, 3, 2, (2,2), (3,2)), # Increase the template size
    (4, 3, 2, 2, (2,2), (3,2)), # Move point in the x-axis
    (4, 3, 4, 3, (2,2), (3,2)), # Move point in the other x-direction
    (4, 3, 3, 1, (2,2), (3,2)), # Move point negative in the y-axis; also tests size reduction
    (4, 3, 3, 3, (2,2), (3,2))  # Move point positive in the y-axis

])
def test_subpixel_phase_cooked(x, y, x1, y1, image_size, expected):
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 1, 1, 1, 0, 1, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 1, 0, 1, 0, 0, 1, 0, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

    # Should yield (-3, 3) offset from image center
    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 1, 1, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

    reference_roi = roi.Roi(test_image, x, y, size_x=image_size[0], size_y=image_size[1])
    walking_roi = roi.Roi(t_shape, x1, y1, size_x=image_size[0], size_y=image_size[1])

    affine, metrics, _ = sp.subpixel_phase(reference_roi, walking_roi)
    print(affine)
    dx, dy = affine.inverse((x1, y1))[0]
    print(affine)
    assert dx == expected[0]
    assert dy == expected[1]
