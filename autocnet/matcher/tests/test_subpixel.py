import math
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock, PropertyMock
import logging

from skimage import transform as tf
from skimage.util import img_as_float
from skimage import color
from skimage import data

import pytest
import tempfile

import numpy as np
from imageio import imread

from plio.io.io_gdal import GeoDataset, array_to_raster

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

    roi_raster1 = tempfile.NamedTemporaryFile()
    roi_raster2 = tempfile.NamedTemporaryFile()

    array_to_raster(image, roi_raster1.name)
    array_to_raster(rts_image, roi_raster2.name)

    roi1 = roi.Roi(GeoDataset(roi_raster1.name), x=705, y=705, size_x=50, size_y=50)
    roi2 = roi.Roi(GeoDataset(roi_raster2.name), x=705, y=705, size_x=50, size_y=50)
    return roi1, roi2

def clip_side_effect(arr, clip=False):
    if not clip:
        return arr
    else:
        center_y = arr.shape[0] / 2
        center_x = arr.shape[1] / 2
        xr, x = math.modf(center_x)
        yr, y = math.modf(center_y)
        x = int(x)
        y = int(y)
        return arr[y-10:y+11, x-10:x+11]

@pytest.fixture
def apollo_subsets():
    roi1 = roi.Roi(GeoDataset(get_path('AS15-M-0295_SML(1).png')), x=173, y=150, size_x=50, size_y=50)
    roi2 = roi.Roi(GeoDataset(get_path('AS15-M-0295_SML(2).png')), x=145, y=285, size_x=50, size_y=50)
    roi1.clip()
    roi2.clip()
    return roi1, roi2

def test_subpixel_template(apollo_subsets):
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    b.size_x = 10
    b.size_y = 10
    affine, metrics, corr_map = sp.subpixel_template(a, b, upsampling=16)
    nx, ny = affine.translation
    assert nx == -0.3125
    assert ny == 1.5
    assert np.max(corr_map) >= 0.9367293
    assert metrics >= 0.9367293

@pytest.mark.parametrize("loc, failure", [((0,4), True),
                                          ((4,0), True),
                                          ((1,1), False)])
def test_subpixel_template_at_edge(apollo_subsets, loc, failure):
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    b.size_x = 10
    b.size_y = 10

    def func(*args, **kwargs):
        corr = np.zeros((10,10))
        corr[loc[0], loc[1]] = 10
        return 0, 0, 0, corr

    if failure:
        affine, metrics, corr_map = sp.subpixel_template(a, b, upsampling=16,
                                                         func=func)
    else:
        affine, metrics, corr_map = sp.subpixel_template(a, b, upsampling=16,
                                                         func=func)
        nx, ny = affine.translation
        assert nx == 0

@pytest.mark.xfail
def test_estimate_logpolar_transform(iris_pair):
    roi1, roi2 = iris_pair
    roi1.size_x = 705
    roi1.size_y = 705
    roi2.size_x = 705
    roi2.size_y = 705
    roi1.clip()
    roi2.clip()
    affine = sp.estimate_logpolar_transform(roi1.clipped_array, roi2.clipped_array)

    assert pytest.approx(affine.scale, 0.1) == 0.71
    assert pytest.approx(affine.rotation, 0.1) == 0.34
    assert pytest.approx(affine.translation[0], 0.1) == 283.68
    assert pytest.approx(affine.translation[1], 0.1) == -198.62

@pytest.mark.xfail
def test_fourier_mellen(iris_pair):
    roi1, roi2 = iris_pair
    roi1.size_x = 200
    roi1.size_y = 200
    roi2.size_x = 200
    roi2.size_y = 200
    roi1.clip()
    roi2.clip()
    affine, metrics, corrmap = sp.fourier_mellen(roi1, roi2, phase_kwargs = {"reduction" : 11, "convergence_threshold" : 1, "max_dist":100})

    assert pytest.approx(nx, 0.01) == 996.39
    assert pytest.approx(ny, 0.01) ==  984.912
    assert pytest.approx(error, 0.01) == 0.0422

@pytest.mark.xfail
@pytest.mark.parametrize("convergence_threshold, expected", [(2.0, (-0.32, 1.66, -9.5e-20))])
def test_iterative_phase(apollo_subsets, convergence_threshold, expected):
    a = apollo_subsets[0]
    b = apollo_subsets[1]
    affine, metrics, corr_map = sp.iterative_phase(a, b,
                                                   convergence_threshold=convergence_threshold,
                                                   upsample_factor=100)
    dx, dy = affine.translation
    assert dx == expected[0]
    assert dy == expected[1]
    if expected[2] is not None:
        # for i in range(len(strength)):
        assert pytest.approx(metrics,6) == expected[2]

@pytest.mark.parametrize("data, expected", [
    ((21,21), (10, 10)),
    ((20,20), (10,10))
])
def test_check_image_size(data, expected):
    assert sp.check_image_size(data) == expected

@pytest.mark.xfail
@pytest.mark.parametrize("x, y, x1, y1, image_size, template_size, expected",[
    (4, 3, 4, 2, (5,5), (3,3), (4,2)),
    (4, 3, 4, 2, (7,7), (3,3), (4,2)),  # Increase the search image size
    (4, 3, 4, 2, (7,7), (5,5), (4,2)), # Increase the template size
    (4, 3, 3, 2, (7,7), (3,3), (4,2)), # Move point in the x-axis
    (4, 3, 5, 3, (7,7), (3,3), (4,2)), # Move point in the other x-direction
    (4, 3, 4, 1, (7,7), (3,3), (4,2)), # Move point negative in the y-axis
    (4, 3, 4, 3, (7,7), (3,3), (4,2))  # Move point positive in the y-axis

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

    dx, dy, corr, corrmap = sp.subpixel_template(x, y, x1, y1, 
                                                 test_image, t_shape,
                                                 image_size=image_size, 
                                                 template_size=template_size, 
                                                 upsampling=1)
    assert corr >= 1.0  # geq because sometime returning weird float > 1 from OpenCV
    assert dx == expected[0]
    assert dy == expected[1]

@pytest.mark.xfail
@pytest.mark.parametrize("x, y, x1, y1, image_size, expected",[
    (4, 3, 3, 2, (3,3), (3,2)),
    (4, 3, 3, 2, (5,5), (3,2)),  # Increase the search image size
    (4, 3, 3, 2, (5,5), (3,2)), # Increase the template size
    (4, 3, 2, 2, (5,5), (3,2)), # Move point in the x-axis
    (4, 3, 4, 3, (5,5), (3,2)), # Move point in the other x-direction
    (4, 3, 3, 1, (5,5), (3,2)), # Move point negative in the y-axis; also tests size reduction
    (4, 3, 3, 3, (5,5), (3,2))  # Move point positive in the y-axis

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

    dx, dy, metrics, _ = sp.subpixel_phase(x, y, x1, y1, 
                                                 test_image, t_shape,
                                                 image_size=image_size)

    assert dx == expected[0]
    assert dy == expected[1]


def test_mutual_information(): 
    d_template = np.array([[i for i in range(50, 100)] for j in range(50)])
    s_image = np.ones((100, 100))
    s_image[25:75, 25:75] = d_template
    
    template = Mock(spec=roi.Roi, clipped_array = d_template)
    image = Mock(spec=roi.Roi, clipped_array = s_image)

    affine, max_corr, corr_map = sp.mutual_information_match(image, template, bins=20)
    assert affine.params[0][2] == -0.5171186125717124
    assert affine.params[1][2] == pytest.approx(-0.5)
    assert max_corr == 2.9755967600033015
    assert corr_map.shape == (51, 51)
    assert np.min(corr_map) >= 0.0