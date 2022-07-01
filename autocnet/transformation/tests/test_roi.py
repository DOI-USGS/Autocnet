import numpy as np
import pytest

import unittest
from unittest.mock import patch, Mock

from plio.io.io_gdal import GeoDataset

from autocnet.transformation.roi import Roi

@pytest.fixture
def array_with_nodata():
    arr = np.ones((10,10))
    arr[5,5] = 0
    return arr

def test_geodata_with_ndv_is_valid(geodata_a):
    roi = Roi(geodata_a, 7, 7, size_x=2, size_y=2)
    # Clip the ROI so that our clipped array is populated
    roi.clip()
    assert roi.is_valid == False

def test_geodata_is_valid(geodata_b):
    roi = Roi(geodata_b, 500, 500, size_x=200, size_y=200)
    roi.data.no_data_value = 2  # Monkey patch in None (default on the Mock)
    # Clip the ROI so that our clipped array is populated
    roi.clip()
    assert roi.is_valid == True

def test_center(geodata_c):
    geodata_c.read_array.return_value = np.ones((20, 20))
    roi = Roi(geodata_c, 5, 5, size_x=5, size_y=5, buffer=5)
    assert roi.center == (5.0, 5.0)

@pytest.mark.parametrize("x, y, axr, ayr",[
                         (10.1, 10.1, .1, .1),
                         (10.5, 10.5, .5, .5),
                         (10.9, 10.9, .9, .9)
    ])
def test_roi_remainder(x, y, axr, ayr, geodata_c):
    geodata_c.read_array.return_value = np.zeros((10,10))
    roi = Roi(geodata_c, x, y)
    pytest.approx(roi._remainder_x, axr)
    pytest.approx(roi._remainder_y, ayr)
    assert roi.x == x
    assert roi.y == y

@pytest.mark.parametrize("x, y, size_arr, size_roi, expected",[
    (50, 50, (100,100), (10,10), [40,60,40,60]),
    (15, 15, (100, 100), (15, 15), [0, 30, 0, 30]),
    (75, 75, (100,100), (25,25), [50, 100, 50, 100])
])
def test_extent_computation(x, y, size_arr, size_roi, expected, geodata_c):
    geodata_c.read_array.return_value = np.zeros(size_arr)
    roi = Roi(geodata_c, x, y, size_x=size_roi[0], size_y=size_roi[1])
    pixels = roi.image_extent
    assert pixels == expected

@pytest.mark.parametrize("x, y, size_arr, size_roi,buffer,expected",[
    (50, 50, (100,100), (10,10), 5, (4040, 6060)),
    (20, 20, (100, 100), (20, 20), 0, (0,3030)),
    (69, 69, (100,100), (30,30), 3, (4545, 9999))
])
def test_array_extent_computation(x, y, size_arr, size_roi, buffer, expected, geodata_c):
    geodata_c.read_array.return_value = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)

    roi = Roi(geodata_c, x, y, size_x=size_roi[0], size_y=size_roi[1], buffer=buffer)
    roi.clip()

    assert roi.clipped_array.dtype == np.float32
    assert (roi.clipped_array.shape == np.asarray(size_roi) * 2 + 1).all()

@pytest.mark.parametrize("x, y, x1, y1, xs, ys, size_arr, size_roi, expected",[
    (50, 50, 50, 50, -5, -5, (100, 100), (10, 10), (45, 45)),
    (50, 50, 10, 10, -5, -5, (100, 100), (20, 20), (5,  5 )),
    (50, 50, 10, 10,  5,  5, (100, 100), (20, 20), (15, 15 ))
])
def test_subpixel_using_roi(x, y, x1, y1, xs, ys, size_arr, size_roi, expected):
    source = Mock(GeoDataset)
    source_array = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)
    source.read_array.return_value = source_array

    destination = Mock(GeoDataset)
    destination_array = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)
    destination.read_array.return_value = destination_array

    s_roi = Roi(source, x, y, size_x=size_roi[0], size_y=size_roi[1])
    d_roi = Roi(destination, x1, y1, size_x=size_roi[0], size_y=size_roi[1])

    # Then subpixel matching happens on the two ROIs
    x_shift = xs
    y_shift = ys

    new_d_x = d_roi.x + x_shift
    new_d_y = d_roi.y + y_shift

    assert new_d_x == expected[0]
    assert new_d_y == expected[1]
