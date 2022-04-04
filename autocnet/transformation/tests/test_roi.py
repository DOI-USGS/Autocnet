import numpy as np
import pytest

from autocnet.transformation.roi import Roi

@pytest.fixture
def array_with_nodata():
    arr = np.ones((10,10))
    arr[5,5] = 0
    return arr

def test_geodata_with_ndv_is_valid(geodata_a):
    roi = Roi(geodata_a, 50, 50, size_x=2, size_y=2)
    assert roi.is_valid == False

def test_geodata_is_valid(geodata_b):
    roi = Roi(geodata_b, 500, 500, size_x=200, size_y=200)
    roi.data.no_data_value = None  # Monkey patch in None (default on the Mock)
    assert roi.is_valid == True

def test_center(array_with_nodata):
    roi = Roi(array_with_nodata, 5, 5, size_x=5, size_y=5)
    assert roi.center == (5,5)

@pytest.mark.parametrize("x, y, axr, ayr",[
                         (10.1, 10.1, .1, .1),
                         (10.5, 10.5, .5, .5),
                         (10.9, 10.9, .9, .9)
    ])
def test_roi_remainder(x, y, axr, ayr):
    gd = np.zeros((10,10))
    roi = Roi(gd, x, y)
    pytest.approx(roi.axr, axr)
    pytest.approx(roi.ayr, ayr)
    assert roi.x == x
    assert roi.y == y

@pytest.mark.parametrize("x, y, size_arr, size_roi, expected",[
    (50, 50, (100,100), (10,10), [40,60,40,60]),
    (15, 15, (100, 100), (15, 15), [0, 30, 0, 30]),
    (75, 75, (100,100), (25,25), [50, 100, 50, 100])
])
def test_extent_computation(x, y, size_arr, size_roi, expected):
    gd = np.zeros(size_arr)
    roi = Roi(gd, x, y, size_x=size_roi[0], size_y=size_roi[1])
    pixels = roi.image_extent
    assert pixels == expected

@pytest.mark.parametrize("x, y, size_arr, size_roi, expected",[
    (50, 50, (100,100), (10,10), (4040, 6060)),
    (20, 20, (100, 100), (20, 20), (0,3030)),
    (69, 69, (100,100), (30,30), (4545, 9999))
])
def test_array_extent_computation(x, y, size_arr, size_roi, expected):
    gd = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)
    
    roi = Roi(gd, x, y, size_x=size_roi[0], size_y=size_roi[1])
    array = roi.clip()

    assert array.dtype == np.float32
    assert (array.shape == np.asarray(size_roi) * 2 + 1).all()

@pytest.mark.parametrize("x, y, x1, y1, xs, ys, size_arr, size_roi, expected",[
    (50, 50, 50, 50, -5, -5, (100, 100), (10, 10), (45, 45)),
    (50, 50, 10, 10, -5, -5, (100, 100), (20, 20), (5,  5 )),
    (50, 50, 10, 10,  5,  5, (100, 100), (20, 20), (15, 15 ))
])
def test_subpixel_using_roi(x, y, x1, y1, xs, ys, size_arr, size_roi, expected):
    source = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)
    destination = np.arange(size_arr[0]*size_arr[1]).reshape(size_arr)
    s_roi = Roi(source, x, y, size_x=size_roi[0], size_y=size_roi[1])
    d_roi = Roi(destination, x1, y1, size_x=size_roi[0], size_y=size_roi[1])

    # Then subpixel matching happens on the two ROIs
    x_shift = xs
    y_shift = ys    

    new_d_x = d_roi.x + x_shift
    new_d_y = d_roi.y + y_shift

    assert new_d_x == expected[0]
    assert new_d_y == expected[1]
