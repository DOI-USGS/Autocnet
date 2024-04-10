import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal


from autocnet.transformation import affine, roi
from autocnet.io.geodataset import AGeoDataset
from autocnet.examples import get_path
from skimage.transform import AffineTransform


@pytest.fixture
def g02_ctx():
    return AGeoDataset(get_path('G02_019154_1800_XN_00N133W.crop.cub'), sensortype='isis')

@pytest.fixture
def n06_ctx():
    return AGeoDataset(get_path('N06_064753_1800_XN_00S133W.crop.cub'), sensortype='isis')

@pytest.fixture
def n6_roi(n06_ctx):
    return roi.Roi(n06_ctx, 300,300)

@pytest.fixture
def g02_roi(g02_ctx):
    return roi.Roi(g02_ctx, 100,100)

def test_isis_estimate_affine_transformation(g02_ctx, n06_ctx):
    projective_transform = affine.estimate_affine_from_sensors(g02_ctx, n06_ctx, 150, 150)
    assert_array_almost_equal(projective_transform.params, np.array([[ 9.93968209e-01, -3.54664170e-03, -1.74012988e+01],
                                                                     [-1.13094574e-03,  1.00394116e+00,  4.91146729e+01],
                                                                     [ 1.65323051e-06,  1.51352489e-05,  1.00000000e+00]]))

def test_isis_estimate_local_affine(g02_roi, n6_roi):
    affine_transform = affine.estimate_local_affine(g02_roi, n6_roi)
    assert_array_almost_equal(affine_transform.params, np.array([[ 9.91887398e-01, -9.64539831e-04,  1.86535259e+00],
                                                                 [-1.37864400e-03,  9.99977935e-01,  2.87845656e-01],
                                                                 [ 2.18858798e-07,  5.47986394e-06,  9.98828912e-01]]))