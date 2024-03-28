import pytest

from autocnet.transformation import affine
from autocnet.io.geodataset import AGeoDataset
from autocnet.examples import get_path


@pytest.fixture
def g02_ctx():
    return get_path('G02_019154_1800_XN_00N133W.crop.cub')

@pytest.fixture
def n06_ctx():
    return get_path('N06_064753_1800_XN_00S133W.crop.cub')

def test_isis_estimate_affine_transformation(g02_ctx, n06_ctx):
    gd_base = AGeoDataset(g02_ctx, sensortype='isis')
    gd_match = AGeoDataset(n06_ctx, sensortype='isis')
    affine_transform = affine.estimate_affine_from_sensors(gd_base, gd_match, 150, 150)
    assert affine_transform.rotation == pytest.approx(-0.0014690876698891149, 6)
    assert affine_transform.shear == pytest.approx(0.006990455804590304)
    assert affine_transform.scale[0] == pytest.approx(0.99125799, 6)
    assert affine_transform.scale[1] == pytest.approx(0.99840631, 6) 
    assert affine_transform.translation[0] == pytest.approx(-17.03364091, 6)
    assert affine_transform.translation[1] == pytest.approx(49.44769083, 6)