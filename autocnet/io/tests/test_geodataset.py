import pytest

from autocnet.io.geodataset import AGeoDataset
from autocnet.examples import get_path
from autocnet.camera import sensor_model as sm
from autocnet.spatial.surface import EllipsoidDem

@pytest.fixture
def ctx_path():
    return get_path('G02_019154_1800_XN_00N133W.crop.cub')

def test_instantiate_isis_sensor(ctx_path):
    obj = AGeoDataset(ctx_path, 'isis')
    assert isinstance(obj.sensormodel, sm.BaseSensor)
    assert isinstance(obj.sensormodel, sm.ISISSensor)

def test_instantiate_csm_sensor(ctx_path):
    obj = AGeoDataset(ctx_path, 'csm')
    assert isinstance(obj.sensormodel, sm.BaseSensor)
    assert isinstance(obj.sensormodel, sm.CSMSensor)    

