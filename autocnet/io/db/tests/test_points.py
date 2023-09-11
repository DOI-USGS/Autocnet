from unittest.mock import MagicMock, patch
import pytest

import shapely
from autocnet.io.db.model import Points
from autocnet.spatial import sensor

def test_points_exists(tables):
    assert Points.__tablename__ in tables

def test_latitudinal_srid():
    p = Points(adjusted=shapely.Point(0,0,0))
    assert p.latitudinal_srid == 4326

def test_rectangular_srid():
    p = Points(adjusted=shapely.Point(0,0,0))
    assert p.rectangular_srid == 4978

@pytest.mark.parametrize("data", [
    {'id':1, 'pointtype':2},
    {'pointtype':2, 'identifier':'123abc'},
    {'pointtype':3, 'apriori':shapely.Point(0,0,0)},
    {'pointtype':3, 'adjusted':shapely.Point(0,0,0)},
    {'pointtype':2, 'adjusted':shapely.Point(1,1,1), 'ignore':False}
])
def test_create_point(session, data):
    p = Points.create(session, **data)
    resp = session.query(Points).filter(Points.id == p.id).first()
    assert p == resp

@pytest.mark.parametrize("data, expected", [
    ({'pointtype':3, 'adjusted':shapely.Point(0,-1000000,0)}, 
                                shapely.Point(270, 0)),
    ({'pointtype':3}, None)
])
def test_create_point_geom(session, data, expected):
    p = Points.create(session, **data)
    resp = session.query(Points).filter(Points.id == p.id).first()

    assert resp.geom == expected

@pytest.mark.parametrize("data, new_adjusted, expected", [
    ({'pointtype':3, 'adjusted':shapely.Point(0,-100000,0)}, None, None),
    ({'pointtype':3, 'adjusted':shapely.Point(0,-100000,0)}, 
                                shapely.Point(0,100000,0), 
                                shapely.Point(90, 0)),
    ({'pointtype':3}, shapely.Point(0,-100000,0), shapely.Point(270, 0))
])
def test_update_point_geom(session, data, new_adjusted, expected):
    p = Points.create(session, **data)
    p.adjusted = new_adjusted
    session.commit()
    resp = session.query(Points).filter(Points.id == p.id).first()
    assert resp.geom == expected

def test_create_point_with_reference_measure(session):
    point_geom = shapely.Point(0,0,1)
    reference_node = MagicMock()
    reference_node.isis_serial = 'serialnum'
    d = {'node_id':1}
    reference_node.__getitem__.side_effect = d.__getitem__
    sampleline = shapely.Point(1,1)

    point = Points.create_point_with_reference_measure(point_geom, reference_node, sampleline)

    assert len(point.measures) == 1
    assert point.geom.x == 0
    assert point.geom.y == 90.0
    assert point.measures[0].imageid == 1
    assert point.measures[0].sample == sampleline.x
    assert point.measures[0].line == sampleline.y

def test_add_measures_to_point(session):
    point = Points()
    point.adjusted = shapely.Point(0,0,0)
    test_sensor = sensor.create_sensor('isis')
    
    node = MagicMock()
    node.isis_serial = 'serial'
    node.__getitem__.side_effect = {'node_id':0, 'image_path':'/'}.__getitem__

    reference_nodes = [node, node, node, node]

    with patch('autocnet.spatial.isis.ground_to_image') as mocked_call:
        mocked_call.return_value = (0.5, 0.5)
        point.add_measures_to_point(reference_nodes, test_sensor)

    assert len(point.measures) == 4
    assert point.measures[0].line == 0.5
    assert point.measures[1].sample == 0.5
    assert point.measures[2].serial == 'serial'
    assert point.measures[3].imageid == 0