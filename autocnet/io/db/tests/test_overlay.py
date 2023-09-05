import pytest
import sqlalchemy
from shapely import Polygon, Point
from autocnet.io.db.model import Overlay

def test_overlay_exists(tables):
    assert Overlay.__tablename__ in tables

def test_latitudinal_srid(session):
    """
    Tests that the object is properly setting the latitudinal SRID
    from the configuration file.
    """
    o1={'id':1, 'intersections':[1,2,3],
        'geom':Polygon([(0,0), (-1,0), (-1,-1), (0,-1), (0,0)])}
    a = Overlay.create(session, **o1)
    assert a.latitudinal_srid == 4326
    session.commit()
    res = session.query(Overlay).first()
    assert res.id == 1
    assert res.latitudinal_srid == 4326

@pytest.mark.parametrize('data', [
    {'id':1},
    {'id':1, 'intersections':[1,2,3]},
    {'id':1, 'intersections':[1,2,3],
    'geom':Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])}

])
def test_create_overlay(session, data):
    d = Overlay.create(session, **data)
    resp = session.query(Overlay).filter(Overlay.id == d.id).first()
    assert d == resp

def test_overlapping_larger_than(session):
    o1={'id':1, 'intersections':[1,2,3],
        'geom':Polygon([(0,0), (-1,0), (-1,-1), (0,-1), (0,0)])}
    o2={'id':2, 'intersections':[1,2,3],
        'geom':Polygon([(0,0), (.2,0), (.2,.2), (0,.2), (0,0)])}
    a = Overlay.create(session, **o1)
    b = Overlay.create(session, **o2)
    session.commit()

    larger = Overlay.overlapping_larger_than(0.5, session)
    assert len(larger) == 1
    assert larger[0].id == 1
