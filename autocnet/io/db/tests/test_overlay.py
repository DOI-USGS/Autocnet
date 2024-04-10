import pytest
from shapely import Polygon
from autocnet.io.db.model import Overlay


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

    session.filter.assert_called_once()
