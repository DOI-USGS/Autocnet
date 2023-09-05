import pytest
from shapely import MultiPolygon, Polygon, Point

from autocnet.io.db.model import Images

def test_images_exists(tables):
    assert Images.__tablename__ in tables

@pytest.mark.parametrize('data', [
    {'id':1},
    {'name':'foo',
     'path':'/neither/here/nor/there'},
    ])
def test_create_images(session, data):
    i = Images.create(session, **data)
    resp = session.query(Images).filter(Images.id==i.id).first()
    assert i == resp

def test_null_footprint(session):
    i = Images.create(session, geom=None,
                                      serial = 'serial')
    assert i.geom is None

def test_broken_bad_geom(session):
    # An irreperablly damaged poly
    truthgeom = MultiPolygon([Polygon([(0,0), (1,1), (1,2), (1,1), (0,0)])])
    i = Images.create(session, geom=truthgeom,
                                      serial = 'serial')
    resp = session.query(Images).filter(Images.id==i.id).one()
    assert resp.ignore == True

def test_fix_bad_geom(session):
    truthgeom = MultiPolygon([Polygon([(0,0), (0,1), (1,1), (0,1), (1,1), (1,0), (0,0) ])])
    i = Images.create(session, geom=truthgeom,
                                     serial = 'serial')
    resp = session.query(Images).filter(Images.id==i.id).one()
    assert resp.ignore == False
    assert resp.geom.is_valid == True

def test_get_images_intersecting_point(session):

    # Create test objects and put them into database
    i1 = {'id':1, 
        'geom':MultiPolygon([Polygon([(0,0), (-1,0), (-1,-1), (0,-1), (0,0)])])}
    i2={'id':2,
        'geom':MultiPolygon([Polygon([(0,0), (2,0), (2,2), (0,2), (0,0)])])}
    a = Images.create(session, **i1)
    b = Images.create(session, **i2)
    session.commit()

    point = Point(1,0)
    overlap_ids = Images.get_images_intersecting_point(point, session)
    session.close()
    assert overlap_ids == [2]