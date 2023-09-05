from datetime import datetime
import json
import sys

import numpy as np
import pandas as pd
import pytest
import sqlalchemy

from autocnet.io.db import model
from autocnet.graph.network import NetworkCandidateGraph

from shapely.geometry import MultiPolygon, Polygon, Point

if sys.platform.startswith("darwin"):
    pytest.skip("skipping DB tests for MacOS", allow_module_level=True)

def test_keypoints_exists(tables):
    assert model.Keypoints.__tablename__ in tables

def test_edges_exists(tables):
    assert model.Edges.__tablename__ in tables

def test_costs_exists(tables):
    assert model.Costs.__tablename__ in tables

def test_matches_exists(tables):
    assert model.Matches.__tablename__ in tables

def test_cameras_exists(tables):
    assert model.Cameras.__tablename__ in tables

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables

def test_points_history_exists(tables):
    assert model.PointsHistory.__tablename__ in tables

def test_measures_history_exists(tables):
    assert model.MeasuresHistory.__tablename__ in tables

def test_job_history_exists(tables):
    assert model.JobsHistory.__tablename__ in tables


def test_create_camera_without_image(session):
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(session, **{'image_id':1})

def test_create_camera(session):
    #with pytest.raises(sqlalchemy.exc.IntegrityError):
    c = model.Cameras.create(session)
    res = session.query(model.Cameras).first()
    assert c.id == res.id

def test_create_camera_unique_constraint(session):
    model.Images.create(session, **{'id':1})
    data = {'image_id':1}
    model.Cameras.create(session, **data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Cameras.create(session, **data)



'''@pytest.mark.parametrize('data', [
    {'id':1},
    {'serial':'foo'}
])
def test_create_images_constrined(session, data):
    """
    Test that the images unique constraint is being observed.
    """
    model.Images.create(session, **data)
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        model.Images.create(session, **data)'''






# def test_point_trigger(session):
#     original = 3
#     new_type = 2

#     data = {'pointtype':original, 'adjusted' : Point(1,10000,1)}

#     with session as s:
#         p = model.Points.create(s, **data)

#         p.pointtype = new_type
#         # p.pointtype(new_type)
#         s.commit()
#         s.delete(p)
#         s.commit()

#         resp = session.query(model.PointsHistory).filter(model.PointsHistory.fk == p.id).first()
#         print(f"\nstart of debugging output for test_point_trigger\n_________________________________________\nthe value of p is: {p}\n----\nthe value of p.id is: {p.id}\n----\nthe value of resp is below\n----\n{resp}\n_________________________________________\nEnd of debugging output for test_point_trigger\n")

#         assert resp.event == "insert"
#         assert resp[0].before == None
#         assert resp[0].after["pointType"] == original 

#         assert resp[1].event == "update"
#         assert resp[1].before['pointType'] == original
#         assert resp[1].after["pointType"] == new_type

#         assert resp[2].event == "delete"
#         assert resp[2].before['pointType'] == new_type
#         assert resp[2].after == None
#         s.close()

@pytest.mark.xfail(reason="Unknown issue on GitHub actions, passes locally")
def test_measure_trigger(session):
    original = 3
    new_type = 2

    with session as s:     
        point_data = {'pointtype':original, 'adjusted' : Point(1,10000,1)}
        p = model.Points.create(s, **point_data)

        measure_data = {'id' : 100 ,'sample': original, 'line': 10, 'pointid': p.id, 'serial': 'measure', '_measuretype' : 2}
        m = model.Measures(**measure_data)
        s.add(m)
        s.commit()

        measures = s.query(model.Measures).all()
        points = s.query(model.Points).all()
        print(measures, len(measures))
        print(points, len(points))

        m.sample = new_type
        s.commit()
        s.delete(p) 
        s.delete(m)
        s.commit()
 
    measures = session.query(model.Measures).all()
    points = session.query(model.Points).all()
    print(measures, len(measures))
    print(points, len(points))

    resp = session.query(model.MeasuresHistory).filter(model.MeasuresHistory.fk == m.id).all()
    print(resp, len(resp))
    assert resp[0].event == "insert"
    assert resp[0].before == None
    assert resp[0].after["sample"] == original 

    assert resp[1].event == "update"
    assert resp[1].before['sample'] == original
    assert resp[1].after["sample"] == new_type

    assert resp[2].event == "delete"
    assert resp[2].before['sample'] == new_type
    assert resp[2].after == None


@pytest.mark.parametrize("measure_data, point_data, image_data", [(
    [{'id': 1, 'pointid': 1, 'imageid': 1, 'serial': 'ISISSERIAL1', 'measuretype': 3, 'sample': 0, 'line': 0},
     {'id': 2, 'pointid': 1, 'imageid': 2, 'serial': 'ISISSERIAL2', 'measuretype': 3, 'sample': 0, 'line': 0},
     {'id': 3, 'pointid': 1, 'imageid': 3, 'serial': 'ISISSERIAL3', 'measuretype': 3, 'sample': 0, 'line': 0},
     {'id': 4, 'pointid': 1, 'imageid': 4, 'serial': 'ISISSERIAL4', 'measuretype': 3, 'sample': 0, 'line': 0}],
    {'id':1,
     'pointtype':2},
    [{'id':1, 'serial': 'ISISSERIAL1'},
     {'id':2, 'serial': 'ISISSERIAL2'},
     {'id':3, 'serial': 'ISISSERIAL3'},
     {'id':4, 'serial': 'ISISSERIAL4'}])])
def test_ignore_image(session, measure_data, point_data, image_data):
    for data in image_data:
        model.Images.create(session, **data)
    model.Points.create(session, **point_data)
    for data in measure_data:
        model.Measures.create(session, **data)
    image_resp = session.query(model.Images).filter(model.Images.id == 1).first()
    image_resp.ignore = True
    ignored_measures_resp = session.query(model.Measures).filter(model.Measures.ignore == True).first()
    assert ignored_measures_resp.imageid == 1
    valid_measures_resp = session.query(model.Measures).filter(model.Measures.ignore == False)
    assert valid_measures_resp.count() == 3
