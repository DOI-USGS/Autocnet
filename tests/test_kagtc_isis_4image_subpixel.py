import os
import pandas as pd
import pytest
import mock
from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from plio.io.io_controlnetwork import to_isis, write_filelist

from autocnet.matcher.subpixel import smart_register_point
from autocnet.io.db.model import Points, Measures, Images
from autocnet.io.db.controlnetwork import db_to_df
from autocnet.examples import get_path
from shapely import Point,to_wkb

@pytest.fixture
def isis_lola_radius_dem():
    isisdata = os.environ.get('ISISDATA', None)
    path = os.path.join(isisdata, 'base/dems/LRO_LOLA_LDEM_global_128ppd_20100915_0002.cub')
    return path

@pytest.fixture
def tc1_487(isis_lola_radius_dem):
    return {'id':0,
            'name':'TC1W2B0_01_02700S189E3487.cub',
            'path':get_path('TC1W2B0_01_02700S189E3487.cub'),
            'serial':'SELENE/TERRAIN CAMERA 1/2008-05-16T22:52:03Z',
            'cam_type':'csm',
            'dem':isis_lola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def tc1_481(isis_lola_radius_dem):
    return {'id':1,
            'name':'TC1W2B0_01_05202S188E3481.cub',
            'path':get_path('TC1W2B0_01_05202S188E3481.cub'),
            'serial':'SELENE/TERRAIN CAMERA 1/2008-12-08T00:11:42Z',
            'cam_type':'csm',
            'dem':isis_lola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def tc2_487(isis_lola_radius_dem):
    return {'id':2,
            'name':'TC2W2B0_01_02700S184E3487.cub',
            'path':get_path('TC2W2B0_01_02700S184E3487.cub'),
            'serial':'SELENE/TERRAIN CAMERA 2/2008-05-16T22:52:30Z',
            'cam_type':'csm',
            'dem':isis_lola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def tc2_481(isis_lola_radius_dem):
    return {'id':3,
            'name':'TC2W2B0_01_05202S184E3481.cub',
            'path':get_path('TC2W2B0_01_05202S184E3481.cub'),
            'serial':'SELENE/TERRAIN CAMERA 2/2008-12-08T00:11:16Z',
            'cam_type':'csm',
            'dem':isis_lola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def tc2_487_image(tc2_487):
    return Images(**tc2_487)

@pytest.fixture
def tc1_487_image(tc1_487):
    return Images(**tc1_487)

@pytest.fixture
def tc1_481_image(tc1_481):
    return Images(**tc1_481)

@pytest.fixture
def tc2_481_image(tc2_481):
    return Images(**tc2_481)

@pytest.fixture
def measure_0():
    data = [
        {'id':0, 'imageid':0,'apriorisample':847.60, 'aprioriline':1648.38,'sample':847.60,'line':1648.38, 'pointid':0, 'measuretype':3},
        {'id':1, 'imageid':1,'apriorisample':2732.23,'aprioriline':2874.40,'sample':2732.23,'line':2874.40, 'pointid':0, 'measuretype':3},
        {'id':2, 'imageid':2,'apriorisample':872.81, 'aprioriline':3232.75,'sample':872.81,'line':3232.75, 'pointid':0, 'measuretype':3},
        {'id':3, 'imageid':3,'apriorisample':2720.61,'aprioriline':1458.21,'sample':2720.61,'line':1458.21, 'pointid':0, 'measuretype':3}
        ]
    return [Measures(**d) for d in data]

@pytest.fixture
def points(measure_0):
    data = [{'id':0,'reference_index':0,'pointtype':2,'measures':[],'_apriori':''},
             #{'id':1,'reference_index':1,'pointtype':2,'measures':[],'_apriori':''},
             #{'id':2,'reference_index':1,'pointtype':2,'measures':[],'_apriori':''},
             #{'id':3,'reference_index':1,'pointtype':2,'measures':[],'_apriori':''},
             #{'id':4,'reference_index':1,'pointtype':2,'measures':[],'_apriori':''}
      ]
    pts = [Points(**d) for d in data]
    pts[0].measures = measure_0
    return pts

@pytest.fixture
def session(measure_0,tc1_487_image,tc1_481_image,tc2_487_image, tc2_481_image):

    # Mocked DB session with calls and responses.
    session = UnifiedAlchemyMagicMock(data=[
        (
            [mock.call.query(Measures),
                mock.call.filter(Measures.pointid == 0),
                mock.call.order_by(Measures.id)],
                measure_0
        ),(
            [mock.call.query(Measures),
             mock.call.filter(Measures.id == 1),
             mock.call.order_by(Measures.id)],
             [measure_0[1]]
        ),(
            [mock.call.query(Measures),
             mock.call.filter(Measures.id == 2),
             mock.call.order_by(Measures.id)],
             [measure_0[2]]
        ),(
            [mock.call.query(Measures),
             mock.call.filter(Measures.id == 3),
             mock.call.order_by(Measures.id)],
             [measure_0[3]]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 0)],
                [tc1_487_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 1)],
                [tc1_481_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 2)],
                [tc2_487_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 3)],
                [tc2_481_image]            
        )
        ])
    return session
    
def test_ctx_pair_to_df(session,
                        tc1_487_image,
                        tc1_481_image,
                        tc2_487_image,
                        tc2_481_image,
                        measure_0,
                        points):
    """
    This is an integration test that takes a pair of ISIS cube files and
    processes them through the smart_subpixel_matcher. This is the same
    matcher that was used by the CTX control project. The test sets up
    a session mock with the appropriate data. This directly of indirectly
    tests the following:

    - subpixel.smart_register_point
    - subpixel.subpixel_register_point_smart
    - subpixel.decider
    - subpixel.check_for_shift_consensus
    - subpixel.validate_candidate_measure
    - transformation.roi.Roi
    - transformation.affine.estimate_affine_transform
    """
    # Pulled directly from CTX control project.
    parameters = [
        {'match_kwargs': {'image_size':(60,60), 'template_size':(30,30)}},
        {'match_kwargs': {'image_size':(75,75), 'template_size':(33,33)}},
        {'match_kwargs': {'image_size':(90,90), 'template_size':(36,36)}},
        {'match_kwargs': {'image_size':(110,110), 'template_size':(40,40)}},
        {'match_kwargs': {'image_size':(125,125), 'template_size':(44,44)}},
        {'match_kwargs': {'image_size':(140,140), 'template_size':(48,48)}}
    ]

    shared_kwargs = {'cost_func':lambda x,y:y,
                     'chooser':'smart_subpixel_registration'}
    for point in points:
        measures_to_update, measures_to_set_false = smart_register_point(point, 
                                                                         session,
                                                                         parameters=parameters,
                                                                         shared_kwargs=shared_kwargs)

        #assert measures_to_set_false == []
        print(measures_to_set_false)
        print(measures_to_update)
        m0 = measures_to_update[0]
        # assert m0['sample'] == pytest.approx(527.257, abs=0.001)
        # assert m0['line'] == pytest.approx(211.025, abs=0.001)  #0.25px!
        # assert m0['template_metric'] == pytest.approx(0.94, abs=0.01)
        # assert m0['ignore'] == False
        # assert m0['template_shift'] == pytest.approx(5.547, abs=0.001)

        m1 = measures_to_update[1]
        # assert m1['sample'] == pytest.approx(357.9, abs=0.001) #0.29px!
        # assert m1['line'] == pytest.approx(230.638, abs=0.001) 
        # assert m1['template_metric'] == pytest.approx(0.88, abs=0.01)
        # assert m1['ignore'] == False
        # assert m1['template_shift'] == pytest.approx(4.384, abs=0.001)

        m2 = measures_to_update[2]
        # assert m2['sample'] == pytest.approx(357.9, abs=0.001) #0.29px!
        # assert m2['line'] == pytest.approx(230.638, abs=0.001) 
        # assert m2['template_metric'] == pytest.approx(0.88, abs=0.01)
        # assert m2['ignore'] == False
        # assert m2['template_shift'] == pytest.approx(4.384, abs=0.001)

        dfs = []
        with mock.patch('pandas.read_sql') as db_response:
            db_cnet = pd.DataFrame([
                [0, 0, tc1_487_image.serial,measure_0[0].sample, measure_0[0].line, point.pointtype, measure_0[0].measuretype, 'test'],
                [1, 0, tc1_481_image.serial,m0['sample'], m0['line'], point.pointtype, measure_0[1].measuretype, 'test'],
                [1, 0, tc2_487_image.serial,m1['sample'], m1['line'], point.pointtype, measure_0[2].measuretype, 'test'],
                [1, 0, tc2_481_image.serial,m2['sample'], m2['line'], point.pointtype, measure_0[3].measuretype, 'test']
                                    ],
                                columns=['id','pointid', 'serialnumber', 'sample', 'line', 
                                            'pointtype', 'measuretype','identifier'])
            db_response.return_value = db_cnet

            df = db_to_df(session)
            dfs.append(df)
    
    df = pd.concat(dfs)
    df.rename(columns={'pointtype':'pointType',
                        'measuretype':'measureType'},
                        inplace=True)
    to_isis(df, 'tests/artifacts/test_kagtc_isis_4image_subpixel.cnet', targetname='Moon')
    write_filelist([tc1_487_image.path, tc1_481_image.path, tc2_487_image.path, tc2_481_image.path], 'tests/artifacts/test_kagtc_isis_4image_subpixel.lis')
    assert False