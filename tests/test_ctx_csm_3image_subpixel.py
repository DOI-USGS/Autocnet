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
def isis_mola_radius_dem():
    isisdata = os.environ.get('ISISDATA', None)
    path = os.path.join(isisdata, 'base/dems/molaMarsPlanetaryRadius0005.cub')
    return path

@pytest.fixture
def g17(isis_mola_radius_dem):
    return {'id':0,
            'name':'G17_024823_2204_XI_40N109W.crop.cub',
            'path':get_path('G17_024823_2204_XI_40N109W.crop.cub'),
            'serial':'MRO/CTX/1005587208:239',
            'cam_type':'csm',
            'dem':isis_mola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def n11(isis_mola_radius_dem):
    return {'id':1,
            'name':'N11_066770_2192_XN_39N109W.crop.cub',
            'path':get_path('N11_066770_2192_XN_39N109W.crop.cub'),
            'serial':'MRO/CTX/1287982533:042',
            'cam_type':'csm',
            'dem':isis_mola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def p10(isis_mola_radius_dem):
    return {'id':2,
            'name':'P10_005031_2197_XI_39N109W.crop.cub',
            'path':get_path('P10_005031_2197_XI_39N109W.crop.cub'),
            'serial':'MRO/CTX/0872335765:217',
            'cam_type':'csm',
            'dem':isis_mola_radius_dem,
            'dem_type':'radius'}

@pytest.fixture
def g17_image(g17):
    return Images(**g17)

@pytest.fixture
def n11_image(n11):
    return Images(**n11)

@pytest.fixture
def p10_image(p10):
    return Images(**p10)

@pytest.fixture
def measure_0():
    data = [
        {'id':0, 'imageid':0,'apriorisample':384.768,'aprioriline':278.000,'sample':384.768,'line':278.000,'pointid':0, 'measuretype':3},
        {'id':1, 'imageid':1,'apriorisample':521.905,'aprioriline':209.568,'sample':521.905,'line':209.568, 'pointid':0, 'measuretype':3},
        {'id':2, 'imageid':2,'apriorisample':355.483,'aprioriline':234.296,'sample':355.483,'line':234.296, 'pointid':0, 'measuretype':3}
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
def session(measure_0,g17_image,n11_image,p10_image):

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
            [mock.call.query(Images),
                mock.call.filter(Images.id == 0)],
                [g17_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 1)],
                [n11_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 2)],
                [p10_image]
        ),
        ])
    return session
    
def test_ctx_pair_to_df(session,
                        g17_image,
                        n11_image,
                        p10_image,
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

        # Somewhere in subpixel, need to add the offsets back to samp/line based
        # on which image is being used. Every samp/line, patch time.

        measures_to_update, measures_to_set_false = smart_register_point(point, 
                                                                         session,
                                                                         parameters=parameters,
                                                                         shared_kwargs=shared_kwargs)

        assert measures_to_set_false == []

        m0 = measures_to_update[0]
        assert m0['sample'] == 528.0616518160688
        assert m0['line'] == 210.8722056871887
        assert m0['template_metric'] == 0.8538808822631836
        assert m0['ignore'] == False
        assert m0['template_shift'] == 445.17368002294427
        
        m1 = measures_to_update[1]
        assert m1['sample'] == 357.3392609551714
        assert m1['line'] == 230.29507805238129
        assert m1['template_metric'] == 0.6922665238380432
        assert m1['ignore'] == False
        assert m1['template_shift'] == 175.5325037366171

        dfs = []
        with mock.patch('pandas.read_sql') as db_response:
            db_cnet = pd.DataFrame([
                [0, 0, g17_image.serial,measure_0[0].sample, measure_0[0].line, point.pointtype, measure_0[0].measuretype, 'test'],
                [1, 0, n11_image.serial,m0['sample'], m0['line'], point.pointtype, measure_0[1].measuretype, 'test'],
                [1, 0, p10_image.serial,m1['sample'], m1['line'], point.pointtype, measure_0[2].measuretype, 'test'],
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
    to_isis(df, 'tests/artifacts/ctx_csm_trio_to_df.cnet', targetname='Mars')
    write_filelist([g17_image.path, n11_image.path, p10_image.path], 'tests/artifacts/ctx_csm_trio_to_df.lis')
    assert False