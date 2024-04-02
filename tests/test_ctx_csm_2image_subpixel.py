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
def g02():
    return {'id':0,
            'name':'G02_019154_1800_XN_00N133W.crop.cub',
            'path':get_path('G02_019154_1800_XN_00N133W.crop.cub'),
            'serial':'MRO/CTX/0967420440:133',
            'cam_type': 'csm'}

@pytest.fixture
def n06():
    return {'id':1,
            'name':'N06_064753_1800_XN_00S133W.crop.cub',
            'path':get_path('N06_064753_1800_XN_00S133W.crop.cub'),
            'serial':'MRO/CTX/1274405748:172',
            'cam_type':'csm'}

@pytest.fixture
def g02_image(g02):
    return Images(**g02)

@pytest.fixture
def n06_image(n06):
    return Images(**n06)

@pytest.fixture
def g02_measureA():
    return Measures(imageid=0,
                    apriorisample=379.473,
                    aprioriline=489.012,
                    sample=379.473,
                    line=489.012,
                    pointid=0,
                    id=0,
                    measuretype=3)
@pytest.fixture
def n06_measureA():
    return Measures(imageid=1,
                    apriorisample=359.498,
                    aprioriline=536.333,
                    sample=359.498,
                    line=536.333,
                    pointid=0,
                    id=1,
                    measuretype=3)
@pytest.fixture
def pointA(g02_measureA, n06_measureA):
    point = Points(id=0, 
                   reference_index=0,
                   pointtype=2)
    point.measures = [g02_measureA,
                      n06_measureA]
    point._apriori = '0101000080A6F7E802D4BE41C1B8674C6EDDE442C12BB5B6271BC9CEC0'
    return point

@pytest.fixture
def session(g02_image, n06_image,
            g02_measureA,n06_measureA):

    # Mocked DB session with calls and responses.
    session = UnifiedAlchemyMagicMock(data=[
        (
            [mock.call.query(Measures),
                mock.call.filter(Measures.pointid == 0),
                mock.call.order_by(Measures.id)],
                [g02_measureA,n06_measureA]
        ),(
            [mock.call.query(Measures),
             mock.call.filter(Measures.id == 1),
             mock.call.order_by(Measures.id)],
             [n06_measureA]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 0)],
                [g02_image]
        ),(
            [mock.call.query(Images),
                mock.call.filter(Images.id == 1)],
                [n06_image]
        ),
        ])
    return session
    
def test_ctx_pair_to_df(session,
                        g02_image,
                        n06_image,
                        g02_measureA,
                        n06_measureA,
                        pointA):
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

    measures_to_update, measures_to_set_false = smart_register_point(pointA, 
                                                                     session,
                                                                     parameters=parameters,
                                                                     shared_kwargs=shared_kwargs)

    assert measures_to_set_false == []

    m0 = measures_to_update[0]
    assert m0['sample'] == 364.6853301321431
    assert m0['line'] == 525.3395556759572
    assert m0['template_metric'] == 0.6238831281661987
    assert m0['ignore'] == False
    assert m0['template_shift'] == 238.67623291833308

    with mock.patch('pandas.read_sql') as db_response:
        db_cnet = pd.DataFrame([
            [0, 0, g02_image.serial,g02_measureA.sample, g02_measureA.line, pointA.pointtype, g02_measureA.measuretype, 'test'],
            [1, 0, n06_image.serial,m0['sample'], m0['line'], pointA.pointtype, n06_measureA.measuretype, 'test']
                                ],
                               columns=['id','pointid', 'serialnumber', 'sample', 'line', 
                                        'pointtype', 'measuretype','identifier'])
        db_response.return_value = db_cnet

        df = db_to_df(session)

    df.rename(columns={'pointtype':'pointType',
                        'measuretype':'measureType'},
                        inplace=True)
    to_isis(df, 'tests/artifacts/ctx_csm_pair_to_df.cnet', targetname='Mars')
    write_filelist([g02_image.path, n06_image.path], 'tests/artifacts/ctx_csm_pair_to_df.lis')
