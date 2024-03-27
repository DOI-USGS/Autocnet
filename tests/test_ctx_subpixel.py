import pytest
import mock
from autocnet.matcher.subpixel import smart_register_point
from mock_alchemy.mocking import UnifiedAlchemyMagicMock
from autocnet.io.db.model import Points, Measures, Images

from autocnet.examples import get_path

@pytest.fixture
def g02():
    return {'id':0,
            'name':'G02_019154_1800_XN_00N133W.crop.cub',
            'path':get_path('G02_019154_1800_XN_00N133W.crop.cub')}

@pytest.fixture
def n06():
    return {'id':1,
            'name':'N06_064753_1800_XN_00S133W.crop.cub',
            'path':get_path('N06_064753_1800_XN_00S133W.crop.cub')}

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
                    id=0)
@pytest.fixture
def n06_measureA():
    return Measures(imageid=1,
                    apriorisample=359.498,
                    aprioriline=536.333,
                    sample=359.498,
                    line=536.333,
                    pointid=0,
                    id=1)
@pytest.fixture
def pointA(g02_measureA, n06_measureA):
    point = Points(id=0, 
                   reference_index=0)
    point.measures = [g02_measureA,
                      n06_measureA]
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
    # Mock-alchemy returns all on filter.query, how to setup properly w/ filter?
    measures_to_update, measures_to_set_false = smart_register_point(pointA, 
                                                                     session,
                                                                     parameters=parameters,
                                                                     shared_kwargs=shared_kwargs)

    assert measures_to_set_false == []

    m0 = measures_to_update[0]
    assert m0['sample'] == 364.7675611360247
    assert m0['line'] == 525.3550626650527
    assert m0['template_metric'] == 0.625694990158081
    assert m0['ignore'] == False
    assert m0['template_shift'] == 238.62787986416774
