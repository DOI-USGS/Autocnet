from unittest import mock
import pandas as pd
import pytest
from autocnet.io.db import model
from autocnet.io.db.controlnetwork import db_to_df, update_from_jigsaw

def test_to_isis(session):
    controlnetwork = pd.DataFrame([[1,2,3,'a',0, '01010000800000000000003E4000000000000024400000000000001440'],
                                   [2,2,3,'a',0, '01010000800000000000003E4000000000000024400000000000001440'],
                                   [3,2,3,'a',1, '01010000800000000000003E4000000000000024400000000000001440'],
                                   [4,2,3,'a',1, '01010000800000000000003E4000000000000024400000000000001440'],
                                   [5,3,3,'a',2, '01010000800000000000003E4000000000000024400000000000001440'],
                                   [6,3,3,'a',2, '01010000800000000000003E4000000000000024400000000000001440']],
                                   columns=['id', 'pointtype', 'measuretype',
                                            'identifier','pointid', 'apriori'])
    with mock.patch('pandas.read_sql') as mock_db_response:
        mock_db_response.return_value = controlnetwork
    
        df = db_to_df(session.bind)

    assert len(df) == 6
    assert df.iloc[0]['pointtype'] == 2
    assert df.iloc[4]['pointtype'] == 3
    assert df.iloc[0]['measuretype'] == 3
    assert df.iloc[0]['aprioriCovar'] == []

@pytest.mark.xfail
def test_update_from_jigsaw(session, db_controlnetwork,):
    connection = session.get_bind()

    existing_measures = pd.read_sql_table('measures', con=connection)
    session.close()

    # This is an intentionally truncated representation so that many unused columns are not repeated.
    isis_cnet = pd.DataFrame(
        [(0.1, 0.1, 0, 'Random0:123', False, 0.0, 0.0),
         (0.2, 0.2, 0, 'Random1:123', False, 1.1, 1.0),
         (-0.5, -0.5, 1, 'Random0:123', False, -0.2, 0.0),
         (8, -11, 1, 'Random1:123', False, 1.0, 0.0),
         (2.2, 1.1, 2, 'Random0:123', False, 1.0, 0.0),
         (0.25, -34, 2, 'Random1:123', False, 0.5, 0.5)
         ],
        columns=['sampleResidual', 'lineResidual', 'id', 'serialnumber','measureJigsawRejected', 'samplesigma', 'linesigma'])
    update_from_jigsaw(isis_cnet, existing_measures, connection)

    updated_measures = pd.read_sql_table('measures', con=connection)
    session.close()
    assert (updated_measures['sampler'] == pd.Series([0.1, 0.2, -0.5, 8, 2.2, 0.25])).all()
    assert (updated_measures['liner'] == pd.Series([0.1, 0.2, -0.5, -11, 1.1, -34])).all()
    assert (updated_measures['samplesigma'] == pd.Series([0.0, 1.1, -0.2, 1.0, 1.0, 0.5])).all()
    assert (updated_measures['linesigma'] == pd.Series([0.0, 1.0, 0.0, 0.0, 0.0, 0.5])).all()