from contextlib import nullcontext
import logging

from sqlalchemy.sql.expression import bindparam
from sqlalchemy.orm import joinedload
from autocnet.io.db.connection import retry
from autocnet.io.db.model import Images, Overlay, Points, Measures
from autocnet.graph.node import NetworkNode

# set up the logger file
log = logging.getLogger(__name__)


#@retry()
def update_measures(ncg, session, measures_iterable_to_update):
    if not measures_iterable_to_update:
        return
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        stmt = Measures.__table__.update().\
                        where(Measures.__table__.c.id == bindparam('_id')).\
                        values({'weight':bindparam('weight'),
                                'measureIgnore':bindparam('ignore'),
                                'templateMetric':bindparam('template_metric'),
                                'templateShift':bindparam('template_shift'),
                                        'line': bindparam('line'),
                                        'sample':bindparam('sample'),
                                        'ChooserName':bindparam('choosername')})
        session.execute(stmt, measures_iterable_to_update)
    return

#@retry()
def ignore_measures(ncg, session, measures_iterable_to_ignore, chooser):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        measures_to_set_false = [{'_id':i} for i in measures_iterable_to_ignore]
        # Set ignore=True measures that failed
        stmt = Measures.__table__.update().\
                                where(Measures.__table__.c.id == bindparam('_id')).\
                                values({'measureIgnore':True,
                                        'ChooserName':chooser})
        session.execute(stmt, measures_to_set_false)
    return 

@retry(wait_time=30)
def get_nodes_for_overlap(ncg, session, overlap):
    # If an NCG is passed, instantiate a session off the NCG, else just pass the session through
    ids = tuple([i for i in overlap.intersections])
    nodes = []
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        results = session.query(Images).filter(Images.id.in_(ids)).all()
    
        for res in results:
            nn = NetworkNode(node_id=res.id, 
                            image_path=res.path, 
                            cam_type=res.cam_type,
                            dem=res.dem,
                            dem_type=res.dem_type)
            nodes.append(nn)
    return nodes

@retry(wait_time=30)
def get_nodes_for_measures(ncg, session, measures):
    nodes = {}
    imageids = tuple([measure.imageid for measure in measures])
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        results = session.query(Images).filter(Images.id.in_(imageids)).all()
        for res in results:
            nn = NetworkNode(node_id=res.id, 
                             image_name=res.name,
                            image_path=res.path,
                            cam_type=res.cam_type,
                            dem=res.dem,
                            dem_type=res.dem_type)
            nodes[res.id] = nn
    return nodes  

@retry(wait_time=30)
def get_overlap(ncg, session, overlapid):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        overlap = session.query(Overlay).filter(Overlay.id == overlapid).one()
        session.expunge_all()
    return overlap

@retry(wait_time=30)
def get_point(ncg, session, pointid):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        point = session.query(Points). \
                filter(Points.id == pointid). \
                options(joinedload('*')). \
                one()
        session.expunge_all()
    return point




def bulk_commit(ncg, session, iterable_of_objs_to_commit):
    with ncg.session_scope() if ncg else nullcontext(session) as session:
        session.add_all(iterable_of_objs_to_commit)
        session.commit()
    return
