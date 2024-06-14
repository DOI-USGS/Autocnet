from contextlib import nullcontext
import logging
import warnings

from sqlalchemy.sql.expression import bindparam

from autocnet.io.db.connection import retry
from autocnet.io.db.model import Images, Overlay, Points, Measures
from autocnet.graph.node import NetworkNode

@retry
def update_measures(ncg, session, measures_iterable_to_update):
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

@retry
def ignore_measures(ncg, session, measures_iterable_to_ignore, chooser):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        measures_to_set_false = [{'_id':i} for i in measures_to_set_false]
        # Set ignore=True measures that failed
        stmt = Measures.__table__.update().\
                                where(Measures.__table__.c.id == bindparam('_id')).\
                                values({'measureIgnore':True,
                                        'ChooserName':chooser})
        session.execute(stmt, measures_to_set_false)

@retry
def get_nodes_for_overlap(ncg, session, overlap):
    # If an NCG is passed, instantiate a session off the NCG, else just pass the session through
    nodes = []
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        for id in overlap.intersections:
            try:
                res = session.query(Images).filter(Images.id == id).one()
            except Exception as e:
                warnings.warn(f'Unable to instantiate image with id: {id} with error: {e}')
                continue
            nn = NetworkNode(node_id=id, 
                             image_path=res.path, 
                             cam_type=res.cam_type,
                             dem=res.dem,
                             dem_type=res.dem_type)
            nodes.append(nn)

@retry
def get_nodes_for_measures(ncg, session, measures):
        nodes = {}
        with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
            for measure in measures:
                res = session.query(Images).filter(Images.id == measure.imageid).one()
                logging.debug(f'Node instantiation image query result: {res.path, res.cam_type, res.dem, res.dem_type}')
                nn = NetworkNode(node_id=measure.imageid, 
                                image_path=res.path,
                                cam_type=res.cam_type,
                                dem=res.dem,
                                dem_type=res.dem_type)
                nodes[measure.imageid] = nn
            session.expunge_all()  
        return nodes  

@retry
def get_overlap(ncg, session, overlapid):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        overlap = session.query(Overlay).filter(Overlay.id == overlapid).one()
        session.expunge_all()
    return overlap

@retry
def get_point(ncg, session, pointid):
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        point = session.query(Points).filter(Points.id == pointid).one()
        session.expunge_all()
    return point



@retry
def bulk_commit(ncg, session, iterable_of_objs_to_commit):
    with ncg.session_scope() if ncg else nullcontext(session) as session:
        session.add_all(iterable_of_objs_to_commit)
        session.commit()
