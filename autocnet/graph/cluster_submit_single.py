import json
import socket
import sys

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from autocnet.graph.node import NetworkNode
from autocnet.graph.edge import NetworkEdge
from autocnet.utils.utils import import_func
from autocnet.utils.serializers import object_hook
from autocnet.io.db.model import Measures, Points, Overlay, Images
from autocnet.io.db.connection import retry, new_connection

apply_iterable_options = {
                'measures' : Measures,
                'measure' : Measures,
                'm' : Measures,
                2 : Measures,
                'points' : Points,
                'point' : Points,
                'p' : Points,
                3 : Points,
                'overlaps': Overlay,
                'overlap' : Overlay,
                'o' :Overlay,
                4: Overlay,
                'image': Images,
                'images': Images,
                'i': Images,
                5: Images
            }



@retry(max_retries=5)
def _instantiate_obj(msg):
    """
    Instantiate either a NetworkNode or a NetworkEdge that is the
    target of processing.

    """
    along = msg['along']
    id = msg['id']
    image_path = msg['image_path']
    if along == 'node':
        obj = NetworkNode(node_id=id, image_path=image_path)
    elif along == 'edge':
        obj = NetworkEdge()
        obj.source = NetworkNode(node_id=id[0], image_path=image_path[0])
        obj.destination = NetworkNode(node_id=id[1], image_path=image_path[1])
    return obj

@retry(max_retries=5)
def _instantiate_row(msg, session):
    """
    Instantiate some db.io.model row object that is the target
    of processing.
    """
    # Get the dict mapping iterable keyword types to the objects
    obj = apply_iterable_options[msg['along']]
    res = session.query(obj).filter(getattr(obj, 'id')==msg['id']).one()
    session.expunge_all() # Disconnect the object from the session
    return res

@retry()
def execute_func(func, *args, **kwargs):
    return func(*args, **kwargs)

def process(msg):
    """
    Given a message, instantiate the necessary processing objects and
    apply some generic function or method.

    Parameters
    ----------
    msg : dict
          The message that parametrizes the job.
    """
    from sqlalchemy.orm import Session

    # Deserialize the message
    msg = json.loads(msg, object_hook=object_hook)

    _, engine = new_connection(msg['config']['database'])
    
    if msg['along'] in ['node', 'edge']:
        obj = _instantiate_obj(msg)
    elif msg['along'] in ['points', 'measures', 'overlaps', 'images']:
        with Session(engine) as session:
            obj = _instantiate_row(msg, session)
    else:
        obj = msg['along']

    # Grab the function and apply. This assumes that the func is going to
    # have a True/False return value. Basically, all processing needs to
    # occur inside of the func, nothing belongs in here.
    #
    # All args/kwargs are passed through the RedisQueue, and then right on to the func.
    func = msg['func']
    with Session(engine) as session:
        if callable(func):  # The function is a de-serialzied function
            msg['args'] = (obj, *msg['args'])
            msg['kwargs']['session'] = session
        elif hasattr(obj, msg['func']):  # The function is a method on the object
            func = getattr(obj, msg['func'])
        else:  # The func is a function from a library to be imported
            func = import_func(msg['func'])
            # Get the object begin processed prepended into the args.
            msg['args'] = (obj, *msg['args'])
            # For now, pass all the potential config items through
            # most funcs will simply discard the unnecessary ones.
            msg['kwargs']['session'] = session

    # Now run the function.
    res = execute_func(func,*msg['args'], **msg['kwargs'])

    del Session
    del engine

    # Update the message with the True/False
    msg['results'] = res
    # Update the message with the correct callback function

    return msg

def main():
    msg = ''.join(sys.argv[1:])
    result = process(msg)
    print(result)
    
if __name__ == '__main__':
    main()
