import sys
import json

from autocnet.graph.network import NetworkCandidateGraph
from autocnet.graph.node import NetworkNode
from autocnet.graph.edge import NetworkEdge
from autocnet.utils.utils import import_func
from autocnet.utils.serializers import JsonEncoder, object_hook

def _instantiate_obj(msg, ncg):
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
    obj.parent = ncg
    return obj

def _instantiate_row(msg, ncg):
    """
    Instantiate some db.io.model row object that is the target
    of processing.
    """
    # Get the dict mapping iterable keyword types to the objects
    objdict = ncg.apply_iterable_options
    obj = objdict[msg['along']]
    with ncg.session_scope() as session:
        res = session.query(obj).filter(getattr(obj, 'id')==msg['id']).one()
        session.expunge_all() # Disconnect the object from the session
    return res

def process(msg):
    """
    Given a message, instantiate the necessary processing objects and
    apply some generic function or method.

    Parameters
    ----------
    msg : dict
          The message that parametrizes the job.
    """
    # Deserialize the message
    msg = json.loads(msg, object_hook=object_hook)

    ncg = NetworkCandidateGraph()
    ncg.config_from_dict(msg['config'])
    if msg['along'] in ['node', 'edge']:
        obj = _instantiate_obj(msg, ncg)
    elif msg['along'] in ['candidategroundpoints', 'points', 'measures', 'overlaps', 'images']:
        obj = _instantiate_row(msg, ncg)
    else:
        obj = msg['along']

    # Grab the function and apply. This assumes that the func is going to
    # have a True/False return value. Basically, all processing needs to
    # occur inside of the func, nothing belongs in here.
    #
    # All args/kwargs are passed through the RedisQueue, and then right on to the func.
    func = msg['func']
    if callable(func):  # The function is a de-serialzied function
        msg['args'] = (obj, *msg['args'])
        msg['kwargs']['ncg'] = ncg
    elif hasattr(obj, msg['func']):  # The function is a method on the object
        func = getattr(obj, msg['func'])
    else:  # The func is a function from a library to be imported
        func = import_func(msg['func'])
        # Get the object begin processed prepended into the args.
        msg['args'] = (obj, *msg['args'])
        # For now, pass all the potential config items through
        # most funcs will simply discard the unnecessary ones.
        msg['kwargs']['ncg'] = ncg
        msg['kwargs']['Session'] = ncg.Session

    # Now run the function.
    res = func(*msg['args'], **msg['kwargs'])

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
