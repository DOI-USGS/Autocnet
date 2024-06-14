import logging
import socket
from time import sleep

from sqlalchemy import orm, create_engine, pool

# set up the logging file
log = logging.getLogger(__name__)

class Parent:
    def __init__(self, config):
        Session, _ = new_connection(config)
        self.session = Session()
        self.session.begin()

def retry(max_retries=5, wait_time=300):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            if retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except:
                    retries += 1
                    sleep(wait_time)
            else:
                raise Exception(f"Maximum retries of {func} exceeded! Is the database accessible?")
        return wrapper
    return decorator

@retry
def new_connection(dbconfig):
    """
    Using the user supplied config create a NullPool database connection.

    Parameters
    ----------
    dbconfig : dict
               Dictionary defining necessary parameters for the database
               connection

    Returns
    -------
    Session : object
              An SQLAlchemy session object

    engine : object
             An SQLAlchemy engine object
    """
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(dbconfig['username'],
                                                  dbconfig['password'],
                                                  dbconfig['host'],
                                                  dbconfig['pgbouncer_port'],
                                                  dbconfig['name'])
    hostname = socket.gethostname()
    engine = create_engine(db_uri,
                poolclass=pool.NullPool,
                connect_args={"application_name":f"AutoCNet_{hostname}"},
                isolation_level="AUTOCOMMIT",
                pool_pre_ping=True)
    Session = orm.sessionmaker(bind=engine, autocommit=False)
    log.debug(Session, engine)
    return Session, engine
