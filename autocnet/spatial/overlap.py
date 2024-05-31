from contextlib import nullcontext
import time
import logging
import warnings

import shapely
import json
from subprocess import CalledProcessError

from autocnet.cg import cg as compgeom
from autocnet.graph.node import NetworkNode
from autocnet.io.db.model import Images, Measures, Overlay, Points, JsonEncoder
from autocnet.transformation import roi
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.matcher.validation import is_valid_lroc_image

# set up the logger file
log = logging.getLogger(__name__)

def place_points_in_overlaps(size_threshold=0.0007,
                             distribute_points_kwargs={},
                             cam_type='csm',
                             point_type=2,
                             ncg=None):
    """
    Place points in all of the overlap geometries by back-projecing using
    sensor models.

    Parameters
    ----------
    nodes : dict-link
            A dict like object with a shared key with the intersection
            field of the database Overlay table and a cg node object
            as the value. This could be a NetworkCandidateGraph or some
            other dict-like object.

    Session : obj
              The session object from the NetworkCandidateGraph

    size_threshold : float
                     overlaps with area <= this threshold are ignored
    cam_type : str
               Either 'csm' (default) or 'isis'. The type of sensor model to use.

    point_type : int
                 Either 2 (free;default) or 3 (constrained). Point type 3 should be used for
                 ground.
    """
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    for overlap in Overlay.overlapping_larger_than(size_threshold, ncg.Session):
        if overlap.intersections == None:
            continue
        place_points_in_overlap(overlap,
                                cam_type=cam_type,
                                distribute_points_kwargs=distribute_points_kwargs,
                                point_type=point_type,
                                ncg=ncg)

def find_interesting_point(nodes, lon, lat, size=71, **kwargs):
    """
    Find an interesting point close the given lon, lat given a list data structure that contains
    the image_path and the geodata for the image.

    Parameters
    ___________
    nodes : list 
        A list of autocnet.graph.node or data structure objects containing image_path and geodata
        This contains the image data for all the images the intersect that lat/lon

    lon : int
        The longitude of the point one is interested in

    lat : int
        The latitude of the point one is interested in

    size : int
        The amount of pixel around a points initial location to search for an
        interesting feature to which to shift the point.
    kwargs : dict
        Contain information to be used if the csm sensor model is passed
        csm_kwargs = {
            'semi_major': int,
            'semi_minor': int,
            'height': int,
            'ncg': obj,
            'needs_projection': bool
        }
    Returns
    _______
    reference_index : int
        An index that refers to image that was choosen to be used as the reference image.
        This is the image in which an interesting point was found.
        
    point : shapely.geometry.point
        The intresting point close to the given lat/lon

    """
    if not nodes:
        log.info("Tried to iterate through a node that does not exist, skipping")
        return None, None
    # Iterate through the images to find an interesting point
    for reference_index, node in enumerate(nodes):
        log.debug(f'Trying image: {node["image_path"].split("/")[-1]}')
        # reference_index is the index into the list of measures for the image that is not shifted and is set at the
        # reference against which all other images are registered.
        try_sample, try_line = node.geodata.sensormodel.lonlat2sampline(lon, lat)

        # If sample/line are None, point is not in image
        if try_sample == None or try_line == None:
            log.info(f'point ({lon}, {lat}) does not project to reference image {node["image_path"]}')
            continue

        # This a prevention in case the last sample/line are NULL when itterating
        sample = try_sample
        line = try_line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        try:
            roi_array = image_roi.clipped_array # Units are pixels for the array
        except:
            log.info(f'Failed to find interesting features in image.')
            continue

        # Check if the image is valid and could be used as the reference
        if not is_valid_lroc_image(roi_array):
            log.info('Failed to find interesting features in image due to poor quality image.')
            continue

        # Extract the most interesting feature in the search window
        interesting = extract_most_interesting(image_roi.clipped_array)
        
        if interesting is not None:
            # We have found an interesting feature and have identified the reference point.
            # kps are in the image space with upper left origin and the roi could be the requested size 
            # or smaller if near an image boundary. So use the roi upper left_x and top_y for the actual origin.
            left_x, _, top_y, _ = image_roi.image_extent
            newsample = left_x + interesting.x
            newline = top_y + interesting.y
            log.debug(f'Current reference index: {reference_index}.')
            return reference_index, shapely.geometry.Point(newsample, newline)
    
    # Tried all the images, so return a shapely point un-modified, the last sample/line.
    log.info('Unable to find an interesting point, falling back to the a priori pointing')
    log.debug(f'Current reference index: {reference_index}.')
    return reference_index, shapely.geometry.Point(sample, line)

def place_points_in_overlap(overlap,
                            identifier="place_points_in_overlaps",
                            interesting_func=find_interesting_point,
                            interesting_func_kwargs={"size":71},
                            distribute_points_kwargs={},
                            point_type=2,
                            ncg=None,
                            session=None,
                            use_cache=False,
                            ratio_size=0.1,
                            **kwargs):
    """
    Place points into an overlap geometry by back-projecting using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    overlap : obj
              An autocnet.io.db.model Overlay model instance.

    identifier: str
                The tag used to distinguish points laid down by this function.

    cam_type : str
               options: {"csm", "isis"}
               Pick what kind of camera model implementation to use.

    interesting_func : callable
        A function that takes a list of nodes, a longitude, a latitude, and arbitrary
        kwargs and returns a tuple with a reference index (integer) and a shapely Point object

    interesting_func_kwargs : dict
                              With keyword arguments required by the passed interesting_func

    point_type: int
        The type of point being placed. Default is pointtype=2, corresponding to
        free points.

    ncg: obj
        An autocnet.graph.network NetworkCandidateGraph instance representing the network
        to apply this function to

    use_cache : bool
        If False (default) this func opens a database session and writes points
        and measures directly to the respective tables. If True, this method writes
        messages to the point_insert (defined in ncg.config) redis queue for
        asynchronous (higher performance) inserts.

    ratio_size : float
            Used in calling the function distribute_points_in_geom to determine the
            minimum size the ratio can be to be considered a sliver and ignored.

    Returns
    -------
    points : list of Points
        The list of points seeded in the overlap

    See Also
    --------
    autocnet.io.db.model.Overlay: for associated properties of the Overlay object

    autocnet.cg.cg.distribute_points_in_geom: for the possible arguments to pass through using disribute_points_kwargs.

    autocnet.model.io.db.PointType: for the point type options.

    autocnet.graph.network.NetworkCandidateGraph: for associated properties and functionalities of the NetworkCandidateGraph class

    """
    t1 = time.time()
    if not isinstance(overlap, Overlay):
        with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
            overlap = session.query(Overlay).filter(Overlay.id == overlap).one()
            session.expunge_all()
    
    # Determine the point distribution in the overlap geom
    geom = overlap.geom
    candidate_points = compgeom.distribute_points_in_geom(geom, ratio_size=ratio_size, **distribute_points_kwargs, **kwargs)
    if not candidate_points.any():
        warnings.warn(f'Failed to distribute points in overlap {overlap.id}')
        return []
    log.info(f'Have {len(candidate_points)} potential points to place in overlap {overlap.id}.')
    
    # If an NCG is passed, instantiate a session off the NCG, else just pass the session through
    nodes = []
    with ncg.session_scope() if ncg is not None else nullcontext(session) as session:
        for id in overlap.intersections:
            try:
                res = session.query(Images).filter(Images.id == id).one()
            except:
                warnings.warn(f'Unable to instantiate image with id: {id}')
                continue
            nn = NetworkNode(node_id=id, 
                             image_path=res.path, 
                             cam_type=res.cam_type,
                             dem=res.dem,
                             dem_type=res.dem_type)
            nodes.append(nn)
    points_to_commit = []
    for valid in candidate_points:
        log.debug(f'Valid point: {valid}')
        lat = valid[1]
        lon = valid[0]

        # Find the intresting sampleline and what image it is in
        reference_index, interesting_sampline = interesting_func(nodes, lon, lat, size=interesting_func_kwargs['size'])
        if not interesting_sampline:
            continue

        log.info(f'Found an interesting feature in {nodes[reference_index]["image_path"]} at {interesting_sampline.x}, {interesting_sampline.y}.')
        # Get the updated X,Y,Z location of the point and reproject to get the updates lon, lat.
        # The io.db.Point class handles all xyz to lat/lon and ographic/ocentric conversions in it's 
        # adjusted property setter.
        reference_node = nodes[reference_index]
        x,y,z = reference_node.geodata.sensormodel.sampline2xyz(interesting_sampline.x, interesting_sampline.y)
        
        # Create the new point
        point = Points.create_point_with_reference_measure(shapely.geometry.Point(x, y, z), 
                                                            reference_node, 
                                                            interesting_sampline,
                                                            choosername=identifier,
                                                            point_type=point_type) 
        log.debug(f'Created point: {point}.')

        # Iterate through all other, non-reference images in the overlap and attempt to add a measure.
        point.add_measures_to_point([n for i, n in enumerate(nodes) if i != reference_index], 
                                    choosername=identifier)
        # Insert the point into the database asynchronously (via redis) or synchronously via the ncg
        if use_cache:
            msgs = json.dumps(point.to_dict(_hide=[]), cls=JsonEncoder)
            ncg.push_insertion_message(ncg.point_insert_queue,
                                    ncg.point_insert_counter,
                                    msgs)
        else:
            if len(point.measures) >= 2:
                points_to_commit.append(point)
    if points_to_commit:
        with ncg.session_scope() if ncg else nullcontext(session) as session:
            session.add_all(points_to_commit)
            session.commit()
    t2 = time.time()
    log.info(f'Placed {len(candidate_points)} in {t2-t1} seconds.')

def place_points_in_image(image,
                          identifier="autocnet",
                          size=71,
                          distribute_points_kwargs={},
                          ncg=None,
                          **kwargs):
    """
    Place points into an image and then attempt to place measures
    into all overlapping images. This function is funcitonally identical
    to place_point_in_overlap except it works on individual images.

    Parameters
    ----------
    image : obj
            An autocnet Images model object

    identifier: str
                The tag used to distiguish points laid down by this function.

    size : int
           The size of the window used to extractor features to find an
           interesting feature to which the point is shifted.

    distribute_points_kwargs: dict
                              kwargs to pass to autocnet.cg.cg.distribute_points_in_geom

    ncg: obj
         An autocnet.graph.network NetworkCandidateGraph instance representing the network
         to apply this function to

    See Also
    --------
    autocnet.cg.cg.distribute_points_in_geom: for the possible arguments to pass through using disribute_points_kwargs.
    autocnet.graph.network.NetworkCandidateGraph: for associated properties and functionalities of the NetworkCandidateGraph class
    """
    # Arg checking
    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    points = []

    # Logic
    geom = image.geom
    # Put down a grid of points over the image; the density is intentionally high
    valid = compgeom.distribute_points_in_geom(geom, **distribute_points_kwargs)
    log.info(f'Have {len(valid)} potential points to place.')
    for v in valid:
        lon = v[0]
        lat = v[1]
        point_geometry = f'SRID=104971;POINT({v[0]} {v[1]})'

        with ncg.session_scope() as session:
            intersecting_images = session.query(Images.id, Images.path).filter(Images.geom.ST_Intersects(point_geometry)).all()
            node_res= [i for i in intersecting_images]
            nodes = []

            for nid, image_path  in node_res:
                # Setup the node objects that are covered by the geom
                nn = NetworkNode(node_id=nid, image_path=image_path)
                nodes.append(nn)

        # Need to get the first node and then convert from lat/lon to image space
        node = nodes[0]
        try:
            sample, line = node.geodata.sensormodel.lonlat2sampline(lon, lat)
        except CalledProcessError as e:
            log.exception(f'{e.stderr}')
            continue

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        image_roi.clip()
        try:
            interesting = extract_most_interesting(image.clipped_array)
        except:
            continue

        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

        lon, lat = node.geodata.sensormodel.sampline2lonlat(newsample, newline)


        if geom.contains(shapely.geometry.Point(lon, lat)):
            x, y, z = node.geodata.sensormodel.sampline2xyz(newsample, newline)
        else:
            x, y, z = node.geodata.sensormodel.sampline2xyz(sample, line)

        point_geom = shapely.geometry.Point(x, y, z)

        # Insert a spatial query to find which overlap this is in.
        with ncg.session_scope() as session:
            oid = session.query(Overlay.id).filter(Overlay.geom.ST_Contains(point_geometry)).one()[0]

        point = Points(identifier=identifier,
                       overlapid=oid,
                       apriori=point_geom,
                       adjusted=point_geom,
                       pointtype=2, # Would be 3 or 4 for ground
                       cam_type=node.geodata.sensor_type)

        for node in nodes:
            sample, line = node.geodata.sensormodel.xyz2sampline(x,y,z)
            point.measures.append(Measures(sample=sample,
                                           line=line,
                                           apriorisample=sample,
                                           aprioriline=line,
                                           imageid=node['node_id'],
                                           serial=node.isis_serial,
                                           measuretype=3,
                                           choosername='place_points_in_image'))

        if len(point.measures) >= 2:
            points.append(point)
    log.info(f'Able to place {len(points)} points.')
    with ncg.session_scope() as session:
        Points.bulkadd(points, session)
