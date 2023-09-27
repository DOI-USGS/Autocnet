import json
import logging
import math

import numpy as np
import shapely

from autocnet.cg.cg import create_points_along_line
from autocnet.io.db.model import Images, Points, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.spatial import isis
from autocnet.spatial import sensor
from autocnet.transformation import roi
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.matcher.validation import is_valid_lroc_polar_image
import time

# Set up logging file
log = logging.getLogger(__name__)

def get_number_of_points(target_distance_km, latitude, radius):
    """
    Calculate the number of points needed to maintain a certain target distance (in km)
    around a specific latitude for a target planetary body. This assume that the body
    is spherical and not ellipsoidal.

    Parameters
    ___________
    target_distance_km : int
        The target distance to maintin between points

    latitude : int
        the latitude where points are needed
    
    radius : int
        radius of the planetary body
    
    Returns
    _______
    num_points : int
        Number of points needed to maintin target distance
    """

    # Convert to radians
    target_distance_rad = target_distance_km / radius

    # Calculate longitudal distance between two points
    longitudinal_distance = 2 * math.asin(math.sqrt(math.sin(target_distance_rad/2)**2 / (math.cos(math.radians(latitude))**2)))

    # Calculate the circumference of the planetary body at the given latitude
    body_circumference = 2 * math.pi * radius * math.cos(math.radians(latitude))

    # Calculate points needed
    num_points = int(body_circumference / longitudinal_distance) + 1

    return num_points

def winnow_points(polygon, list_of_points):
    """
    This removes any points in a list that arent within a given polygon

    Parameters
    ___________

    polygon : shapely.geom
        A shapely geometry object
    
    list_of_points : list
        list of point coordinates in the form
        [(x1,y1), (x2,y2), ..., (xn, yn)]
    
    Returns
    _______
    points : list
        list of point coordinates in the form
        [(x1,y1), (x2,y2), ..., (xn, yn)]
    """
    return [(point[0], point[1]) for point in list_of_points if polygon.contains(shapely.geometry.Point(point[0], point[1]))]

def find_points_in_centroids(radius, 
                             target_distance_in_km,
                             polygon=None,
                             min_lat=None,
                             max_lat=None,
                             longitude_min=-180,
                             longitude_max=180):
    """
    This finds points within a specified geometry.
    It finds those points be placing points in a centroid pattern
    around latitude lines (default longitude: -180, 180) where the center
    is the center of the min_lat/max_lat/min_lon/max_lon

    Parameters
    ___________
    radius : int
        radius of the planetary body
    
    target_distance_in_km : int
        The target distance to maintain between points

    polygon : shapely.geom
        A shapely geometry object
    
    min_lat : int
        The minimum latitude of the wanted area
        If polygon is passed, not necessary
    
    min_long : int
        The minimum longitude of the wanted area
        If polygon is passed, not necessary
    
    longitude_min : int
        The minimum longitde (ex. -180 or 0)

    longitude_max : int
        The maximum longitde (ex. 180 or 360)
    
    Returns
    _______
    points : list
        list of point coordinates in the form
        [(x1,y1), (x2,y2), ..., (xn, yn)]
    """
    # Set up processes
    if polygon:
        _, min_lat, _, max_lat = polygon.bounds
    latitude_intervals = np.arange(max_lat, min_lat, -0.005)
    points=[]

    # Itterate through each latitude, decrimenting at -0.005
    # TODO: decide if this decrement should be a user input
    for lat in latitude_intervals:
        # Calculate the number of points needed to space points along a latitude line every __km
        num_points = get_number_of_points(target_distance_in_km, lat, radius)
        p1 = (longitude_min, lat)
        p2 = (longitude_max, lat)
        # Create a list of points for latitude lines
        line_points = [(point[0], point[1]) for point in create_points_along_line(p1, p2, num_points)]
        # If a polygon was given limit points so that they are only within the polygon
        if polygon:
            line_points = winnow_points(polygon, line_points)
        if len(line_points)!=0:
            points.extend(line_points)
    return points

def find_intresting_point(nodes, lon, lat, size=71):
    """
    Find an intresting point close the given lon, lat given a list data structure that contains
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

    Returns
    _______
    reference_index : int
        An index that refers to image that was choosen to be used as the reference image.
        This is the image in which an interesting point was found.
        
    point : shapely.geometry.point
        The intresting point close to the given lat/lon

    """
    # Itterate through the images to find an interesting point
    for reference_index, node in enumerate(nodes):
        log.debug(f'Trying image: {node["image_path"].split("/")[-1]}')
        # reference_index is the index into the list of measures for the image that is not shifted and is set at the
        # reference against which all other images are registered.
        try_sample, try_line = isis.ground_to_image(node["image_path"], lon, lat)

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
        if not is_valid_lroc_polar_image(roi_array, include_var=True, include_mean=True, include_std=True):
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

def place_points_in_centroids(candidate_points,
                            identifier="place_points_in_centroid",
                            interesting_func=find_intresting_point,
                            interesting_func_kwargs={"size":71},
                            point_type=2,
                            ncg=None, 
                            use_cache=False, 
                            **kwargs):
    """
    Place points into an centroid geometry by back-projecting using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    candidate_points : list
        list of point coordinates in the form
        [(x1,y1), (x2,y2), ..., (xn, yn)]

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
    """
    t1 = time.time()
    for valid in candidate_points:
        add_point_to_network(valid,
                             identifier=identifier,
                             interesting_func=interesting_func,
                             interesting_func_kwargs=interesting_func_kwargs,
                             point_type=point_type,
                             ncg=ncg,
                             use_cache=use_cache)
    t2 = time.time()
    log.info(f'Placed {len(candidate_points)} in {t2-t1} seconds.')

def add_point_to_network(valid,
                        identifier="place_points_in_centroid",
                        interesting_func=find_intresting_point,
                        interesting_func_kwargs={"size":71},
                        point_type=2,
                        ncg=None, 
                        use_cache=False):
    """
    Place points into an centroid geometry by back-projecting using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    valid : list
        point coordinates in the form (x1,y1)

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
    """
    t1 = time.time()
    
    point = shapely.geometry.Point(valid[0], valid[1])

    # Extract the overlap ids for the point
    with ncg.session_scope() as session:
        overlap_ids = Images.get_images_intersecting_point(point, session)

    # Instantiate the nodes in the NCG. This is done because we assume that the ncg passed is empty
    # and part of a cluster submission.
    nodes = []
    with ncg.session_scope() as session:
        for id in overlap_ids:
            res = session.query(Images).filter(Images.id == id).one()
            nn = NetworkNode(node_id=id, image_path=res.path)
            nn.parent = ncg
            nodes.append(nn)

    # Extract an interesting point 
    log.info(f'Searching for an interesting point at {point.x}, {point.y} (lat,lon) in {len(nodes)} images.')
    reference_index, interesting_sampline = interesting_func(nodes, point.x, point.y, **interesting_func_kwargs)
    log.info(f'Found an interesting feature in {nodes[reference_index]["image_path"]} at {interesting_sampline.x}, {interesting_sampline.y}.')

    # Get the updated X,Y,Z location of the point and reproject to get the updates lon, lat.
    # The io.db.Point class handles all xyz to lat/lon and ographic/ocentric conversions in it's 
    # adjusted property setter.
    reference_node = nodes[reference_index]
    x,y,z = isis.linesamp2xyz(reference_node['image_path'], interesting_sampline.x, interesting_sampline.y)

    # Create the point object for insertion into the database
    point_geom = shapely.geometry.Point(x, y, z)
    point = Points.create_point_with_reference_measure(point_geom, 
                                                       reference_node, 
                                                       interesting_sampline,
                                                       choosername=identifier,
                                                       point_type=point_type) 
    log.debug(f'Created point: {point}.')

    # Remove the reference_indexed measure from the list of candidates.
    # It has been added by the create_point_with_reference_measure function.
    del nodes[reference_index]

    # Determine what sensor type to use
    current_sensor = sensor.create_sensor('isis')

    # Iterate through all other, non-reference images in the overlap and attempt to add a measure.
    point.add_measures_to_point(nodes, current_sensor, choosername=identifier)

    # Insert the point into the database asynchronously (via redis) or synchronously via the ncg
    if use_cache:
        msgs = json.dumps(point.to_dict(_hide=[]), cls=JsonEncoder)
        ncg.push_insertion_message(ncg.point_insert_queue,
                                   ncg.point_insert_counter,
                                   msgs)
    else:
        with ncg.session_scope() as session:
            if len(point.measures) >= 2:
                session.add(point)
    t2 = time.time()
    log.info(f'Total processing time was {t2-t1} seconds.')

    return