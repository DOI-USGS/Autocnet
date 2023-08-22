import numpy as np
import shapely
import math
import csmapi
import os
import logging
from sqlalchemy import text
from autocnet.cg.cg import create_points_along_line
from autocnet.io.db.model import Images, Measures, Overlay, Points, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.spatial import isis
from autocnet.transformation import roi
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation.spatial import reproject, og2oc, oc2og
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

def place_points_in_centroids(valid,
                            identifier="autocnet",
                            cam_type="csm", 
                            size=71,
                            point_type=2,
                            ncg=None, 
                            use_cache=False, 
                            **kwargs):
    """
    Place points into an centroid geometry by back-projecting using sensor models.
    The DEM specified in the config file will be used to calculate point elevations.

    Parameters
    ----------
    valid : list
        list of point coordinates in the form
        [(x1,y1), (x2,y2), ..., (xn, yn)]

    identifier: str
        The tag used to distinguish points laid down by this function.

    cam_type : str
        options: {"csm", "isis"}
        Pick what kind of camera model implementation to use.

    size : int
        The amount of pixel around a points initial location to search for an
        interesting feature to which to shift the point.

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
    points = []
    
    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']

    v=shapely.geometry.Point(valid[0], valid[1])
    # Get all the images that the point intersects
    sql_query=text(f"SELECT id FROM images WHERE ST_Intersects('SRID=30100;{v}', geom)")
    with ncg.session_scope() as session:
        res = session.execute(sql_query).all()
    overlap_ids = [id[0] for id in res]

    # Make the nodes
    nodes = []
    with ncg.session_scope() as session:
        for id in overlap_ids:
            res = session.query(Images).filter(Images.id == id).one()
            nn = NetworkNode(node_id=id, image_path=res.path)
            nn.parent = ncg
            nodes.append(nn)

    lon = v.x
    lat = v.y
    log.info(f'Point: {lon, lat}')

    # Need to get the first node and then convert from lat/lon to image space
    for reference_index, node in enumerate(nodes):
        log.debug(f'Starting with reference_index: {reference_index}')
        # reference_index is the index into the list of measures for the image that is not shifted and is set at the
        # reference against which all other images are registered.
        if cam_type == "isis":
            sample, line = isis.ground_to_image(node["image_path"], lon, lat)
            if sample == None or line == None:
                log.info(f'point ({lon}, {lat}) does not project to reference image {node["image_path"]}')
                continue
        if cam_type == "csm":
            lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
            x, y, z = reproject([lon_og, lat_og, height],
                                semi_major, semi_minor,
                                'latlon', 'geocent')
            # The CSM conversion makes the LLA/ECEF conversion explicit
            gnd = csmapi.EcefCoord(x, y, z)
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line

        # Extract ORB features in a sub-image around the desired point
        image_roi = roi.Roi(node.geodata, sample, line, size_x=size, size_y=size)
        try:
            if image_roi.variance == 0:
                log.info(f'Failed to find interesting features in image.')
                continue
        except:
            log.info(f'Failed to find interesting features in image.')
            continue
        # Extract the most interesting feature in the search window
        interesting = extract_most_interesting(image_roi.clipped_array)
        if interesting is not None:
            # We have found an interesting feature and have identified the reference point.
            break
    log.debug(f'Current reference index: {reference_index}.')

    if interesting is None:
        log.info('Unable to find an interesting point, falling back to the a priori pointing')
        newsample = sample
        newline = line
    else:
        # kps are in the image space with upper left origin and the roi
        # could be the requested size or smaller if near an image boundary.
        # So use the roi upper left_x and top_y for the actual origin.
        left_x, _, top_y, _ = image_roi.image_extent
        newsample = left_x + interesting.x
        newline = top_y + interesting.y

    # Get the updated lat/lon from the feature in the node
    if cam_type == "isis":
        try:
            p = isis.point_info(node["image_path"], newsample, newline, point_type="image")
        except CalledProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                log.debug(node["image_path"])
                log.info(f'interesting point ({newsample}, {newline}) does not project back to ground')
        try:
            x, y, z = p["BodyFixedCoordinate"].value
        except:
            x, y, z = ["BodyFixedCoordinate"]

        if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
            x = x * 1000
            y = y * 1000
            z = z * 1000
    elif cam_type == "csm":
        image_coord = csmapi.ImageCoord(newline, newsample)
        pcoord = node.camera.imageToGround(image_coord)
        # Get the BCEF coordinate from the lon, lat
        updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                    semi_major, semi_minor, 'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        updated_height = ncg.dem.get_height(updated_lat, updated_lon)


        # Get the BCEF coordinate from the lon, lat
        x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                            semi_major, semi_minor, 'latlon', 'geocent')


    updated_lon_og, updated_lat_og, updated_height = reproject([x, y, z], semi_major, semi_minor,
                                                        'geocent', 'latlon')
    updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

    # Make the point
    point_geom = shapely.geometry.Point(x, y, z)
    log.debug(f'Creating point with reference_index: {reference_index}')
    point = Points(identifier=identifier,
                apriori=point_geom,
                adjusted=point_geom,
                pointtype=point_type, # Would be 3 or 4 for ground
                cam_type=cam_type,
                reference_index=reference_index)

    # Compute ground point to back project into measurtes
    gnd = csmapi.EcefCoord(x, y, z)

    for current_index, node in enumerate(nodes):
        if cam_type == "csm":
            image_coord = node.camera.groundToImage(gnd)
            sample, line = image_coord.samp, image_coord.line
        if cam_type == "isis":
            # If this try/except fails, then the reference_index could be wrong because the length
            # of the measures list is different than the length of the nodes list that was used
            # to find the most interesting feature.
            if not os.path.exists(node["image_path"]):
                log.info(f'Unable to find input image {node["image_path"]}')
                continue
            try:
                sample, line = isis.ground_to_image(node["image_path"], updated_lon, updated_lat)
            except:
                log.info(f"{node['image_path']} failed ground_to_image. Likely due to being processed incorrectly or is just a bad image that failed campt.")
            if sample == None or line == None:
            #except CalledProcessError as e:
            #except:  # CalledProcessError is not catching the ValueError that this try/except is attempting to handle.
                log.info(f'interesting point ({updated_lon},{updated_lat}) does not project to image {node["image_path"]}')
                # If the current_index is greater than the reference_index, the change in list size does
                # not impact the positional index of the reference. If current_index is less than the
                # reference_index, then the reference_index needs to de-increment by one for each time
                # a measure fails to be placed.
                if current_index < reference_index:
                    reference_index -= 1
                    log.debug('Reference de-incremented.')
                continue
        if node.isis_serial:
            point.measures.append(Measures(sample=sample,
                                        line=line,
                                        apriorisample=sample,
                                        aprioriline=line,
                                        imageid=node['node_id'],
                                        serial=node.isis_serial,
                                        measuretype=3,
                                        choosername='place_points_in_centroid'))
        else:
            log.info(f"{node['node_id']} serial number is NULL")
    log.debug(f'Current reference index in code: {reference_index}.')
    log.debug(f'Current reference index on point: {point.reference_index}')
    if len(point.measures) >= 2:
        points.append(point)
    log.info(f'Able to place {len(points)} points.')

    if not points: return

    # Insert the points into the database asynchronously (via redis) or synchronously via the ncg
    if use_cache:
        pipeline = ncg.redis_queue.pipeline()
        msgs = [json.dumps(point.to_dict(_hide=[]), cls=JsonEncoder) for point in points]
        pipeline.rpush(ncg.point_insert_queue, *msgs)
        pipeline.execute()
        # Push
        log.info('Using the cache')
        ncg.redis_queue.incr(ncg.point_insert_counter, amount=len(points))
    else:
        with ncg.session_scope() as session:
            for point in points:
                session.add(point)
    t2 = time.time()
    log.info(f'Total processing time was {t2-t1} seconds.')

    return