from plio.io.io_gdal import GeoDataset
import numpy as np
import os
import os.path
import pvl
import kalasiris

import pandas as pd
from plio.utils.utils import find_in_dict

import geopandas as gpd

from geoalchemy2 import functions

from math import floor

from shapely import wkt
from shapely.geometry import Point

from autocnet.transformation.affine import estimate_local_affine
from autocnet.matcher.subpixel import subpixel_template
from autocnet.matcher.naive_template import pattern_match

from autocnet.io.db.model import Images, Points, Measures
from autocnet.cg.cg import distribute_points_in_geom, xy_in_polygon
from autocnet.spatial import isis
from autocnet.spatial.surface import GdalDem, EllipsoidDem
from autocnet.transformation.spatial import reproject, oc2og
from autocnet.matcher.cpu_extractor import extract_most_interesting
from autocnet.transformation import roi
from autocnet.utils.utils import bytescale


import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)

def generate_ground_points(Session, 
                           ground_mosaic, 
                           nspts_func=lambda x: int(round(x,1)*1), 
                           ewpts_func=lambda x: int(round(x,1)*4), 
                           size=(100,100),
                           verbose=False):
    """

    Parameters
    ----------
    ground_db_config : dict
                       In the form: {'username':'somename',
                                     'password':'somepassword',
                                     'host':'somehost',
                                     'pgbouncer_port':6543,
                                     'name':'somename'}
    nspts_func : func
                       describes distribution of points along the north-south
                       edge of an overlap.

    ewpts_func : func
                       describes distribution of points along the east-west
                       edge of an overlap.

    size : tuple of int
                       (size_x, size_y) maximum distances on either access point
                       can move when attempting to find an interesting feature.

    verbose : boolean
              an indicator which determines if visual plots are output illustrating the movement
              of the orginally laid point to an interesting feature
    """

    if isinstance(ground_mosaic, str):
        ground_mosaic = GeoDataset(ground_mosaic)

    log.warning('There are no unit tests tracking the operation of this function, users should take care to verify the results of this function for each dataset it is applied to.')

    session = Session()
    fp_poly = wkt.loads(session.query(functions.ST_AsText(functions.ST_Union(Images.geom))).one()[0])
    session.close()

    coords = distribute_points_in_geom(fp_poly, nspts_func=nspts_func, ewpts_func=ewpts_func, method="new", Session=Session)
    coords = np.asarray(coords)

    old_coord_list = []
    coord_list = []
    lines = []
    samples = []
    newlines = []
    newsamples = []

    # throw out points not intersecting the ground reference images
    log.info(f'points to lay down: {len(coords)}')
    for i, coord in enumerate(coords):
        # res = ground_session.execute(formated_sql)
        p = Point(*coord)
        log.info(f'point {i}'),


        linessamples = isis.point_info(ground_mosaic.file_name, p.x, p.y, 'ground', allowoutside=True)
        if linessamples is None:
            log.warning('Unable to find point in ground image')
            continue

        line = linessamples.get('Line')
        sample = linessamples.get('Sample')

        if line <= 0 or sample <= 0:
            log.warning(f'Line or sample are outside of the base.')
            continue
        
        image = roi.Roi(ground_mosaic, sample, line, size_x=size[0], size_y=size[1])
        try:
            image.clip()
        except:
            continue
        image_roi = image.clipped_array

        interesting = extract_most_interesting(bytescale(image_roi),  extractor_parameters={}, extractor_method='vlfeat')
        # interesting = extract_most_interesting(bytescale(image_roi),  extractor_parameters={'nfeatures':500}, extractor_method='orb')

        if interesting is None:
            log.info('extract_most_interesting was not able to find an interesting feature. Reverting to original location.')
            newsample = sample
            newline = line
        else:
            # kps are in the image space with upper left origin, so convert to
            # center origin and then convert back into full image space
            left_x = floor(image.x) - image.size_x
            top_y = floor(image.y) - image.size_y
            newsample = left_x + interesting.x
            newline = top_y + interesting.y

        if verbose:
            fig, axes = plt.subplots(1,2,figsize=(12,10))
            axes[0].imshow(image_roi, cmap='Greys')
            axes[0].plot(image.clip_center[0], image.clip_center[1], 'bo', label='Original Center')
            axes[0].plot(interesting.x, interesting.y, 'ro', label='Interesting Feature')
            axes[0].legend()

            axes[1].imshow(image_roi, cmap='Greys')
            axes[1].plot(interesting.x, interesting.y, 'ro', label='Interesting Feature')
            axes[1].set_xlim([interesting.x-10, interesting.x+11])
            axes[1].set_ylim([interesting.y-10, interesting.y+11])
            axes[1].legend()

        # return image, interesting

        newpoint = isis.point_info(ground_mosaic.file_name, newsample, newline, 'image')
        newp = Point(newpoint.get('PositiveEast360Longitude'),
                    newpoint.get('PlanetocentricLatitude'))

        if not (xy_in_polygon(p.x, p.y, fp_poly)):
                log.warning('Interesting point not in mosaic area, ignore')
                continue

        old_coord_list.append(p)
        lines.append(line)
        samples.append(sample)
        coord_list.append(newp)
        newlines.append(newline)
        newsamples.append(newsample)


    # start building the cnet
    ground_cnet = pd.DataFrame()
    ground_cnet["path"] = [ground_mosaic.file_name]*len(coord_list)
    ground_cnet["pointid"] = list(range(1,len(coord_list)+1))
    ground_cnet["original point"] = old_coord_list
    ground_cnet["point"] = coord_list
    ground_cnet['original_line'] = lines
    ground_cnet['original_sample'] = samples
    ground_cnet['line'] = newlines
    ground_cnet['sample'] = newsamples
    ground_cnet = gpd.GeoDataFrame(ground_cnet, geometry='point')
    return ground_cnet, fp_poly, coord_list


def propagate_point(lon,
                    lat,
                    pointid,
                    path,
                    line,
                    sample,
                    ncg,
                    dem,
                    match_kwargs={'image_size': (39, 39), 'template_size': (21, 21)},
                    verbose=False,
                    threshold=None):
    """
    Propagate a point into a stack of images. First, the ground location (lon, lat) will be projected
    into all overlapping images. Subpixel image registration (using autocnet.matcher.subpixel.subpixel_template)
    will be run between the image described by 'path' and all images existing in the database. Then the 
    best (as described by the highest correlation from subpixel_template) will become the reference measure 
    for the resulting point. 
    
    The other measures will need to be registered to the reference measure to fully control the point. See 
    autoncet.matcher.subpixel.smart_register_point

    Parameters
    ----------
    Session : sqlalchemy.sessionmaker
              session maker associated with the database you want to propagate to

    config : dict
             configuration file associated with database you want to propagate to
             In the form: {'username':'somename', 'password':'somepassword',
             'host':'somehost', 'pgbouncer_port':6543, 'name':'somename'}

    dem : surface
          surface model of target body

    lon : np.float
          longitude of point you want to project

    lat : np.float
          planetocentric latitude of point you want to project

    pointid : int
             clerical input used to trace point from generate_ground_points output

    path : list of str
            absolute paths pointing to the image(s) from which you want to try propagating the point 
    
    line : list of np.float
            a priori line(s) corresponding to point projected in 'paths' image(s 

    sample : list of np.float
              a priori sample(s) corresponding to point projected in 'paths' image(s 

    match_kwargs : dict
                  must contain 'image_size' and 'template_size' keyword. These parameters determine
                  the half size of the autocnet.transformation.roi's passed through to matcher.subpixel_template(). 
                  Note that 'image_size' must be larger than 'template_size'.

    verbose : boolean
              an indicator which determines if visual plots are output illustrating the matcher areas 
              and their correlation map for each point propagation.

    threshold : float
                when performing a subpixel_template match between the ground source and the relative images
                this is an optional value dictating the lowest possible template metric that a point can be project


    Returns
    -------
    new_measures : pd.DataFrame
                   Dataframe containing pointid, imageid, image serial number, line, sample, and ground location (both latlon
                   and cartesian) of successfully propagated points

    See Also
    --------
    autocent.spatial.surface: for description of EllipsoidDem and GdalDem classes
    """
    
    # check we have all of the inputs we need
    if isinstance(dem, str):
        try:
            dem = GdalDem(dem)
        except Exception as e:
            log.exception(f'Unable to create dem of type autocnet.spatial.surface.GdalDem from {dem} with exception {e}')

    with ncg.session_scope() as session:
        string = f"select * from images where ST_Intersects(geom, ST_SetSRID(ST_Point({lon}, {lat}), {ncg.config['spatial']['latitudinal_srid']}))"
        images = pd.read_sql(string, ncg.engine)

    new_measures = []
    match_results = []
    # lazily iterate for now

    base_image = GeoDataset(path)
    base_sample, base_line = sample, line

    for i,image in images.iterrows():
        dest_image = GeoDataset(image["path"])
        moving_sample, moving_line = isis.ground_to_image(dest_image.file_name, lon, lat)

        if os.path.basename(path) == os.path.basename(image['path']):
            continue

        # Calculate a buffer value that will provide enough data to extract meaningful data
        # after the affine (including a large scaling aspect) is applied
        
        dest_res = find_in_dict(dest_image.metadata, 'PixelResolution')
        if dest_res is None:
            try:
                # use camera
                dest_campt = pvl.loads(kalasiris.campt(dest_image.file_name).stdout)["GroundPoint"]
                dest_res = dest_campt["LineResolution"]
            except:
                log.error(f'{dest_image} does not have associated camera or mapping information. Please attach spice kernels or map project.')
        
        base_res = find_in_dict(base_image.metadata, 'PixelResolution')
        if base_res is None:
            try:
                base_campt = pvl.loads(kalasiris.campt(base_image.file_name).stdout)["GroundPoint"]
                base_res = base_campt["LineResolution"]
            except:
                log.error(f'{base_image} does not have associated camera or mapping information. Please attach spice kernels or map project.')
        
        if hasattr(base_res, 'value'):
            base_res = base_res.value
        if hasattr(dest_res, 'value'):
            dest_res = dest_res.value

        n_double = np.ceil(np.log2(base_res/dest_res))
        dest_buffer = int(match_kwargs['template_size'][0]*2**n_double)

        log.info(f'prop point: base_image: {base_image}')
        log.info(f'prop point: dest_image: {dest_image}')
        log.info(f'prop point: base point (sample, line): ({base_sample}, {base_line})')
        log.info(f'prop point: dest point (sample, line): ({moving_sample}, {moving_line})')

        base_roi = roi.Roi(base_image,
                            base_sample,
                            base_line,
                            size_x=match_kwargs['image_size'][0],
                            size_y=match_kwargs['image_size'][1], 
                            buffer=0)
        moving_roi = roi.Roi(dest_image,
                            moving_sample,
                            moving_line,
                            size_x=match_kwargs['template_size'][0],
                            size_y=match_kwargs['template_size'][1], 
                            buffer=dest_buffer)
        try:
            baseline_affine = estimate_local_affine(base_roi, moving_roi)
        except Exception as e:
            log.error('Unable to transform image to reference space. Likely too close to the edge of the non-reference image. Setting ignore=True')
            match_results.append(e)
            continue

        try:
            updated_affine, metrics, corrmap = subpixel_template(base_roi, 
                                                    moving_roi, 
                                                    affine=baseline_affine,
                                                    func=pattern_match)
            if verbose:
                fig, axes = plt.subplots(1,3,figsize=(15,10));

                base_roi.clip()
                axes[0].imshow(base_roi.clipped_array, cmap='Greys');
                axes[0].scatter(base_roi.clip_center[0], base_roi.clip_center[1], c='b');
                axes[0].set_title(f'Base Image at ({base_line},{base_sample})');
                
                moving_roi.clip()
                axes[1].imshow(moving_roi.clipped_array, cmap='Greys');
                axes[1].scatter(moving_roi.clip_center[0], moving_roi.clip_center[1], c='r', label='original point');
                axes[1].set_title(f'Moving Image');

                moving_roi.clip(affine=baseline_affine)
                axes[2].imshow(moving_roi.clipped_array, cmap='Greys');
                axes[2].scatter(moving_roi.clip_center[0], moving_roi.clip_center[1], c='r', label='original point');
                axes[2].set_title(f'Affine Transformed Moving Image');

                new_center_x, new_center_y = updated_affine([moving_roi.clip_center[0], moving_roi.clip_center[1]])[0]
                print(new_center_x, new_center_y)
                axes[2].scatter(new_center_x, new_center_y, c='b', label='registered point');
                axes[2].legend();
                plt.show();

        except Exception as e:
            log.error(f'Unable to subpixel register with exception {e}.')
            match_results.append(e)
            continue

        
        x, y = updated_affine([moving_sample, moving_line])[0]
        dist = np.linalg.norm(list(updated_affine.translation))            


        # except Exception as e:
        #     # raise Exception(e)
        #     match_results.append(e)
        #     continue

        match_results.append([x, y,
                             metrics, dist, corrmap, path, image["path"],
                             image['id'], image['serial']])

    # remove results that contain exceptions or all None entries
    match_results = np.asarray([res for res in match_results if isinstance(res, list) and all(r is not None for r in res)], dtype=object)
    
    if len(match_results) == 0:
        log.warning(f'Point propagation failed, no match results.')
        return new_measures

    # find the highest subpixel correlation 
    metrics = list(match_results[:,2])
    best_result = match_results[np.argmax(metrics)]

    if threshold is not None:
        if best_result[2] < threshold:
            log.warning(f'Point propagation failed, no match satisfied threshold.')
            return new_measures

    log.debug(f'match_results final length: {len(match_results)}')
    log.debug(f'Full results: {best_result}')
    log.debug(f'Full set of matching correlations: {metrics}')
    log.debug(f'Winning CORR: {best_result[2]}, Pixel shift (in moving space): {best_result[3]}')
    log.debug(f'Base (Ground) Image: {best_result[5]}, Relative image: {best_result[6]}')
    log.debug(f'Base (Ground) Sample: {base_sample},  Relative Sample: {best_result[0]}')
    log.debug(f'Base (Ground) Line: {base_line},  Relative Line: {best_result[1]}')

    try:
        height = dem.get_height(lat, lon)
    except Exception as e:
        log.warning(f'Could not generate height from DEM')

    semi_major = ncg.config['spatial']['semimajor_rad']
    semi_minor = ncg.config['spatial']['semiminor_rad']
    # The CSM conversion makes the LLA/ECEF conversion explicit
    # reprojection takes ographic lat (gdal assumption) to geocentric lat (ISIS assumption)
    lon_og, lat_og = oc2og(lon, lat, semi_major, semi_minor)
    x, y, z = reproject([lon_og, lat_og, height],
                         semi_major, semi_minor,
                         'latlon', 'geocent')

    best_path = best_result[6]
    best_sample = best_result[0]
    best_line = best_result[1]
    
    new_measures.append({
        'pointid' : pointid,
        'imageid' : best_result[7],
        'serial' : best_result[8],
        'path': best_path,
        'line' : best_line,
        'sample' : best_sample,
        'template_metric' : best_result[2],
        'template_shift' : best_result[3],
        'point' : Point(lon, lat),
        'point_ecef' : Point(x, y, z)
        })

    # add the other relative images in the stack.
    # It is important to go from best_result (sample, line) to ground (lon, lat) 
    # and then pierce the other relative images instead of going directly from 
    # the ground (lon, lat) assosciated with the best_result. This is because
    # the (lon, lat) associated with the best_result is the ground location 
    # of the feature in the image as defined by the ground source file. This 
    # is not necessarily equivalent to the ground location of the feature in 
    # the image as defined by the relative image's camera.
    ls_lon, ls_lat = isis.image_to_ground(best_path, best_sample, best_line)
     
    for i,image in images.iterrows():
        if image["path"] == best_path:
            continue

        dest_image = GeoDataset(image["path"])
        apriori_sample, apriori_line = isis.ground_to_image(dest_image.file_name, ls_lon, ls_lat)

        if apriori_sample is None or apriori_line is None:
            log.info(f'Registered location in reference (image {best_result[7]}) does not project into image {image["id"]}')
            continue 

        new_measures.append({
            'pointid' : pointid,
            'imageid' : image["id"],
            'serial' : image["serial"],
            'path': image["path"],
            'line' : apriori_line,
            'sample' : apriori_sample,
            'point' : Point(lon, lat),
            'point_ecef' : Point(x, y, z)
        })

    return new_measures


def propagate_ground_points(ground_df,
                              ncg,
                              dem_path,
                              match_kwargs={'image_size': (39,39), 'template_size': (21,21)},
                              threshold=None,
                              verbose=False):
    """
    Loops over a base control network's measure information (line, sample, image path) and uses image matching
    algorithms (autocnet.matcher.subpixel.WHATEVAAAAA) to find the corresponding line(s)/sample(s) in database images.

    Parameters
    ----------
    base_cnet : pd.DataFrame
                Dataframe representing the points you want to propagate. Must contain the location of the point 
                (in the form Point(lon, lat)) contained in a 'point' column, the  'line', 'sample' location of
                the measure, and the 'path' to the corresponding image  

    ncg : NetworkCandidateGraph instance
          the NetworkCandidateGraph object associated with the network you want to which you want to propagate the points
          of you base network

    dem_path : str
               file path pointing to the surface model of the target body 

    match_kwargs : dict
                  Must contain 'image_size' and 'template_size' keyword. These parameters determine
                  the half size of the autocnet.transformation.roi's passed through to matcher.subpixel_template(). 
                  Note that 'image_size' must be larger than 'template_size'.

    cost : anonymous boolean function
           determines to which image(s) the point should be propagated. x corresponds to a list
           of all match correlation metrics, while y corresponds to each indiviudal element
           of the x array.
           Example:
           cost = lambda x,y: y == np.max(x) will get you one result corresponding to the image that
           has the maximum correlation with the source image
           cost = lambda x,y: y > 0.6 will propegate the point to all images whose correlation
           result is greater than 0.6
    
    identifier : str
                The tag used to distinguish points laid down by this function.
    
    verbose : boolean
              an indicator which determines if visual plots are output illustrating the matcher areas 
              and their correlation map for each point propagation.


    Returns
    -------
    ground : pd.DataFrame
             Dataframe containing pointid, imageid, image serial number, line, sample, and ground location (both latlon
             and cartesian) of successfully propagated points

    """
    
    log.warning('There are no unit tests tracking the operation of this function, users should take care to verify the results of this function for each dataset it is applied to.')

    try:
        semi_major = ncg.config['spatial']['semimajor_rad']
        semi_minor = ncg.config['spatial']['semiminor_rad']
        dem_type='height'
        dem = GdalDem(ground_file, semi_major, semi_minor, dem_type='height')
    except Exception as e1:
        try:
            dem = EllipsoidDem(dem_path)
        except Exception as e2:
            log.error(f'Unable to create autocnet.spatial.surface.GdalDem object from {dem_path} with exception {e1}.')
            log.error(f'Unable to create autocnet.spatial.surface.EllipsoidDem object from {dem_path} with exception {e2}.')

    # groups = ground_df.groupby('pointid').groups

    # append CNET info into structured Python list
    constrained_net = []

    # easily parallelized on the cpoint level, dummy serial for now
    for i, row in ground_df.iterrows():

        # get image in the destination that overlap
        p = row["point"]
        lon, lat = p.xy

        gp_measures = propagate_point(lon[0],
                                      lat[0],
                                      row["pointid"],
                                      row["path"],
                                      row["line"],
                                      row["sample"],
                                      ncg,
                                      dem,
                                      match_kwargs,
                                      verbose=verbose,
                                      threshold=threshold)

        # do not append if propagate_point is unable to find result
        if len(gp_measures) == 0:
            continue
        constrained_net.extend(gp_measures)
    ground = gpd.GeoDataFrame.from_dict(constrained_net).set_geometry('point')
    groundpoints = ground.groupby('pointid').groups

    points = []

    # conditionally upload a new point to DB or updated existing point with new measures
    lat_srid = ncg.config['spatial']['latitudinal_srid']
    with ncg.session_scope() as session:
        for gp_point, indices in groundpoints.items():
            point = ground.loc[indices].iloc[0]

            # check DB for if point already exists there
            lon = point.point.x
            lat = point.point.y

            spatial_point = functions.ST_Point(lon, lat)
            spatial_setSRID = functions.ST_SetSRID(spatial_point, lat_srid)
            spatial_buffer = functions.ST_Buffer(spatial_setSRID, 10e-10)
            spatial_intersects = functions.ST_Intersects(Points.geom, spatial_buffer)

            res = session.query(Points).filter(spatial_intersects).all()

            if len(res) > 1:
                log.warning(f"There is more than one point at lon: {lon}, lat: {lat}")

            elif len(res) == 1:
                # update existing point with new measures
                for i in indices:
                    row = ground.loc[i]
                    pid = res[0].id
                    meas = session.query(Measures.serial).filter(Measures.pointid == pid).all()
                    serialnumbers = [m[0] for m in meas]

                    if row['serial'] in serialnumbers:
                        continue

                    points.append(Measures(pointid = pid,
                                        line = float(row['line']),
                                        sample = float(row['sample']),
                                        aprioriline = float(row['line']),
                                        apriorisample = float(row['sample']),
                                        imageid = int(row['imageid']),
                                        template_metric = float(row['template_metric']),
                                        template_shift = float(row['template_shift']),
                                        serial = row['serial'],
                                        measuretype = 3))
            else:
                # upload new point
                p = Points()
                p.pointtype = 3
                p.apriori = point['point_ecef']
                p.adjusted = point['point_ecef']
                for i in indices:
                    row = ground.loc[i]
                    p.measures.append(Measures(line = float(row['line']),
                                            sample = float(row['sample']),
                                            aprioriline = float(row['line']),
                                            apriorisample = float(row['sample']),
                                            imageid = int(row['imageid']),
                                            serial = row['serial'],
                                            template_metric = float(row['template_metric']),
                                            template_shift = float(row['template_shift']),
                                            measuretype = 3))
                points.append(p)

        session.add_all(points)
        session.commit()
    

    return ground


