from collections import defaultdict
import json
import logging
from math import floor
import numbers
import sys
import time
import warnings
import logging

import numpy as np

from skimage import transform as tf
from skimage import registration
from skimage import filters
from scipy import fftpack
from scipy.spatial import distance_matrix
import numpy as np
from sqlalchemy.sql.expression import bindparam

from matplotlib import pyplot as plt

from plio.io.io_gdal import GeoDataset

import pvl

import PIL
from PIL import Image

from autocnet.matcher.naive_template import pattern_match
from autocnet.matcher.mutual_information import mutual_information_match
from autocnet.spatial import isis
from autocnet.io.db.model import Measures, Points, Images, JsonEncoder
from autocnet.graph.node import NetworkNode
from autocnet.transformation import roi
from autocnet.transformation.affine import estimate_local_affine
from autocnet import spatial
from autocnet.utils.serializers import JsonEncoder

from sqlalchemy import inspect

PIL.Image.MAX_IMAGE_PIXELS = sys.float_info.max
log = logging.getLogger(__name__)


def check_image_size(imagesize):
    """
    Given an x,y tuple, ensure that the values
    are odd. Used by the subpixel template to also ensure
    that the template size is the one requested and not 2x
    the template size.

    Parameters
    ----------
    imagesize : tuple
                in the form (size_x, size_y)
    """
    if isinstance(imagesize, numbers.Number):
        imagesize = (int(imagesize), int(imagesize))


    x = imagesize[0]
    y = imagesize[1]

    if x % 2 == 0:
        x += 1
    if y % 2 == 0:
        y += 1
    x = floor(x/2)
    y = floor(y/2)
    return x,y

def subpixel_phase(reference_roi, moving_roi, affine=tf.AffineTransform(), **kwargs):
    """
    Apply the spectral domain matcher to a search and template image. To
    shift the images, the x_shift and y_shift, need to be subtracted from
    the center of the search image. It may also be necessary to apply the
    fractional pixel adjustment as well (if for example the center of the
    search is not an integer); this function do not manage shifting.

    Parameters
    ----------
    reference_roi : Object
                    An Roi object from autocnet, the reference image to use when computing subpixel offsets.
    moving_roi : Object
                  An Roi object from autocnet, the walking image to move around and make comparisons to
                  the reference roi.
    affine : Object
             Scikit image affine tranformation. This affine transformation is applied to the moving_roi
             before any comparisons are made

    Returns
    -------
    new_affine : Object
                 Scikit image affine transformation. An updated affine transformation that is the new
                 translation from the moving_roi to the reference_roi
    : tuple
      With the RMSE error and absolute difference in phase
    """
    reference_image = reference_roi.clip()
    walking_template = moving_roi.clip(affine)
    
    if reference_image.shape != walking_template.shape:
        reference_size = reference_image.shape
        walking_size = walking_template.shape
        updated_size_x = int(min(walking_size[1], reference_size[1]))
        updated_size_y = int(min(walking_size[0], reference_size[0]))

        # Have to subtract 1 from even entries or else the round up that
        # occurs when the size is split over the midpoint causes the
        # size to be too large by 1.
        if updated_size_x % 2 == 0:
            updated_size_x -= 1
        if updated_size_y % 2 == 0:
            updated_size_y -= 1

        # Since the image is smaller than the requested size, set the size to
        # the current maximum image size and reduce from there on potential
        # future iterations.
        size = check_image_size((updated_size_x, updated_size_y))

        reference_roi.size_x, reference_roi.size_y = size
        moving_roi.size_x, moving_roi.size_y = size

        reference_image = reference_roi.clip()
        walking_template = moving_roi.clip(affine)
    (shift_y, shift_x), error, diffphase = registration.phase_cross_correlation(reference_image,
                                                                                walking_template,
                                                                                **kwargs)
    # get shifts in input pixel space 
    shift_x, shift_y = affine([shift_x, shift_y])[0]
    new_affine = tf.AffineTransform(translation=(-shift_x, -shift_y))
    return new_affine, error, diffphase

def subpixel_template(reference_roi, 
                      moving_roi, 
                      affine=tf.AffineTransform(), 
                      func=pattern_match,
                      **kwargs):
    """
    Uses a pattern-matcher on subsets of two images determined from the passed-in keypoints and optional sizes to
    compute an x and y offset from the search keypoint to the template keypoint and an associated strength.
    Parameters
    ----------
    reference_roi : autocnet.roi.Roi
                    Roi object representing the reference image 
    
    moving_roi : autocnet.roi.Roi
                 Roi object representing the moving image, this image is registered to the reference_roi   
    
    affine : skimage.transform.AffineTransform
             scikit-image Affine transformation, used as a seed transform that 

    func : callable 
           Some subpixel template matching function. Default autocnet.matcher.naive_template.pattern_match 
    
    kwargs : dict 
             keyword args for func 

    Returns
    -------
    affine : skimage.transform.AffineTransform
             Affine transform containing new translations to be applied to the x and y attributes of 
             the input moving_roi. For example, returned_affine((moving_roi.x, moving_roi.y))
    
    strength : float
               Strength of the correspondence in the range [-1, 1]
    
    corrmap : np.array 
              containing correlation coefficients computed at each tested location in the reference_roi
    
    See Also
    --------
    autocnet.matcher.naive_template.pattern_match : for the kwargs that can be passed to the matcher
    autocnet.matcher.naive_template.pattern_match_autoreg : for the kwargs that can be passed to the autoreg style matcher
    """

    try:
        s_image_dtype = isis.isis2np_types[pvl.load(reference_roi.data.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        s_image_dtype = None

    try:
        d_template_dtype = isis.isis2np_types[pvl.load(moving_roi.data.file_name)["IsisCube"]["Core"]["Pixels"]["Type"]]
    except:
        d_template_dtype = None
    
    
    # In ISIS, the reference image is the search and moving image is the pattern.
    reference_roi.clip(dtype = s_image_dtype)
    ref_clip = reference_roi.clipped_array
    moving_roi.clip(affine=affine, dtype=d_template_dtype)
    moving_clip = moving_roi.clipped_array
    
    if moving_clip.var() == 0:
        warnings.warn('Input ROI has no variance.')
        return [None] * 3

    if (ref_clip is None) or (moving_clip is None):
        return None, None, None

    matcher_shift_x, matcher_shift_y, metrics, corrmap = func(moving_clip, ref_clip, upsampling=16, **kwargs)
    if matcher_shift_x is None:
        return None, None, None

    # shift_x, shift_y are in the affine transformed space. They are relative to center of the affine transformed ROI.
    # Apply the shift to the center of the moving roi to the center of the reference ROI in index space.
    new_center_x = moving_roi.clip_center[0] + matcher_shift_x
    new_center_y = moving_roi.clip_center[1] + matcher_shift_y
    
    new_x, new_y = moving_roi.clip_coordinate_to_image_coordinate(new_center_x, new_center_y)
    
    new_affine = tf.AffineTransform(translation=(-(moving_roi.x - new_x),
                                                 -(moving_roi.y - new_y)))
    
    return new_affine, metrics, corrmap


def iterative_phase(reference_roi, moving_roi, affine=tf.AffineTransform(), reduction=11, convergence_threshold=0.1, max_dist=50, **kwargs):
    """
    Iteratively apply a subpixel phase matcher to source (s_img) and destination (d_img)
    images. The size parameter is used to set the initial search space. The algorithm
    is recursively applied to reduce the total search space by reduction until the convergence criteria
    are met. Convergence is defined as the point at which the computed shifts (x_shift,y_shift) are
    less than the convergence_threshold. In instances where the size is reducted to 1 pixel the
    algorithm terminates and returns None.

    Parameters
    ----------
    reference_roi : Object
                    An Roi object from autocnet, the reference image to use when computing subpixel offsets.
                    Contains either an ndarray or a GeoDataset Object
    moving_roi : Object
                  An Roi object from autocnet, the walking image to move around and make comparisons to
                  the reference roi. Contains either an ndarray or a GeoDataset Object
    size : tuple
           Size of the template in the form (x,y)
    reduction : int
                With each recursive call to this func, the size is reduced by this amount
    convergence_threshold : float
                            The value under which the result can shift in the x and y directions to force a break

    Returns
    -------
    metrics : tuple
              A tuple of metrics. In the case of the phase matcher this are difference
              and RMSE in the phase dimension.

    See Also
    --------
    subpixel_phase : the function that applies a single iteration of the phase matcher
    """
    # get initial destination location
    dsample, dline = moving_roi.x, moving_roi.y
    dx, dy = moving_roi.x, moving_roi.y

    size = (moving_roi.size_x, moving_roi.size_y)
    original_size = (moving_roi.size_x, moving_roi.size_y)

    subpixel_affine = tf.AffineTransform()

    while True:
        new_subpixel_affine, error, diffphase = subpixel_phase(reference_roi, moving_roi, affine, **kwargs)
        # Compute the amount of move the matcher introduced
        delta_dx, delta_dy = abs(new_subpixel_affine.translation)
        subpixel_affine += new_subpixel_affine
        dx, dy = subpixel_affine.inverse((reference_roi.x, reference_roi.y))[0]
        moving_roi.x, moving_roi.y = dx, dy

        # Break if the solution has converged
        size = (size[0] - reduction, size[1] - reduction)
        moving_roi.size_x, moving_roi.size_y = size
        reference_roi.size_x, reference_roi.size_y = size
        dist = np.linalg.norm([dsample-dx, dline-dy])

        if min(size) < 1:
            return None, None, None
        if delta_dx <= convergence_threshold and\
           delta_dy <= convergence_threshold and\
           abs(dist) <= max_dist:
            break

    moving_roi.size_x, moving_roi.size_y = original_size
    reference_roi.size_x, reference_roi.size_y = original_size
    moving_roi.x, moving_roi.y = dsample, dline
    return subpixel_affine, error, diffphase

def subpixel_register_point(pointid,
                            cost_func=lambda x,y: 1/x**2 * y,
                            threshold=0.005,
                            ncg=None,
                            match_func=subpixel_template,
                            match_kwargs={},
                            use_cache=False,
                            verbose=False,
                            chooser='subpixel_register_point',
                            **kwargs):

    """
    Given some point, subpixel register all of the measures in the point to the
    first measure.

    Parameters
    ----------
    pointid : int or obj
              The identifier of the point in the DB or a Points object

    cost_func : func
                A generic cost function accepting two arguments (x,y), where x is the
                distance that a point has shifted from the original, sensor identified
                intersection, and y is the correlation coefficient coming out of the
                template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    geom_func : callable
                function used to tranform the source and/or destination image before
                running a matcher.

    match_func : callable
                 subpixel matching function to use registering measures

    use_cache : bool
                If False (default) this func opens a database session and writes points
                and measures directly to the respective tables. If True, this method writes
                messages to the point_insert (defined in ncg.config) redis queue for
                asynchronous (higher performance) inserts.
    """

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if isinstance(pointid, Points):
        pointid = pointid.id

    t1 = time.time()
    with ncg.session_scope() as session:
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        point = session.query(Points).filter(Points.id == pointid).one()
        reference_index = point.reference_index
        t2 = time.time()
        log.info(f'Query took {t2-t1} seconds to find the measures and reference measure.')
        # Get the reference measure. Previously this was index 0, but now it is a database tracked attribute
        source = measures[reference_index]

        log.info(f'Using measure {source.id} on image {source.imageid}/{source.serial} as the reference.')
        log.info(f'Measure reference index is: {reference_index}')
        source.template_metric = 1
        source.template_shift = 0
        source.phase_error = 0
        source.phase_diff = 0
        source.phase_shift = 0

        sourceid = source.imageid
        sourceres = session.query(Images).filter(Images.id == sourceid).one()
        source_node = NetworkNode(node_id=sourceid, image_path=sourceres.path)
        source_node.parent = ncg
        t3 = time.time()
        log.info(f'Query for the image to use as source took {t3-t2} seconds.')
        log.info(f'Attempting to subpixel register {len(measures)-1} measures for point {pointid}')
        nodes = {}
        for measure in measures:
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            nodes[measure.imageid] = NetworkNode(node_id=measure.imageid, image_path=res.path)
        session.expunge_all()

    resultlog = []
    updated_measures = []
    for i, measure in enumerate(measures):
        if i == reference_index:
            continue

        currentlog = {'measureid':measure.id,
                    'status':''}
        cost = None

        destination_node = nodes[measure.imageid]

        reference_roi = roi.Roi(source_node.geodata, 
                                source.apriorisample, 
                                source.aprioriline,
                                size_x=match_kwargs['image_size'][0],
                                size_y=match_kwargs['image_size'][1])
        moving_roi = roi.Roi(destination_node.geodata, 
                             measure.apriorisample, 
                             measure.aprioriline,
                             size_x=match_kwargs['template_size'][0],
                             size_y=match_kwargs['template_size'][1],
                             buffer=5)

        affine = estimate_local_affine(reference_roi, moving_roi)

        # Updated so that the affine used is computed a single time.
        # Has not scale or shear or rotation.
        updated_affine, maxcorr, _ = subpixel_template(reference_roi,
                                                        moving_roi,
                                                        affine=baseline_affine)
        
        if updated_affine is None:
            log.warn('Unable to match with this parameter set.')
            currentlog['status'] = f"subpixel registration failed on measure {measure.id}"
            resultlog.append(currentlog)
            if measure.weight is None:
                measure.ignore = True # Geom match failed and no previous sucesses
            updated_measures.append(measure)
            continue
        new_x, new_y = updated_affine([measure.apriorisample, measure.aprioriline])[0]

        if new_x == None or new_y == None:
            currentlog['status'] = f'Failed to register measure {measure.id}.'
            resultlog.append(currentlog)
            if measure.weight is None:
                measure.ignore = True # Unable to geom match and no previous sucesses
            updated_measures.append(measure)
            continue

        measure.template_metric = maxcorr
        dist = np.linalg.norm([measure.apriorisample-new_x, measure.aprioriline-new_y])
        measure.template_shift = dist

        cost = cost_func(measure.template_shift, measure.template_metric)

        log.info(f'Current Cost: {cost},  Current Weight: {measure.weight}')

        # Check to see if the cost function requirement has been met
        if measure.weight and cost <= measure.weight:
            currentlog['status'] = f'Previous match provided better correlation. {measure.weight} > {cost}.'
            resultlog.append(currentlog)
            updated_measures.append(measure)
            continue

        if cost <= threshold:
            currentlog['status'] = f'Cost failed. Distance calculated: {measure.template_shift}. Metric calculated: {measure.template_metric}.'
            resultlog.append(currentlog)
            updated_measures.append(measure)
            if measure.weight is None:
                measure.ignore = True # Threshold criteria not met and no previous sucesses
            continue

        # Update the measure
        measure.sample = new_x
        measure.line = new_y
        measure.weight = cost
        measure.choosername = chooser

        # In case this is a second run, set the ignore to False if this
        # measures passed. Also, set the source measure back to ignore=False
        measure.ignore = False
        # Maybe source?
        source.ignore = False
        updated_measures.append(measure)
        currentlog['status'] = f'Success. Distance shifted: {measure.template_shift}. Metric: {measure.template_metric}.'
        resultlog.append(currentlog)

    # Once here, update the source measure (possibly back to ignore=False)
    updated_measures.append(source)

    if use_cache:
        ncg.redis_queue.rpush(ncg.measure_update_queue,
                              *[json.dumps(measure.to_dict(_hide=[]), cls=JsonEncoder) for measure in updated_measures])
        ncg.redis_queue.incr(ncg.measure_update_counter, amount=len(updated_measures))

    else:
        # Commit the updates back into the DB
        with ncg.session_scope() as session:
            for m in updated_measures:
                ins = inspect(m)
                session.add(m)

    return resultlog

def subpixel_register_points(subpixel_template_kwargs={'image_size':(251,251)},
                             cost_kwargs={},
                             threshold=0.005,
                             Session=None):
    """
    Serial subpixel registration of all of the points in a given DB table.

    Parameters
    ----------
    Session : obj
              A SQLAlchemy Session factory.

    pointid : int
              The identifier of the point in the DB

    subpixel_template_kwargs : dict
                               Ay keyword arguments passed to the template matcher

    cost : func
           A generic cost function accepting two arguments (x,y), where x is the
           distance that a point has shifted from the original, sensor identified
           intersection, and y is the correlation coefficient coming out of the
           template matcher.

    threshold : numeric
                measures with a cost <= the threshold are marked as ignore=True in
                the database.
    """
    if not Session:
        raise BrokenPipeError('This func requires a database session.')
    session = Session()
    pointids = [point.id for point in session.query(Points)]
    session.close()
    for pointid in pointids:
        subpixel_register_point(pointid,
                                subpixel_template_kwargs=subpixel_template_kwargs,
                                **cost_kwargs)

def register_to_base(pointid,
                     base_image,
                     cost_func=lambda x, y: y == np.max(x),
                     ncg=None,
                     geom_func='simple',
                     geom_kwargs={"size_x": 16, "size_y": 16},
                     match_func='classic',
                     match_kwargs={},
                     verbose=False,
                     **kwargs):
    """
    """

    if not ncg.Session:
     raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    geom_func = check_geom_func(geom_func)
    match_func = check_match_func(match_func)
    session = ncg.Session()

    if isinstance(base_image, str):
        base_image = GeoDataset(base_image)

    if isinstance(pointid, Points):
        point = pointid
        pointid = pointid.id

    with ncg.session_scope() as session:
        if isinstance(pointid, Points):
            point = pointid
            pointid = point.id
        else:
            point = session.query(Points).filter(Points.id == pointid).one()

        # Get all of the measures associated with the given point
        measures = point.measures

        # Attempt to project the point into the base image
        bpoint = spatial.isis.point_info(base_image.file_name, point.geom.x, point.geom.y, 'ground')
        if bpoint is None:
            log.warning('unable to find point in ground image')
            # Need to set the point to False
            return
        bline = bpoint.get('Line')
        bsample = bpoint.get('Sample')

        # Setup a cache so that we can get the file handles one time instead of
        # once per measure in the measures list.
        image_cache = {}

        # list of matching results in the format:
        # [measure_id, measure_index, x_offset, y_offset, offset_magnitude]
        match_results = []

        # Step over all of the measures (images) that are not the base image
        for measure_index, measure in enumerate(measures):
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            try:
                measure_image = image_cache[res.id]
            except:
                measure_image = GeoDataset(measure_image.path)
                image_cache[res.id] = measure_image

            # Attempt to match the base
            try:
                log.info(f'prop point: base_image: {base_image}')
                log.info(f'prop point: dest_image: {measure_image}')
                log.info(f'prop point: (sx, sy): ({measure.sample}, {measure.line})')
                x, y, dist, metrics = geom_func(base_image, measure_image,
                        bsample, bline,
                        match_func = match_func,
                        match_kwargs = match_kwargs,
                        verbose=verbose,
                        **geom_kwargs)

            except Exception as e:
                raise Exception(e)
                match_results.append(e)
                continue

            match_results.append([measure.id, measure_index, x, y,
                                 metrics, dist, base_image.file_name, measure_image.file_name])

    if verbose:
      log.info("Match Results: ", match_results)

    # Clean out any instances where None has been return by the geom matcher.
    match_results = np.copy(np.array([res for res in match_results if isinstance(res, list) and all(r is not None for r in res)]))

    # If all of the results are None, this point was not matchable
    if match_results.shape[0] == 0:
        raise Exception("Point with id {pointid} has no measure that matches criteria, reference measure will remain unchanged")

    # Compute the cost function for each match using the
    costs = [cost_func(match_results[:,3], match[3]) for match in match_results]

    if verbose:
      log.info("Values:", costs)

    # column index 3 is the metric returned by the geom matcher
    best_results = match_results[np.argmax(costs)]

    if verbose:
        log.info("match_results final length: ", len(match_results))
        log.info("best_results length: ", len(best_results))
        log.info("Full results: ", best_results)
        log.info("Winning CORRs: ", best_results[3], "Base Pixel shifts: ", best_results[4])
        log.info('\n')

    if len(best_results[3])==1 or best_results[3] is None:
        raise Exception("Point with id {pointid} has no measure that matches criteria, reference measure will remain unchanged")

    # Finally, update the point that will be the reference
    with ncg.session_scope() as session:
       measure = session.query(Measures).filter(Measures.id == best_results[0]).one()
       measure.sample = best_results[2]
       measure.line = best_results[3]

       point = session.query(Points).filter(Points.id == pointid).one()
       point.ref_measure = best_results[1]
    return

def estimate_logpolar_transform(img1, img2, low_sigma=0.5, high_sigma=30, verbose=False):
    """
    Estimates the rotation and scale difference for img1 that maps to img2 using phase cross correlation on a logscale projection.

    Scale and angular changes in cartesian space become translation in log-polar space. Translation from subpixel registration
    in log-polar space can then be translated into scale/rotation change in the original cartesian images. This scale + roation
    change estimation is then returned as an affine transform object. This can then be used before other subpixel registration
    methods to enable scale+rotation invariance.

    See Also
    --------

    skimage.filters.difference_of_gaussians : Bandpass filtering using a difference of gaussians
    skimage.filters.window : Simple wondowing function to remove spectral leakage along the axes in the fourier transform

    References
    ----------

    .. [1] Rittavee Matungka. 2009. Studies on log-polar transform for image registration and improvements
       using adaptive sampling and logarithmic spiral. Ph.D. Dissertation. Ohio State University, USA. Advisor(s) Yuan F. Zheng.
       Order Number: AAI3376091.
    .. [2] https://github.com/polakluk/fourier-mellin
    .. [3] https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html

    Parameters
    ----------

    img1: np.ndarray
          The source image, this is the image that is used as a base as img2 is registered to the center on img1

    img2: np.ndarray
          The image that will be moved to match img1

    low_sigma : float, list, np.array
                The low standard deviation for the Gaussian kernel used in the difference of gaussians filter. This reccomended
                to remove high frequency noise from the image before the log-polar projection as high frequency noise negatively impact registration
                in log-polar space. The lower the sigma, the sharper the resulting image will be. Use a small low_sigma with a large high_sigma
                to remove high frequency noise. Default is 0.5.

    high_sigma : float, list, np.array
                Standard deviation for the Gaussian kernel with the larger sigmas across all axes used in the difference of gaussians filter. This reccomended
                to remove high frequency noise from the image before the log-polar projection as high frequency noise negatively impact registration
                in log-polar space. The higher this sigma compared to the low_sigma, the more detail will be preserved. Use a small low_sigma with a large high_sigma
                to remove high frequency noise. A high sigma equal to ~1.6x the low sigma is reccomended for edge detection, so consider high_sigmas >= low_sigma*1.6. Default is 30.

    verbose : bool
              If true, prints out information detailing the registration process

    Returns
    -------
    : skimage.transform.SimilarityTansform
      Scikit-image affine transformation object containing rotation and scale information to warp img1 to img2

    """
    # First, band-pass filter both images
    img1 = filters.difference_of_gaussians(img1, low_sigma, high_sigma)
    img2 = filters.difference_of_gaussians(img2, low_sigma, high_sigma)

    # window images
    wimg1 = img1 * (filters.window('hann', img1.shape))
    wimg2 = img2 * (filters.window('hann', img2.shape))

    # work with shifted FFT magnitudes
    img1_fs = np.abs(fftpack.fftshift(fftpack.fft2(wimg1)))
    img2_fs = np.abs(fftpack.fftshift(fftpack.fft2(wimg2)))

    # Create log-polar transformed FFT mag images and register
    shape = img1_fs.shape
    radius = shape[0] // 4  # only take lower frequencies
    warped_img1_fs = tf.warp_polar(img1_fs, radius=radius, output_shape=shape,
                                 scaling='log', order=0)
    warped_img2_fs = tf.warp_polar(img2_fs, radius=radius, output_shape=shape,
                               scaling='log', order=0)

    warped_img1_fs = warped_img1_fs[:shape[0] // 2, :]
    warped_img2_fs = warped_img2_fs[:shape[0] // 2, :]
    shifts, error, phasediff = registration.phase_cross_correlation(warped_img1_fs,
                                                       warped_img2_fs,
                                                       upsample_factor=10)

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = -(360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)
    if recovered_angle < - 45:
        recovered_angle += 180
    else:
        if recovered_angle > 90.0:
            recovered_angle -= 180

    if verbose:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()
        ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
        center = np.array(shape) // 2
        ax[0].imshow(img1_fs[center[0] - radius:center[0] + radius,
                              center[1] - radius:center[1] + radius],
                     cmap='magma')
        ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
        ax[1].imshow(img2_fs[center[0] - radius:center[0] + radius,
                            center[1] - radius:center[1] + radius],
                     cmap='magma')
        ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
        ax[2].imshow(warped_img1_fs, cmap='magma')
        ax[3].set_title("Log-Polar-Transformed\nModified FFT")
        ax[3].imshow(warped_img2_fs, cmap='magma')
        fig.suptitle('Working in frequency domain can recover rotation and scaling')
        plt.show()

        log.info(f"Recovered value for cc rotation: {recovered_angle}")
        log.info(f"Recovered value for scaling difference: {shift_scale}")

    # offset by the center of the image, scikit's ceter image rotation is defined by `axis / 2 - 0.5`
    shift_y, shift_x = np.asarray(img1.shape) / 2 - 0.5
    tf_scale = tf.SimilarityTransform(scale=shift_scale)
    tf_rotate = tf.SimilarityTransform(rotation=np.deg2rad(recovered_angle))
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

    tf_rotate_from_center = (tf_shift + (tf_rotate + tf_shift_inv))
    return tf.SimilarityTransform((tf_rotate_from_center + tf_scale)._inv_matrix)
       
def fourier_mellen(ref_image, moving_image, affine=tf.AffineTransform(), verbose=False, phase_kwargs={}):
    """
    Iterative phase registration using a log-polar projection to estimate an affine for scale and roation invariance.


    Parameters
    ----------

    ref_image: np.ndarray
               The source image, this is the image that is used as a base as img2 is registered to the center on img1

    moving_image: np.ndarray
                  The image that will be moved to match img1

    verbose : bool
              If true, prints out information detailing the registration process

    phase_kwargs : dict
                   Parameters to be passed into the iterative_phase matcher

    Returns
    -------

    : float
      The new x coordinate for moving_image that registers to the center of center_image

    : float
      The new y coordinate for moving_image that registers to the center of center_image

    : float
      Error returned by the iterative phase matcher
    """
    # Get the affine transformation for scale + rotation invariance
    affine = estimate_logpolar_transform(ref_image.array, moving_image.array, verbose=verbose)

    # warp the source image to match the destination
    ref_warped = ref_image.clip(affine)
    sx, sy = affine.inverse(np.asarray(ref_image.array.shape)/2)[0]

    # get translation with iterative phase
    newx, newy, error = iterative_phase(sx, sy, sx, sy, ref_warped, moving_image, **phase_kwargs)

    if verbose:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax = axes.ravel()

        ax[0].imshow(ref_warped)
        ax[0].set_title("Image 1 Transformed")
        ax[0].axhline(y=sy, color="red", linestyle="-", alpha=1, linewidth=1)
        ax[0].axvline(x=sx, color="red", linestyle="-", alpha=1, linewidth=1)

        ax[2].imshow(ref_image)
        ax[2].set_title("Image 1")
        ax[2].axhline(y=ref_image.array.shape[0]/2, color="red", linestyle="-", alpha=1, linewidth=1)
        ax[2].axvline(x=ref_image.array.shape[1]/2, color="red", linestyle="-", alpha=1, linewidth=1)

        ax[1].imshow(moving_image)
        ax[3].imshow(moving_image)

        if not newx or not newy:
            ax[1].set_title("Image 2 REGISTRATION FAILED")
            ax[3].set_title("Image 2 REGISTRATION FAILED")
        else :
            ax[3].set_title("Image 2 Registered")
            ax[1].axhline(y=newy, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[1].axvline(x=newx, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[3].axhline(y=newy, color="red", linestyle="-", alpha=1, linewidth=1)
            ax[3].axvline(x=newx, color="red", linestyle="-", alpha=1, linewidth=1)

    return newx, newy, error

def subpixel_register_point_smart(pointid,
                            cost_func=lambda x,y: 1/x**2 * y,
                            ncg=None,
                            parameters=[],
                            chooser='subpixel_register_point_smart',
                            verbose=False):

    """
    Given some point, subpixel register all of the measures in the point to the
    reference measure.

    Parameters
    ----------
    
    pointid : int or obj
              The identifier of the point in the DB or a Points object

    cost_func : func
                A generic cost function accepting two arguments (x,y), where x is the
                distance that a point has shifted from the original, sensor identified
                intersection, and y is the correlation coefficient coming out of the
                template matcher.

    ncg : obj
          the network candidate graph that the point is associated with; used for
          the DB session that is able to access the point.

    parameters : list
                 of dicts containing "match_kwargs" used for specified match_func.
                 The passed parameters describe image and template chips that are tested.
                 For example parameters = [
                 {'match_kwargs': {'image_size':(121,121), 'template_size':(61,61)}},
                 {'match_kwargs': {'image_size':(151,151), 'template_size':(67,67)}},
                 {'match_kwargs': {'image_size':(181,181), 'template_size':(73,73)}}]
    """


    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    if isinstance(pointid, Points):
        pointid = pointid.id

    with ncg.session_scope() as session:
        # Order by is important here because the measures get ids in sequential order when initially placed
        # and the reference_index is positionally linked to the ordered vector of measures.
        measures = session.query(Measures).filter(Measures.pointid == pointid).order_by(Measures.id).all()
        point = session.query(Points).filter(Points.id == pointid).one()
        reference_index = point.reference_index

        # Get the reference measure to instantiate the source node. All other measures will
        # match to the source node.
        source = measures[reference_index]
        reference_index_id = source.imageid

        log.info(f'Using measure {source.id} on image {source.imageid}/{source.serial} as the reference.')
        log.info(f'Measure reference index is: {reference_index}')

        # Build a node cache so that this is an encapsulated database call. Then nodes
        # can be pulled from the lookup sans database.
        nodes = {}
        for measure in measures:
            res = session.query(Images).filter(Images.id == measure.imageid).one()
            nn = NetworkNode(node_id=measure.imageid, image_path=res.path)
            nn.parent = ncg
            nodes[measure.imageid] = nn

        session.expunge_all()

    log.info(f'Attempting to subpixel register {len(measures)-1} measures for point {pointid}')
    # Set the reference image
    source_node = nodes[reference_index_id]

    log.info(f'Source: sample: {source.sample} | line: {source.line}')
    updated_measures = []
    for i, measure in enumerate(measures):

        # If this is the reference node, do not attempt to match it.
        if i == reference_index:
            continue

        cost = None
        destination_node = nodes[measure.imageid]
        log.info(f'Registering measure {measure.id} (image: {measure.imageid})')

        # Compute the baseline metrics using the smallest window
        size_x = np.inf
        size_y = np.inf
        for p in parameters:
            match_kwarg = p['match_kwargs']
            if match_kwarg['template_size'][0] < size_x:
                size_x = match_kwarg['template_size'][0]
            if match_kwarg['template_size'][1] < size_y:
                size_y = match_kwarg['template_size'][1]

        reference_roi = roi.Roi(source_node.geodata, 
                                source.apriorisample, 
                                source.aprioriline, 
                                size_x=size_x, 
                                size_y=size_y, 
                                buffer=0)
        moving_roi = roi.Roi(destination_node.geodata, 
                             measure.apriorisample, 
                             measure.aprioriline, 
                             size_x=size_x, 
                             size_y=size_y, 
                             buffer=20)

        try:
            baseline_affine = estimate_local_affine(reference_roi,
                                                    moving_roi)
        except Exception as e:
            log.exception(e)
            m = {'id': measure.id,
                 'sample':measure.apriorisample,
                 'line':measure.aprioriline,
                 'status':False,
                 'choosername':chooser}
            updated_measures.append([None, None, m])
            continue

        reference_roi.clip()
        reference_clip = reference_roi.clipped_array

        # If the image read center plus buffer is outside the image, clipping
        # raises an index error. Handle and set the measure false.
        try:
            moving_roi.clip(affine=baseline_affine)
            moving_clip = moving_roi.clipped_array
        except Exception as e:
            log.error(e)
            m = {'id': measure.id,
                 'sample':measure.apriorisample,
                 'line':measure.aprioriline,
                 'status':False,
                 'choosername':chooser}
            updated_measures.append([None, None, m])
            continue

        reference_nan = np.isnan(reference_clip).any()
        moving_nan = np.isnan(moving_clip).any()
        if reference_nan or moving_nan:
            if reference_nan:
                log.warning(f'Unable to process due to NaN values in the reference data.')
            if moving_nan:
                log.warning(f'Unable to process due to NaN values in the moving data.')
            m = {'id': measure.id,
                    'status': False,
                    'choosername': chooser}
            updated_measures.append([None, None, m])
            continue

        if reference_clip.shape != moving_clip.shape:
            log.warning('Unable to process. ROIs are different sizes for MI matcher')
            m = {'id': measure.id,
                 'status': False,
                 'choosername': chooser}
            updated_measures.append([None, None, m])
            continue

        # Compute the a priori template and MI correlations with no shifts allowed. 
        _, baseline_corr, _ = subpixel_template(reference_roi, 
                                                moving_roi,
                                                affine=baseline_affine)
        
        baseline_mi = 0 #mutual_information_match(reference_roi, 
                        #                         moving_roi, 
                        #                         affine=baseline_affine)
        
        log.info(f'Baseline MI: {baseline_mi} | Baseline Corr: {baseline_corr}')
        
        for parameter in parameters:
            match_kwargs = parameter['match_kwargs']

            reference_roi = roi.Roi(source_node.geodata,
                                    source.apriorisample,
                                    source.aprioriline,
                                    size_x=match_kwargs['image_size'][0],
                                    size_y=match_kwargs['image_size'][1], 
                                    buffer=0)
            moving_roi = roi.Roi(destination_node.geodata,
                                 measure.apriorisample,
                                 measure.aprioriline,
                                 size_x=match_kwargs['template_size'][0],
                                 size_y=match_kwargs['template_size'][1], 
                                 buffer=20)
            if verbose:
                fig, axes = plt.subplots(1,2)
                reference_roi.clip()
                moving_roi.clip(affine=baseline_affine)
                axes[0].imshow(reference_roi.clipped_array, cmap='Greys')
                axes[1].imshow(moving_roi.clipped_array, cmap='Greys')
                plt.show()
            try:  # Handle the case where the parameter set plus the buffer is outside
                  # the image and the clip raises an index error.
                updated_affine, maxcorr, _ = subpixel_template(reference_roi,
                                                            moving_roi,
                                                            affine=baseline_affine)
            except Exception as e:
                log.error(e)
                updated_affine = None

            if updated_affine is None:
                log.warning(f'Unable to match with this parameter set.')
                continue
            else:
                mi_metric=0
                metric = maxcorr
                new_x, new_y = updated_affine([measure.apriorisample, measure.aprioriline])[0]
                
                dist = np.linalg.norm([measure.aprioriline-new_x, 
                                      measure.apriorisample-new_y])
                cost = cost_func(dist, metric)

                m = {'id': measure.id,
                    'sample':new_x,
                    'line':new_y,
                    'weight':cost,
                    'choosername':chooser,
                    'template_metric':metric,
                    'template_shift':dist,
                    'mi_metric': mi_metric,
                    'status': True}
                log.info(f'METRIC: {metric}| SAMPLE: {new_x} | LINE: {new_y} | MI: {mi_metric}')
            
            updated_measures.append([baseline_mi, baseline_corr, m])

    # Baseline MI, Baseline Correlation, updated measures to select from
    return updated_measures

def check_for_shift_consensus(shifts, tol=0.1):
    """
    Find matched locations from a set of multiple different solutions that have
    the same position within some user supplied tolerance. If the distance between
    two measures (shifts) is <= the tolerance, the measures are considered to have
    found consensus.

    This doc string uses 'measure' to describe each solution found by a subpixel
    matching attempt. If n-attempts are made, using n-different parameter sets,
    this function will find shift consensus between those n-different solutions.

    The function works by computing the full distance matrix between all solutions,
    generating a boolean mask for distances less than the tolerance, then generating
    a vector of column sums where the sum is the number of inliers, and finally,
    returning a boolean vector where the column sums are greater than 2.

    Parameters
    ----------
    shifts : ndarray
             (n,2) array of (x,y) coordinates representing the subpixel registered
             measure locations. n must be >= 3.

    tol : float
          The tolerance value required for measures to be inliers. Distances between
          points less than or equal to the tolerance are inliers. In pixel space.

    Returns
    -------
     : ndarray
       (n,1) boolean array where the nth element corresponds to the nth measure
       in the shifts input array. True values indicate that the measure has shift
       consensus with at least 2 other measures
    """
    dists = distance_matrix(shifts, shifts)
    inliers = dists <= tol
    col_sums = np.sum(inliers, 1)
    # The distance matrix is zero diagonal, so 2+ means one other matcher found
    # a close location
    return col_sums >= 2
    
def decider(measures, tol=0.5):
    """
    The logical decision function that determines which measures would be updated
    with subpixel registration or ignored. The function iterates over the measures,
    looks for shift consensus between subpixel registration runs.

    Parameters
    ----------
    measures : list
               A list of candidate measures (dicts) objects from the smart subpixel matcher

    tol : float
          The tolerance value required for points to be inliers. Distances between
          points less than or equal to the tolerance are inliers. In pixel space.

    Returns
    -------
    measures_to_update : list
                         of measures (dicts) to be updated to subpixel accuracy

    measures_to_set_false : list
                            of meaure ids to be ignored beause theu fail the consensus
                            building approach
    """
    by_id = defaultdict(list)
    measures_to_set_false = []
    for m in measures:
        baseline_mi = m[0]
        baseline_corr = m[1]
        m = m[2]
        if m['status'] and m['mi_metric'] is not None and m['template_metric'] is not None:
            choosername = m['choosername']
            by_id[m['id']].append([m['line'],
                                  m['sample'],
                                  m['mi_metric'],
                                  m['template_metric'],
                                  baseline_mi,
                                  baseline_corr,
                                  m['template_shift']])
        else:
            measures_to_set_false.append(m['id'])

    measures_to_update = []
    for k, v in by_id.items():
        v = np.asarray(v)
        mi = v[:,2]
        corr = v[:,3]
        baseline_mi = v[:,4]
        baseline_corr = v[:,5]
        cost = (baseline_mi - mi) + (baseline_corr - corr)

        # At least two of the correlators need to have found a soln within 0.5 pixels.
        shift_mask = check_for_shift_consensus(v[:,:2], tol=tol)

        # This is formulated as a minimization, so the best is the min cost
        best_cost = np.argmin(cost)

        if shift_mask[best_cost] == False:
            # The best cost does not have positional consensus
            measures_to_set_false.append(k)
        else:
            best_measure = v[best_cost]
            m = {'id':k,
                 'line': best_measure[0],
                 'sample': best_measure[1],
                 'weight': cost[best_cost],
                 'template_metric': best_measure[3],
                 'template_shift': best_measure[6],
                 'choosername': choosername,
                 'ignore':False,
                 'best_parameter_index': best_cost}
            measures_to_update.append(m)
    # A measure could have one bad regitration and get set false, if a different parameter set passed,
    # remove from the set false list.
    ids_to_update = [d['id'] for d in measures_to_update]
    measures_to_set_false = [i for i in measures_to_set_false if i not in ids_to_update]

    return measures_to_update, measures_to_set_false

def validate_candidate_measure(measure_to_register,
                               ncg=None,
                               parameters=[],
                               **kwargs):
    """
    Compute the matching distances, matching the reference measure to the measure
    originally registered to it. This is an inverse check from the original mathcing.
    In other words, the first registration registers A->B to find measure_to_register (B-naught).
    This func then matches B->A (B-prime) and computes the distance between B-naught and B-prime.

    Parameters
    ----------
    measure_to_register : dict
                          A dictionary containing information about the measure to validate, the
                          {keys: types} needed for this function are: 
                          {'id': int, 
                          'line': np.float, 
                          'sample': np.float,
                          'parameters_index': dict}

    ncg : obj
          A network candidate graph object

    geom_func : str
                The func to use to perform geometric matching

    match_func : str
                 The function to use to perform matching

    parameters : list
                 A list of matching parametrizations to test. Each entry results in
                 a subpixel registration attempt and then set of these results is
                 used ot ientify inliner and outlier parameter sets.

    Returns
    -------
    dists : list
            Of reprojection distances for each parameter set.
    """

    if not ncg.Session:
        raise BrokenPipeError('This func requires a database session from a NetworkCandidateGraph.')

    measure_to_register_id = measure_to_register['id']

    with ncg.session_scope() as session:
        # Get the measure to be registered
        measure = session.query(Measures).filter(Measures.id == measure_to_register_id).order_by(Measures.id).one()
        # Get the references measure
        point = measure.point
        reference_index = point.reference_index
        reference_measure = point.measures[reference_index]


        # Match the reference measure to the measure_to_register - this is the inverse of the first match attempt
        # Source is the image that we are seeking to validate, destination is the reference measure.
        # This is the inverse of other functions as this is a validator.

        source_imageid = measure.imageid
        source_image = session.query(Images).filter(Images.id == source_imageid).one()
        source_node = NetworkNode(node_id=source_imageid, image_path=source_image.path)
        source_node.parent = ncg

        destination_imageid = reference_measure.imageid
        destination_image = session.query(Images).filter(Images.id == destination_imageid).one()
        destination_node = NetworkNode(node_id=destination_imageid, image_path=destination_image.path)
        destination_node.parent = ncg

        sample = measure_to_register['sample']
        line = measure_to_register['line']

        log.info(f'Validating measure: {measure_to_register_id} on image: {source_imageid}')

        reference_roi = roi.Roi(source_node.geodata, 
                                sample, 
                                line, 
                                size_x=parameters[0]['match_kwargs']['image_size'][0],
                                size_y=parameters[0]['match_kwargs']['image_size'][1], 
                                buffer=10)
        moving_roi = roi.Roi(destination_node.geodata, 
                                reference_measure.sample, 
                                reference_measure.line, 
                                size_x=parameters[0]['match_kwargs']['template_size'][0],
                                size_y=parameters[0]['match_kwargs']['template_size'][1],
                                buffer=10)

        try:
            baseline_affine = estimate_local_affine(reference_roi, moving_roi)
        except:
            log.error('Unable to transform image to reference space. Likely too close to the edge of the non-reference image. Setting ignore=True')
            return [np.inf] * len(parameters)
        

        dists = []
        for i, parameter in enumerate(parameters):
            match_kwargs = parameter['match_kwargs']

            reference_roi = roi.Roi(source_node.geodata, 
                                    sample, 
                                    line, 
                                    size_x=match_kwargs['image_size'][0],
                                    size_y=match_kwargs['image_size'][1], 
                                    buffer=10)
            moving_roi = roi.Roi(destination_node.geodata, 
                                    reference_measure.sample, 
                                    reference_measure.line, 
                                    size_x=match_kwargs['template_size'][0],
                                    size_y=match_kwargs['template_size'][1],
                                    buffer=10)

            # Handle the exception where the clip can raise an index error if it is outside the image
            try:
                updated_affine, maxcorr, _ = subpixel_template(reference_roi,
                                                            moving_roi,
                                                            affine=baseline_affine)
            except:
                updated_affine = None

            if updated_affine is None:
                continue
            
            new_x, new_y = updated_affine([reference_measure.sample, 
                                           reference_measure.line])[0]
            
            dist = np.sqrt((new_y - reference_measure.line) ** 2 +\
                           (new_x - reference_measure.sample) ** 2)
            log.info(f'Validating using parameter set {i}. Reprojection distance: {dist}. Metric: {maxcorr}')
            dists.append(dist)
        return dists

def smart_register_point(pointid, parameters=[], shared_kwargs={}, valid_reprojection_distance=1.1, ncg=None, Session=None):
    """
    The entry func for the smart subpixel registration code. This is the user
    side API func for subpixel registering a point using the smart matcher.

    This function runs multiple rounds of subpixel registration on a point
    using 'subpixel_register_point_smart', checks for a consensus from the
    subpixel registration results, and validates the new location by inverting
    the matching direction. This function writes to the database and outputs
    the updated and ignored measures for logging purposes.

    This func writes to the databse. The returns are for logging and
    debugging convenience.

    Parameters
    ----------
    pointid : int or object
              The id of the point to register or a point object from which the
              id is accessed.

    parameters : list
                 A list of dict subpixel registration kwargs, {template_size: (x,x), image_size: (y,y)}

    shared_kwargs : dict
                    of kwargs passed to the subpixel matcher that are shared between all of the parameter sets

    ncg : obj
          A network candidate graph object

    Session : obj
              An optional sqlalchemy Session factory

    valid_reprojection_distance : float
                                  measures matched from the moving image to the reference image with a 
                                  distance less than this value in pixels are considered valid. Default: 1.1 

    Returns
    -------
    measures_to_update : list
                         of measures (dicts) to be updated to subpixel accuracy

    measures_to_set_false : list
                            of meaure ids to be ignored beause theu fail the consensus
                            building approach

    """
    if isinstance(pointid, Points):
        pointid = pointid.id
    measure_results = subpixel_register_point_smart(pointid, ncg=ncg, parameters=parameters, **shared_kwargs)
    measures_to_update, measures_to_set_false = decider(measure_results)

    log.info(f'Found {len(measures_to_update)} measures that found subpixel registration consensus. Running validation now...')
    # Validate that the new position has consensus
    for measure in measures_to_update:
        reprojection_distances = validate_candidate_measure(measure, parameters=parameters, ncg=ncg, **shared_kwargs)
        if np.sum(np.array(reprojection_distances) < valid_reprojection_distance) < 2:
            log.info(f"Measure {measure['id']} failed validation. Setting ignore=True for this measure.")
            measures_to_set_false.append(measure['id'])

    for measure in measures_to_update:
        measure['_id'] = measure.pop('id', None)

    # Update the measures that passed registration
    with ncg.engine.connect() as conn:
        if measures_to_update:
            stmt = Measures.__table__.update().\
                                    where(Measures.__table__.c.id == bindparam('_id')).\
                                    values({'weight':bindparam('weight'),
                                            'measureIgnore':bindparam('ignore'),
                                            'templateMetric':bindparam('template_metric'),
                                            'templateShift':bindparam('template_shift'),
                                            'line': bindparam('line'),
                                            'sample':bindparam('sample'),
                                            'ChooserName':bindparam('choosername')})
            resp = conn.execute(
                stmt, measures_to_update
            )
        if measures_to_set_false:
            measures_to_set_false = [{'_id':i} for i in measures_to_set_false]
            # Set ignore=True measures that failed
            stmt = Measures.__table__.update().\
                                    where(Measures.__table__.c.id == bindparam('_id')).\
                                    values({'measureIgnore':True,
                                            'ChooserName':shared_kwargs['chooser']})
            resp = conn.execute(
                stmt, measures_to_set_false
            )
    log.info(f'Updated measures: {json.dumps(measures_to_update, indent=2, cls=JsonEncoder)}')
    log.info(f'Ignoring measures: {measures_to_set_false}')

    return measures_to_update, measures_to_set_false
