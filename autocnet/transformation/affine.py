import logging
import time
import numpy as np
from plio.io.io_gdal import GeoDataset
from skimage import transform as tf

from autocnet.transformation.roi import Roi
from autocnet.spatial import isis

log = logging.getLogger(__name__)

def estimate_affine_from_sensors(reference_image,
                                moving_image,
                                bcenter_x,
                                bcenter_y,
                                size_x=60,
                                size_y=60):
    """
    Using the a priori sensor model, project corner and center points from the reference_image into
    the moving_image and use these points to estimate an affine transformation.

    Parameters
    ----------
    reference_image:  plio.io.io_gdal.GeoDataset
                source image
    moving_image: plio.io.io_gdal.GeoDataset
                destination image; gets matched to the source image
    bcenter_x:  int
                sample location of source measure in reference_image
    bcenter_y:  int
                line location of source measure in reference_image
    size_x:     int
                half-height of the subimage used in the affine transformation
    size_y:     int
                half-width of the subimage used in affine transformation

    Returns
    -------
    affine : object
             The affine transformation object

    """
    t1 = time.time()
    if not isinstance(moving_image, GeoDataset):
        raise Exception(f"Input cube must be a geodataset obj, but is type {type(moving_image)}.")
    if not isinstance(reference_image, GeoDataset):
        raise Exception(f"Match cube must be a geodataset obj, but is type {type(reference_image)}.")

    base_startx = int(bcenter_x - size_x)
    base_starty = int(bcenter_y - size_y)
    base_stopx = int(bcenter_x + size_x)
    base_stopy = int(bcenter_y + size_y)

    match_size = reference_image.raster_size

    '''# for now, require the entire window resides inside both cubes.
    if base_stopx > match_size[0]:
        raise Exception(f"Window: {base_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    if base_startx < 0:
        raise Exception(f"Window: {base_startx} < 0, center: {bcenter_x},{bcenter_y}")
    if base_stopy > match_size[1]:
        raise Exception(f"Window: {base_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    if base_starty < 0:
        raise Exception(f"Window: {base_starty} < 0, center: {bcenter_x},{bcenter_y}")
    '''
    x_coords = [base_startx, base_startx, base_stopx, base_stopx, bcenter_x]
    y_coords = [base_starty, base_stopy, base_stopy, base_starty, bcenter_y]

    # Dispatch to the sensor to get the a priori pixel location in the input image
    lons, lats = isis.image_to_ground(reference_image.file_name, x_coords, y_coords, allowoutside=True)
    xs, ys = isis.ground_to_image(moving_image.file_name, lons, lats, allowoutside=True)


    # Check for any coords that do not project between images
    base_gcps = []
    dst_gcps = []
    for i, (base_x, base_y) in enumerate(zip(x_coords, y_coords)):
        if xs[i] is not None and ys[i] is not None:
            dst_gcps.append((xs[i], ys[i]))
            base_gcps.append((base_x, base_y))
            
    if len(dst_gcps) < 3:
        raise ValueError(f'Unable to find enough points to compute an affine transformation. Found {len(dst_corners)} points, but need at least 3.')

    affine = tf.estimate_transform('affine', np.array([*base_gcps]), np.array([*dst_gcps]))
    t2 = time.time()
    log.debug(f'Estimation of local affine took {t2-t1} seconds.')
    return affine


def estimate_local_affine(reference_roi, moving_roi):
    """
    Applies the affine transfromation calculated in estimate_affine_from_sensors to the moving region of interest (ROI).
    

    Parameters
    ----------
    reference_image : plio.io.io_gdal.GeoDataset
                      Image that is expected to be used as the reference during the matching process, 
                      points are laid onto here and projected onto moving image to compute an affine
    moving_image : plio.io.io_gdal.GeoDataset
                   Image that is expected to move around during the matching process, 
                   points are projected onto this image to compute an affine  

    Returns
    -------
    affine
        Affine matrix to transform the moving image onto the center image
    """
    # get initial affine
    roi_buffer = reference_roi.buffer
    size_x = 60# reference_roi.size_x + roi_buffer
    size_y = 60# reference_roi.size_y + roi_buffer
    
    affine_transform = estimate_affine_from_sensors(reference_roi.data, moving_roi.data, reference_roi.x, reference_roi.y, size_x=size_x, size_y=size_y)


    # The above coordinate transformation to get the center of the ROI handles translation. 
    # So, we only need to rotate/shear/scale the ROI. Omitting scale, which should be 1 (?) results
    # in an affine transoformation that does not match the full image affine
    tf_rotate = tf.AffineTransform(rotation=affine_transform.rotation, 
                                   shear=affine_transform.shear,
                                   scale=affine_transform.scale)

    # This rotates about the center of the image
    shift_x, shift_y = (30.5, 30.5) #moving_roi.center
        
    tf_shift = tf.SimilarityTransform(translation=[shift_x, shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    
    # Define the full chain multiplying the transformations (read right to left),
    # this is 'shift to the center', apply the rotation, shift back
    # to the origin.
    trans = tf_shift_inv + tf_rotate + tf_shift
    return trans
