import logging
import time
import numpy as np
from plio.io.io_gdal import GeoDataset
from skimage import transform as tf

from autocnet.transformation.roi import Roi
from autocnet.io.geodataset import AGeoDataset

log = logging.getLogger(__name__)

def check_for_excessive_shear(transformation):
    """
    This function checks a skimage projective transform for excessive
    shearing of the data.

    In a projective transformation the final row encodes and adjustment to 
    the vertical/horizontal lines to infinity. The first element u encodes
    a hinging of the along the horizontal and the second element v encodes
    a hinging along the vertical line. As the values approach 0.001 and -0.001
    the amount of shear related distortion (hinging along the horizontal/vertical)
    becomes high enough that matching is problematic as the amount of data
    needed in the templates becomes quite high.
    """
    m = transformation.params
    u = m[2][0]
    v = m[2][1]

    if abs(u) >= 0.001 or abs(v) >= 0.001:
        return True
    return False

def estimate_affine_from_sensors(reference_image,
                                 moving_image,
                                 bcenter_x,
                                 bcenter_y,
                                 size_x=40,
                                 size_y=40):
    """
    Using the a priori sensor model, project corner and center points from the reference_image into
    the moving_image and use these points to estimate an affine transformation.

    Parameters
    ----------
    reference_image: autocnet.io.geodataset.AGeoDataset
                source image
    moving_image: autocnet.io.geodataset.AGeoDataset
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
    if not isinstance(moving_image,AGeoDataset):
        raise Exception(f"Input cube must be a geodataset obj, but is type {type(moving_image)}.")
    if not isinstance(reference_image, AGeoDataset):
        raise Exception(f"Match cube must be a geodataset obj, but is type {type(reference_image)}.")

    base_startx = int(bcenter_x - size_x)
    base_starty = int(bcenter_y - size_y)
    base_stopx = int(bcenter_x + size_x)
    base_stopy = int(bcenter_y + size_y)

    match_size = reference_image.raster_size

    # for now, require the entire window resides inside both cubes.
    # if base_stopx > match_size[0]:
    #     raise Exception(f"Window: {base_stopx} > {match_size[0]}, center: {bcenter_x},{bcenter_y}")
    # if base_startx < 0:
    #     raise Exception(f"Window: {base_startx} < 0, center: {bcenter_x},{bcenter_y}")
    # if base_stopy > match_size[1]:
    #     raise Exception(f"Window: {base_stopy} > {match_size[1]}, center: {bcenter_x},{bcenter_y} ")
    # if base_starty < 0:
    #     raise Exception(f"Window: {base_starty} < 0, center: {bcenter_x},{bcenter_y}")
    
    x_coords = [base_startx, base_startx, base_stopx, base_stopx, bcenter_x]
    y_coords = [base_starty, base_stopy, base_stopy, base_starty, bcenter_y]
    # Dispatch to the sensor to get the a priori pixel location in the input image
    lons, lats = reference_image.sensormodel.sampline2lonlat(x_coords, y_coords, allowoutside=True)
    xs, ys = moving_image.sensormodel.lonlat2sampline(lons, lats, allowoutside=True)
    log.debug(f'Lon/Lats for affine estimate are: {list(zip(lons, lats))}')
    log.debug(f'Image X / Image Y for affine estimate are: {list(zip(xs, ys))}')

    # Check for any coords that do not project between images
    base_gcps = []
    dst_gcps = []
    for i, (base_x, base_y) in enumerate(zip(x_coords, y_coords)):
        if xs[i] is not None and ys[i] is not None:
            dst_gcps.append((xs[i], ys[i]))
            base_gcps.append((base_x, base_y))
    if len(dst_gcps) < 3:
        raise ValueError(f'Unable to find enough points to compute an affine transformation. Found {len(dst_gcps)} points, but need at least 3.')

    log.debug(f'Number of GCPs for affine estimation: {len(dst_gcps)}')
    affine = tf.AffineTransform()
    # Estimate the affine twice. The first time to get an initial estimate
    # and the second time to drop points with an estimated reprojection 
    # error greater than or equal to 0.1px.
    affine.estimate(np.array(base_gcps), np.array(dst_gcps))
    residuals = affine.residuals(np.array(base_gcps), np.array(dst_gcps))
    mask = residuals <= 1
    if len(np.array(base_gcps)[mask]) < 3:
        raise ValueError(f'Unable to find enough points to compute an affine transformation. Found {len(np.array(dst_gcps)[mask])} points, but need at least 3.')

    affine.estimate(np.array(base_gcps)[mask], np.array(dst_gcps)[mask])
    affine = tf.estimate_transform('affine', np.array(base_gcps), np.array(dst_gcps))
    log.debug(f'Computed afffine: {affine}')
    t2 = time.time()
    log.debug(f'Estimation of local affine took {t2-t1} seconds.')
    return affine


def estimate_local_affine(reference_roi, moving_roi, size_x=60, size_y=60):
    """
    Applies the affine transfromation calculated in estimate_affine_from_sensors to the moving region of interest (ROI).
    

    Parameters
    ----------
    reference_image : autocnet.io.geodataset.AGeoDataset
                      Image that is expected to be used as the reference during the matching process, 
                      points are laid onto here and projected onto moving image to compute an affine
    moving_image : autocnet.io.geodataset.AGeoDataset
                   Image that is expected to move around during the matching process, 
                   points are projected onto this image to compute an affine  

    Returns
    -------
    affine
        Affine matrix to transform the moving image onto the center image
    """
    transformation_matrix = estimate_affine_from_sensors(reference_roi.data, 
                                                    moving_roi.data, 
                                                    reference_roi.x, 
                                                    reference_roi.y, 
                                                    size_x=size_x, 
                                                    size_y=size_y)


    # Remove the translation from the transformation. Users of this function should add
    matrix = transformation_matrix.params
    matrix[0][-1] = 0
    matrix[1][-1] = 0
    tf_rotate = tf.AffineTransform(matrix)
    return tf_rotate
