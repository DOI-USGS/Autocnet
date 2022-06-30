import logging
from math import floor

import cv2
import numpy as np

from scipy.ndimage.measurements import center_of_mass
from skimage.transform import rescale
from skimage.util import img_as_float32

log = logging.getLogger(__name__)


def pattern_match_autoreg(template, image, subpixel_size=3, metric=cv2.TM_CCOEFF_NORMED):
    """
    Call an arbitrary pattern matcher using a subpixel approach where a center of gravity using
    the correlation coefficients are used for subpixel alignment.
    
    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image
    image : ndarray
            The image or sub-image to be searched
    subpixel_size : int
                    An odd integer that defines the window size used to compute
                    the moments
    max_scaler : float
                 The percentage offset to apply to the delta between the maximum
                 correlation and the maximum edge correlation.
    metric : object
             The function to be used to perform the template based matching
             Options: {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED}
             In testing the first two options perform significantly better with Apollo data.

    Returns
    -------
    x : float
        The x offset
    y : float
        The y offset
    max_corr : float
               The strength of the correlation in the range [-1, 1].
    """

    # Apply the pixel scale template matcher
    result = cv2.matchTemplate(image, template, method=metric)

    # Find the 'best' correlation
    if metric == cv2.TM_SQDIFF or metric == cv2.TM_SQDIFF_NORMED:
        y, x = np.unravel_index(np.argmin(result, axis=None), result.shape)
    else:
        y, x = np.unravel_index(np.argmax(result, axis=None), result.shape)
    max_corr = result[(y,x)]
    
    # Get the area around the best correlation; the surface
    upper = int(2 + floor(subpixel_size / 2))
    lower = upper - 1
    # x, y are the location of the upper left hand corner of the template in the image
    area = result[y-lower:y+upper,
                  x-lower:x+upper]

    # If the area is not square and large enough, this method should fail
    if area.shape != (subpixel_size+2, subpixel_size+2):
        log.warning("Max correlation is too close to the boundary.")
        return None, None, 0, None

    cmass = center_of_mass(area)
    subpixel_y_shift = subpixel_size - 1 - cmass[0]
    subpixel_x_shift = subpixel_size - 1 - cmass[1]
    
    # Apply the subpixel shift to the whole pixel shifts computed above
    y += subpixel_y_shift
    x += subpixel_x_shift
    
    # Compute the idealized shift (image center)
    ideal_y = image.shape[0] / 2
    ideal_x = image.shape[1] / 2
    
    #Compute the shift from the template upper left to the template center
    y += (template.shape[0] / 2)
    x += (template.shape[1] / 2)
    
    x -= ideal_x
    y -= ideal_y
    
    return x, y, max_corr, result

def pattern_match(template, image, upsampling=16, metric=cv2.TM_CCOEFF_NORMED):
    """
    Call an arbitrary pattern matcher using a subpixel approach where the template and image
    are upsampled using a third order polynomial.

    This function assumes that the image center (0,0) and template center (0,0) are the a priori
    coordinates. This function returns the offset to the center of the template such that the 
    template is brought into alignment with the image.

    For example, if the image to be searched appears as follow:

    0, 0, 0, 0, 0, 0, 0, 1, 0
    0, 0, 0, 0, 0, 0, 0, 1, 0
    1, 1, 1, 0, 0, 0, 0, 1, 0
    0, 1, 0, 0, 0, 0, 0, 0, 0
    0, 1, 0, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 1, 1, 1
    0, 1, 1, 1, 0, 0, 1, 0, 1
    0, 1, 0, 1, 0, 0, 1, 0, 1
    0, 1, 1, 1, 0, 0, 1, 0, 1
    0, 0, 0, 0, 0, 0, 1, 1, 1

    and the template image is: 

    1, 1, 1
    0, 1, 0
    0, 1, 0 

    the expected shift from the center of the search image to have template image align is -3 in
    the y-direction and -3 in the x-direction. Conversely, if the search image is:

    1, 1, 1
    1, 0, 1
    1, 0, 1
    1, 0, 1
    1, 1, 1         

    the expected shift is 4 in the y-direction and 3 in the x-direction.

    Parameters
    ----------
    template : ndarray
               The input search template used to 'query' the destination
               image
    image : ndarray
            The image or sub-image to be searched
    upsampling : int
                 The multiplier to upsample the template and image.
    func : object
           The function to be used to perform the template based matching
           Options: {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED}
           In testing the first two options perform significantly better with Apollo data.

    Returns
    -------
    x : float
        The x offset of the template center the image center, the inverse of this shift will 
        need to be applied to the template as a correction.
    y : float
        The y offset of the template center to the image center, the inverse of this shift
        will need to be applied to the template as a correction.
    max_corr : float
               The strength of the correlation in the range [-1, 1].
    result : ndarray
             (m,n) correlation matrix showing the correlation for all tested coordinates. The
             maximum correlation is reported where the upper left hand corner of the template
             maximally correlates with the image.
    """
    if upsampling < 1:
        raise ValueError

    # Fit a 3rd order polynomial to upsample the images
    if upsampling != 1:
        u_template = img_as_float32(rescale(template, upsampling, order=3, mode='edge'))
        u_image = img_as_float32(rescale(image, upsampling, order=3, mode='edge'))
    else:
        u_template = template
        u_image = image
    corrmap = cv2.matchTemplate(u_image, u_template, method=metric)
    
    if metric == cv2.TM_SQDIFF or metric == cv2.TM_SQDIFF_NORMED:
        matched_y, matched_x = np.unravel_index(np.argmin(corrmap), corrmap.shape)
    else:
        matched_y, matched_x = np.unravel_index(np.argmax(corrmap), corrmap.shape)

    max_corr = corrmap[matched_y, matched_x]
    
    assert max_corr == np.max(corrmap)

    shift_x = (matched_x - ((corrmap.shape[1]-1)/2)) / upsampling
    shift_y = (matched_y - ((corrmap.shape[0]-1)/2)) / upsampling

    return -shift_x, -shift_y , max_corr, corrmap
