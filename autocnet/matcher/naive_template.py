import logging
from math import floor

import cv2
import numpy as np

from scipy.ndimage import center_of_mass
from skimage.transform import rescale
from skimage.util import img_as_float32
try:
    from image_registration import chi2_shift
except:
    chi2_shift = None


log = logging.getLogger(__name__)

def _template_match(image, template, metric):
    template = img_as_float32(template)
    image = img_as_float32(image)
    
    # If image is WxH and templ is wxh  , then result is (W-w)+1, (H-h)+1 .
    w, h = template.shape[::-1]
    corrmap = cv2.matchTemplate(image, template, method=metric)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrmap)
    if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        max_corr = min_val
    else:
        top_left = max_loc
        max_corr = max_val
    # This is the location in the image where the match has occurred.
    # I need the shift in the template needed to get into alignment (so the inverse!)
    # Need to take the initial location in the image and diff the updated location.
    matched_x = top_left[0] + w//2
    matched_y = top_left[1] + h//2
    assert max_corr == np.max(corrmap)
    return matched_x, matched_y, max_corr, corrmap

def pattern_match_chi2(image, template, usfac=16):
    assert image.shape == template.shape

    # Swapped so that we get the adjustment to the image to match the template
    # like the other matchers.
    if chi2_shift is None:
        raise ValueError("chi2_shift function is not defined. You need to install the 'image_registration' package with 'pip install image_registration'.")

    dx, dy, err_x, err_y = chi2_shift(image, template, return_error=True, upsample_factor=usfac, boundary='constant')
    shape_y, shape_x = np.array(template.shape)//2
    err =  (err_x + err_y) / 2.
    return shape_x-dx, shape_y-dy, err, None


def pattern_match_autoreg(image, template, subpixel_size=5, metric=cv2.TM_CCOEFF_NORMED, upsampling=16):
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

    x, y, max_corr, corrmap = _template_match(image, template, metric)
   
    maxy, maxx = np.unravel_index(corrmap.argmax(), corrmap.shape)
    area = corrmap[maxy - subpixel_size:maxy + subpixel_size + 1,
                   maxx - subpixel_size:maxx + subpixel_size + 1]

    # # If the area is not square and large enough, this method should fail
    # if area.shape != (subpixel_size+2, subpixel_size+2):
    #     raise Exception("Max correlation is too close to the boundary.")
    #     # return None, None, 0, None

    cmass = center_of_mass(area)

    subpixel_y_shift = subpixel_size - 1 - cmass[0]
    subpixel_x_shift = subpixel_size - 1 - cmass[1]
    
    # Apply the subpixel shift to the whole pixel shifts computed above
    y += subpixel_y_shift
    x += subpixel_x_shift
    return x, y, max_corr, corrmap

def pattern_match(image, template, upsampling=8, metric=cv2.TM_CCOEFF_NORMED):
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
        u_template = rescale(template, upsampling, order=3, mode='edge')
        u_image = rescale(image, upsampling, order=3, mode='edge')
    else:
        u_template = template
        u_image = image
 
    # new_x, new_y is the updated center location in the image
    new_x, new_y, max_corr, corrmap = _template_match(u_image, u_template, metric)
    new_x /= upsampling
    new_y /= upsampling

    return new_x, new_y, max_corr, corrmap
