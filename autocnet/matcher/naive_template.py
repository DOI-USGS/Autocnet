from math import floor
import cv2
import numpy as np
from scipy.ndimage.interpolation import zoom
from skimage.transform import rescale
from scipy.ndimage.measurements import center_of_mass


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
        print("Max correlation is too close to the boundary.")
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

def pattern_match(template, image, upsampling=8, metric=cv2.TM_CCOEFF_NORMED, error_check=False):
    """
    Call an arbitrary pattern matcher using a subpixel approach where the template and image
    are upsampled using a third order polynomial.

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
    error_check : bool
                  If True, also apply a different matcher and test that the values
                  are not too divergent.  Default, False.
    Returns
    -------
    x : float
        The x offset
    y : float
        The y offset
    max_corr : float
               The strength of the correlation in the range [-1, 1].
    result : ndarray
             (m,n) correlation matrix showing the correlation for all tested coordinates. The
             maximum correlation is reported when the upper left hand corner of the template
             maximally correlates with the image.
    """
    if upsampling < 1:
        raise ValueError

    # Fit a 3rd order polynomial to upsample the images
    if upsampling != 1:
        u_template = rescale(template, upsampling, order=1, mode='edge', preserve_range=True)
        u_image = rescale(image, upsampling, order=1, mode='edge', preserve_range=True)
    else:
        u_template = template
        u_image = image

    result = cv2.matchTemplate(u_image, u_template, method=metric)

    _, max_corr, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if metric == cv2.TM_SQDIFF or metric == cv2.TM_SQDIFF_NORMED:
        x = min_loc[0]
        y = min_loc[1]
    else:
        x = max_loc[0]
        y = max_loc[1]

    # Transform from the results array shape to the template shape
    x = x - (result.shape[1] - u_template.shape[1]) // 2
    y = y - (result.shape[0] - u_template.shape[0]) // 2

    # Recenter the origin from the upper left to the center of the template
    ideal_x = u_template.shape[1] // 2
    ideal_y = u_template.shape[0] // 2

    x -= ideal_x
    y -= ideal_y

    y /= upsampling
    x /= upsampling


    return -x, -y, max_corr, result

    # -1 because the returned results array is W-w+1 and H-h+1 in shape, 
    # where W, H are the width and height of the image and w,h are the 
    # width and height of the template

    print(u_template.shape, u_image.shape, result.shape, x,y)
    print(u_image.shape, u_template.shape, max_loc)


    # the max_loc array is of shape W-w+1, H-h+1, where W, H are the width
    # and height of the image and w,h are the width and height of the template

    print(x / upsampling - 1, y / upsampling - 1)


    # Compute the idealized shift (image center)
    ideal_y = (u_image.shape[0] + 1) / 2. - 0.5
    ideal_x = (u_image.shape[1] + 1) / 2. - 0.5

    # Compute the shift from template upper left to template center
    y += (u_template.shape[0] / 2.)
    x += (u_template.shape[1] / 2.)

    # 
    x = (x - ideal_x) / upsampling
    y = ((y - ideal_y) / upsampling) + 1
    print(x, y)
    return x, y, max_corr, result
