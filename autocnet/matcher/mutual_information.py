from math import floor
from autocnet.transformation.roi import Roi
import numpy as np

from scipy.ndimage.measurements import center_of_mass
import skimage.transform as tf

def mutual_information(reference_arr, moving_arr, **kwargs):
    """
    Computes the correlation coefficient between two images using a histogram
    comparison (Mutual information for joint histograms). The corr_map coefficient
    will be between 0 and 4

    Parameters
    ----------

    reference_arr : ndarray
                    First image to use in the histogram comparison
    
    moving_arr : ndarray
                   Second image to use in the histogram comparison
    
    
    Returns
    -------

    : float
      Correlation coefficient computed between the two images being compared
      between 0 and 4

    See Also
    --------
    numpy.histogram2d : for the kwargs that can be passed to the comparison
    """
   
    if np.isnan(reference_arr).any() or np.isnan(moving_arr).any():
        print('Unable to process due to NaN values in the input data')
        return
    
    if reference_arr.shape != moving_arr.shape:
        print('Unable compute MI. Image sizes are not identical.')
        return

    hgram, x_edges, y_edges = np.histogram2d(reference_arr.ravel(),moving_arr.ravel(), **kwargs)

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def mutual_information_match(moving_roi,
                             reference_roi, 
                             affine=tf.AffineTransform(), 
                             subpixel_size=3,
                             func=None, **kwargs):
    """
    Applies the mutual information matcher function over a search image using a
    defined template


    Parameters
    ----------
    moving_roi : roi 
                 The input search template used to 'query' the destination
                 image

    reference_roi : roi
              The image or sub-image to be searched

    subpixel_size : int
                    Subpixel area size to search for the center of mass
                    calculation

    func : function
           Function object to be used to compute the histogram comparison

    Returns
    -------
    new_affine :AffineTransform
                The affine transformation

    max_corr : float
               The strength of the correlation in the range [0, 4].

    corr_map : ndarray
               Map of corrilation coefficients when comparing the template to
               locations within the search area
    """
    reference_template = reference_roi.clip()
    moving_image = moving_roi.clip(affine)

    if func == None:
        func = mutual_information

    image_size = moving_image.shape
    template_size = reference_template.shape

    y_diff = image_size[0] - template_size[0]
    x_diff = image_size[1] - template_size[1]

    max_corr = -np.inf
    corr_map = np.zeros((y_diff+1, x_diff+1))
    for i in range(y_diff+1):
        for j in range(x_diff+1):
            sub_image = moving_image[i:i+template_size[1],  # y
                                j:j+template_size[0]]  # x
            corr = func(sub_image, reference_template, **kwargs)
            if corr > max_corr:
                max_corr = corr
            corr_map[i, j] = corr

    y, x = np.unravel_index(np.argmax(corr_map, axis=None), corr_map.shape)

    upper = int(2 + floor(subpixel_size / 2))
    lower = upper - 1
    area = corr_map[y-lower:y+upper,
                    x-lower:x+upper]

    # Compute the y, x shift (subpixel) using scipys center_of_mass function
    cmass  = center_of_mass(area)
    if area.shape != (subpixel_size + 2, subpixel_size + 2):
        return  None, 0, None
        

    subpixel_y_shift = subpixel_size - 1 - cmass[0]
    subpixel_x_shift = subpixel_size - 1 - cmass[1]
    y = abs(y - (corr_map.shape[1])/2)
    x = abs(x - (corr_map.shape[0])/2)
    y += subpixel_y_shift
    x += subpixel_x_shift
    new_affine = AffineTransform(translation=(-x, -y))
    return new_affine, np.max(max_corr), corr_map