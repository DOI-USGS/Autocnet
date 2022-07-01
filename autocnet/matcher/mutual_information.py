from math import floor
from autocnet.transformation.roi import Roi
import numpy as np

from scipy.ndimage.measurements import center_of_mass
import skimage.transform as tf

import logging 
# setup logging file
log = logging.getLogger(__name__)


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
        log.warning('Unable compute MI. Image sizes are not identical.')        
        return
    
    if reference_arr.shape != moving_arr.shape:
        log.warning('Unable compute MI. Image sizes are not identical.')
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

