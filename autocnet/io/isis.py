# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import logging

import numpy as np

try:
    import kalasiris as isis
except Exception as exception:
    from autocnet.utils.utils import FailedImport
    isis = FailedImport(exception)

log = logging.getLogger(__name__)

isis2np_types = {
        "UnsignedByte" : "uint8",
        "SignedWord" : "int16",
        "Double" : "float64",
        "Real" : "float32"
}

np2isis_types = {v: k for k, v in isis2np_types.items()}

def get_isis_special_pixels(arr):
    """
    Returns coordinates of any ISIS no data pixels. Essentially, 
    np.argwhere results of where pixels match ISIS special 
    data types (NIRs, NHRs, HIS, HRS, NULLS).

    Parameters
    ----------
    arr : np.array 
          Array to find special pixels in 
    
    Returns
    -------
    : sp
      np.array of coordinates in y,x format containing special pixel coordinates

    """
    isis_dtype = np2isis_types[str(arr.dtype)]
    sp_pixels = getattr(isis.specialpixels, isis_dtype)

    null = np.argwhere(arr==sp_pixels.Null)
    lrs = np.argwhere(arr==sp_pixels.Lrs)
    lis = np.argwhere(arr==sp_pixels.Lis)
    his = np.argwhere(arr==sp_pixels.His)
    hrs = np.argwhere(arr==sp_pixels.Hrs)
    sp = np.concatenate((null, lrs, lis, his, hrs))

    return sp
