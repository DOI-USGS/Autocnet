import numpy as np

def is_valid_lroc_polar_image(roi_array, 
                   include_var=True, 
                   include_mean=False,
                   include_std=False):
    """
    Checks if a numpy array representing an ROI from an lorc polar image is valid.
    Can check using variance, mean, and standard deviation, baed on user input. 
    It is highly encouraged that at the very least the variance check is used.

    Parameters
    __________
    roi_array : np.array
        A numpy array representing a ROI from an image, meaning the values are pixels
    include_var : bool
        Choose whether to filter images based on variance. Default True.
    include_mean : bool
        Choose whether to filter images based on mean. Default True.
        Goal is to get rid of overally dark images.
    include_std : bool
        Choose whether to filter images based on standard deviation. Default True.
        Goal is to get rid of overally saturated images.
    
    Returns
    _______
    is_valid : bool
        Returns True is passes the checks, returns false otherwise.
    """
    functions = []

    if include_var:
        # Get rid of super bad images
        var_func = lambda x : False if np.var(roi_array) == 0 else True
        functions.append(var_func)
    if include_mean:
        # Get rid of overally dark images
        mean_func = lambda x : False if np.mean(roi_array) < 0.0005 else True
        functions.append(mean_func)
    if include_std:
        # Get rid over overally saturated images
        std_func = lambda x : False if np.std(roi_array) > 0.001 else True
        functions.append(std_func)

    return all(func(roi_array) for func in functions)

def is_valid_lroc_image(roi_array, include_var=True):
    """
    Checks if a numpy array representing an ROI from an lroc image is valid.
    Can check using variance, based on user input. 
    It is highly encouraged that the variance check is used.

    Parameters
    __________
    roi_array : np.array
        A numpy array representing a ROI from an image, meaning the values are pixels
    include_var : bool
        Choose whether to filter images based on variance. Default True.
    
    Returns
    _______
    is_valid : bool
        Returns True is passes the checks, returns false otherwise.
    """
    functions = []

    if include_var:
        # Get rid of super bad images
        var_func = lambda x : False if np.var(roi_array) == 0 else True
        functions.append(var_func)

    return all(func(roi_array) for func in functions)