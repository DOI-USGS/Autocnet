import logging
from math import floor

import numpy as np
from skimage import transform as tf

from autocnet.io.geodataset import AGeoDataset

log = logging.getLogger(__name__)


class Roi():
    """
    Region of interest (ROI) object that is a sub-image taken from
    a larger image or array. This object supports transformations
    between the image coordinate space and the ROI coordinate
    space.

    Attributes
    ----------
    data : object
           A plio GeoDataset object

    x : float
        The x coordinate in image space

    y : float
        The y coordinate in image space

    size_x : int
             1/2 the total ROI width in pixels

    size_y : int
             1/2 the total ROI height in pixels

    ndv : float
          An optional no data value override to set a custom no data value on the ROI.

    center : tuple
             The x,y coordinates as a tuple.

    affine : object
             a scikit image affine transformation object that is applied when clipping. The default,
             identity matrix results in no transformation.
    """
    def __init__(self, data, x, y, size_x=200, size_y=200, ndv=None, ndv_threshold=0.5, buffer=5):
        if not isinstance(data, AGeoDataset):
            raise TypeError('Error: data object must be an autocnet AGeoDataset')
        self.data = data
        self.x = x
        self.y = y
        self.size_x = size_x
        self.size_y = size_y
        self.ndv = ndv
        self._ndv_threshold = ndv_threshold
    
    @property
    def center(self):
        return (self.x, self.y)
    
    @property
    def clip_center(self):
        return (self.size_x - 0.5, 
                self.size_y - 0.5)

    @property
    def x(self):
        return self._whole_x + self._remainder_x

    @x.setter
    def x(self, x):
        self._whole_x = floor(x)
        self._remainder_x = x - self._whole_x

    @property
    def y(self):
        return self._whole_y + self._remainder_y

    @y.setter
    def y(self, y):
        self._whole_y = floor(y)
        self._remainder_y = y - self._whole_y

    @property
    def ndv_threshold(self):
        return self._ndv_threshold

    @ndv_threshold.setter
    def ndv_threshold(self, threshold):
        self._ndv_threshold = threshold

    @property
    def ndv(self):
        """
        The no data value of the ROI. Used by the is_valid
        property to determine if the ROI contains any null
        pixels.
        """
        if hasattr(self.data, 'no_data_value'):
            self._ndv = self.data.no_data_value
        return self._ndv

    @ndv.setter
    def ndv(self, ndv):
        self._ndv = ndv

    @property
    def size_x(self):
        return self._size_x

    @size_x.setter
    def size_x(self, size_x):
        if not isinstance(size_x, int):
            raise TypeError(f'size_x must be type integer, not {type(size_x)}')
        self._size_x = size_x

    @property
    def size_y(self):
        return self._size_y

    @size_y.setter
    def size_y(self, size_y):
        if not isinstance(size_y, int):
            raise TypeError(f'size_y must be type integer, not {type(size_y)}')
        self._size_y = size_y

    @property
    def image_extent(self):
        """
        In full image space, this method computes the valid
        pixel indices that can be extracted.
        """
        left_x = self._whole_x - self.size_x
        right_x = self._whole_x + self.size_x
        top_y = self._whole_y - self.size_y
        bottom_y = self._whole_y + self.size_y

        return [left_x, right_x, top_y, bottom_y]

    def clip_coordinate_to_roi_coordinate(self, xy):
        """
        Take a passed coordinate in an array clipped from the ROI 
        and return the coordinate in ROI reference frame.

        Parameters
        ----------
        xy : iterable
            The (x,y) coordinate pair to be transformed.

        Returns
        -------
        xy_in_image_space : iterable
                           The transformed xy in ROI reference frame
        """
        clip_affine = (tf.SimilarityTransform(translation=((-self.size_x, -self.size_y))) + \
                        (self.affine + \
                        tf.SimilarityTransform(translation=(self.size_x, self.size_y))))
        transformed = clip_affine.inverse(xy)
        if len(transformed) == 1:
            return transformed[0]
        else:
            return transformed

    def roi_coordinate_to_image_coordinate(self, xy):
        """
        Take a passed coordinate in the ROI reference frame and
        transform it into the image reference frame.

        Parameters
        ----------
        xy : iterable
            The (x,y) coordinate pair to be transformed.

        Returns
        -------
        xy_in_image_space : iterable
                           The transformed xy in full image reference frame
        """
        roi2image = tf.SimilarityTransform(translation=(self.x-self.size_x, self.y-self.size_y))
        transformed = roi2image(xy)
        if len(transformed) == 1:
            return transformed[0]
        else:
            return transformed

    def clip_coordinate_to_image_coordinate(self, xy):
        """
        Take a passed coordinate in an array clipped from the ROI 
        and return the coordinate in full images reference frame.

        Parameters
        ----------
        xy : iterable
            The (x,y) coordinate pair to be transformed.

        Returns
        -------
        xy_in_image_space : iterable
                           The transformed xy in full image reference frame
        """
        clipped2roi = self.clip_coordinate_to_roi_coordinate(xy)
        roi2image = self.roi_coordinate_to_image_coordinate(clipped2roi)
        if len(roi2image) == 1:
            return roi2image[0]
        else:
            return roi2image

    def _compute_valid_read_range(self, buffer):
        """
        Compute the valid range of pixels that can be read
        from the ROI's geodataset object.
        """
        min_x = self._whole_x - self.size_x - buffer
        min_y = self._whole_y - self.size_y - buffer
        x_read_length = (self.size_x * 2) + (buffer * 2)
        y_read_length = (self.size_y * 2) + (buffer * 2)
        log.debug('XY ROI read start and read length: ', min_x, min_y, x_read_length, y_read_length)
        # series of checks to make sure all pixels inside image limits
        raster_xsize, raster_ysize = self.data.raster_size
        if min_x < 0 or min_y < 0 or min_x+x_read_length > raster_xsize or min_y+y_read_length > raster_ysize:
            raise IndexError('Image coordinates plus read buffer are outside of the available data. Please select a smaller ROI and/or a smaller read buffer.')

        return [min_x, min_y, x_read_length, y_read_length]
    
    def clip(self, size_x=None, size_y=None, affine=tf.AffineTransform(), buffer=0, dtype=None, warp_mode="constant", coord_mode="constant", min_size=24):
        """
        Compatibility function that makes a call to the array property.
        Warning: The dtype passed in via this function resets the dtype attribute of this
        instance.
        Parameters
        ----------
        size_x : int
             1/2 the total ROI width in pixels

        size_y : int
             1/2 the total ROI height in pixels

        dtype : str
                The datatype to be used when reading the ROI information if the read
                occurs through the data object using the read_array method. When using
                this object when the data are a numpy array the dtype has not effect.

        affine : object
                 A scikit image AffineTransform object that is used to warp the clipped array.
        
        buffer : int
                 The number of pixels to buffer the read by. The buffer argument is used to ensure
                 that the final ROI does not have no data values in it due to reprojection by the 
                 affine.

        mode : string
               An optional mode to be used when affinely transforming the clipped array. Ideally,
               a sufficiently large buffer has been specified on instiantiation so that the mode
               used is not visible in the final warped array. See scikitimage.transform.warp for
               for possible values.
        Returns
        -------
         : ndarray
           The array attribute of this object.
        """
        self.affine = affine
        if size_x:
            self.size_x = size_x
        if size_y:
            self.size_y = size_y
        pixels = self._compute_valid_read_range(buffer)

        data = self.data.read_array(pixels=pixels, dtype=dtype)

        # Create a data centered transformation (scikit-image defaults
        # to warping about the upper-left origin).
        roi_center = (np.array(data.shape))[::-1] / 2.  # Where center is a zero based index of the image center
        self.subwindow_affine = (tf.SimilarityTransform(translation=(-roi_center)) + \
                                (affine + \
                                tf.SimilarityTransform(translation=roi_center)))

        warped_data = tf.warp(data,
                              self.subwindow_affine.inverse,
                              order=3,
                              preserve_range=True)
        # If a buffer was passed, clip the returned data to remove the buffer.
        # This is important when the affine transformation might introduce no
        # data values about the edges that will impact template matching.
        if buffer:
            return warped_data[buffer:-buffer,buffer:-buffer]
        else:
            return warped_data
