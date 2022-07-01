from math import floor
import numpy as np
from plio.io.io_gdal import GeoDataset
import scipy.ndimage as ndimage

from skimage import transform as tf
from skimage.util import img_as_float32

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

    buffer : int
             An integer number of pixels to buffer the read number of pixels from a GeoDataset. This parameter
             can be used to ensure that an affinely warped ROI contains only valid data. A buffer too small can
             result in no data on one or more edges.

    clip_center : tuple
                  on instantiation, set to (). When clip is called and the clipped_array
                  variable is set, the clip_center is set to the center of the, potentially
                  affine transformed, cliped_array.

    clipped_array : ndarray
                    After calling the clip method, this is the resulting clipped, and possibly affinely warped
                    data array.

    warped_array_center : tuple
                          During clipping, if an affine transformation is provided, the clipped array is warped. The center
                          of the warped array is not the same as the center of the clipped array. This attribute is the
                          affine transformation of the clip_center.
    affine : object
             a scikit image affine transformation object that is applied when clipping. The default,
             identity matrix results in no transformation.
    """
    def __init__(self, data, x, y, size_x=200, size_y=200, ndv=None, ndv_threshold=0.5, buffer=5, affine=tf.AffineTransform()):
        if not isinstance(data, GeoDataset):
            raise TypeError('Error: data object must be a plio GeoDataset')
        self.data = data
        self.x = x
        self.y = y
        self.size_x = size_x
        self.size_y = size_y
        self.ndv = ndv
        self._ndv_threshold = ndv_threshold
        self.buffer = buffer
        self.clip_center = ()
        self.affine = affine
        self._clipped_array = None

    @property
    def center(self):
        return (self.x, self.y)

    @property
    def clip_center(self):
        if not getattr(self, '_clip_center', None):
            self.clip()
        return self._clip_center

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, affine=tf.AffineTransform()):
        self._affine = affine

    @property
    def x(self):
        return self._whole_x + self._remainder_x

    @x.setter
    def x(self, x):
        self._whole_x = floor(x)
        self._remainder_x = x - self._whole_x
        return self._whole_x + self._remainder_x

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
        raster_size = self.data.raster_size

        # what is the extent that can actually be extracted?
        left_x = self._whole_x - self.size_x
        right_x = self._whole_x + self.size_x
        top_y = self._whole_y - self.size_y
        bottom_y = self._whole_y + self.size_y

        return [left_x, right_x, top_y, bottom_y]


    @property
    def is_valid(self):
        """
        True if all elements in the clipped ROI are valid, i.e.,
        no null pixels (as defined by the no data value (ndv)) are
        present.
        """
        if self.ndv == None:
            return 
        if len(self._clipped_array) == 0:
            return False
        # Check if we have any ndv values this will return an inverted array
        # where all no data values are true, we need to then invert the array
        # and return the all result. This ensures that a valid array will return
        # True
        return np.invert(np.isclose(self.ndv, self.clipped_array)).all()


    @property
    def variance(self):
        return np.var(self.clipped_array)

    @property
    def clipped_array(self):
        """
        The clipped array associated with this ROI.
        """
        if not hasattr(self, "_clipped_array"):
            self.clip()
        return self._clipped_array

    def clip_coordinate_to_image_coordinate(self, x, y):
        """
        Take a passed coordinate in a clipped array from an ROI and return the coordinate
        in the full image.

        Parameters
        ----------
        x : float
            The x coordinate in the clipped array to be transformed into full image coordinates

        y : float
            The y coordinate in the clipped array to be transformed into full image coordinates

        Returns
        -------
        x_in_image_space : float
                           The transformed x in image coordinate space

        y_in_imag_space : float
                          The transformed y in image coordinate space
        """
        x_in_affine_space = x + self._clip_start_x
        y_in_affine_space = y + self._clip_start_y

        x_in_clip_space, y_in_clip_space = self.affine((x_in_affine_space,
                                                                y_in_affine_space))[0]

        x_in_image_space = x_in_clip_space + self._roi_x_to_clip_center
        y_in_image_space = y_in_clip_space + self._roi_y_to_clip_center

        return x_in_image_space, y_in_image_space

    def clip(self, size_x=None, size_y=None, affine=None, dtype=None, mode="reflect"):
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
        if size_x:
            self.size_x = size_x
        if size_y:
            self.size_y = size_y

        min_x = self._whole_x - self.size_x - self.buffer
        min_y = self._whole_y - self.size_y - self.buffer
        x_read_length = (self.size_x * 2) + 1 + (self.buffer * 2)
        y_read_length = (self.size_y * 2) + 1 + (self.buffer * 2)

        if min_x < 1:
            min_x = 1
        if min_y < 1:
            min_y = 1

        pixels = [min_x, min_y, x_read_length, y_read_length]
        if (np.asarray(pixels) < 0).any():
            raise IndexError('Image coordinates plus read buffer are outside of the available data. Please select a smaller ROI and/or a smaller read buffer.')

        # This data is an nd-array that is larger than originally requested, because it may be affine warped.
        data = self.data.read_array(pixels=pixels, dtype=dtype)

        data_center = np.array(data.shape[::-1]) / 2.  # Location within a pixel
        self._roi_x_to_clip_center = self.x - data_center[0]
        self._roi_y_to_clip_center = self.y - data_center[1]

        if affine:
            self.affine = affine
            # The cval is being set to the mean of the array,
            warped_data = tf.warp(data,
                         self.affine,
                         order=3,
                         mode='reflect')

            self.warped_array_center = self.affine.inverse(data_center)[0]

            # Warped center coordinate - offset from pixel center to pixel edge - desired size
            self._clip_start_x = self.warped_array_center[0] - 0.5 - self.size_x
            self._clip_start_y = self.warped_array_center[1] - 0.5 - self.size_y

            # Now that the whole pixel array has been warped, interpolate the array to align pixel edges
            xi = np.linspace(self._clip_start_x,
                             self._clip_start_x + (self.size_x * 2) + 1,
                             (self.size_x * 2) + 1)
            yi = np.linspace(self._clip_start_y,
                             self._clip_start_y + (self.size_y * 2) + 1,
                             (self.size_y * 2) + 1)

            # the xi, yi are intentionally handed in backward, because the map_coordinates indexes column major
            pixel_locked = ndimage.map_coordinates(warped_data,
                                        np.meshgrid(yi, xi, indexing='ij'),
                                        mode=mode,
                                        order=3)

            self._clip_center = (np.array(pixel_locked.shape)[::-1]) / 2.0

            self._clipped_array = img_as_float32(pixel_locked)
        else:

            # Now that the whole pixel array has been read, interpolate the array to align pixel edges
            xi = np.linspace(self._remainder_x,
                            ((self.buffer*2) + self._remainder_x + (self.size_x*2)),
                            (self.size_x*2+1)+(self.buffer*2))
            yi = np.linspace(self._remainder_y,
                            ((self.buffer*2) + self._remainder_y + (self.size_y*2)),
                            (self.size_y*2+1)+(self.buffer*2))

            # the xi, yi are intentionally handed in backward, because the map_coordinates indexes column major
            pixel_locked = ndimage.map_coordinates(data,
                                        np.meshgrid(yi, xi, indexing='ij'),
                                        mode=mode,
                                        order=3)

            if self.buffer != 0:
                pixel_locked = pixel_locked[self.buffer:-self.buffer,
                                            self.buffer:-self.buffer]
            self._clip_center = tuple(np.array(pixel_locked.shape)[::-1] / 2.)
            self.warped_array_center = self.clip_center
            self._clipped_array = img_as_float32(pixel_locked)
