from math import modf, floor
import numpy as np

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
    data : ndarray/object
           An ndarray or an object with a raster_size attribute

    x : float
        The x coordinate in image space

    y : float
        The y coordinate in image space

    size_x : int
             1/2 the total ROI width in pixels

    size_y : int
             1/2 the total ROI height in pixels

    left_x : int
             The left pixel coordinate in image space

    right_x : int
              The right pixel coordinage in image space

    top_y : int
            The top image coordinate in image space

    bottom_y : int
               The bottom image coordinate in imge space
    """
    def __init__(self, data, x, y, size_x=200, size_y=200, ndv=None, ndv_threshold=0.5, buffer=5):
        self.data = data
        self.x = x
        self.y = y
        self._affine = tf.AffineTransform(translation=(self._remainder_x, 
                                                       self._remainder_y))
        self.size_x = size_x
        self.size_y = size_y
        self.ndv = ndv
        self._ndv_threshold = ndv_threshold
        self.buffer = buffer

    @property
    def affine(self):
        """
        This affine sets the origin of the ROI to be (0,0).
        """
        return self._affine

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
        try:
            # Geodataset object
            raster_size = self.data.raster_size
        except:
            # Numpy array in y,x form
            raster_size = self.data.shape[::-1]

        # Should this modify (+-) and then round to whole pixel?

        # what is the extent that can actually be extracted?
        left_x = self._whole_x - self.size_x
        right_x = self._whole_x + self.size_x
        top_y = self._whole_y - self.size_y
        bottom_y = self._whole_y + self.size_y 

        #if left_x < 0 or top_y < 0 or right_x > raster_size[0] or bottom_y > raster_size[1]:
        #    raise IndexError(f"Input window size {(self.size_x, self.size_y)}) at center {(self.x, self.y)} is out of the image bounds") 

        return [left_x, right_x, top_y, bottom_y]

    @property
    def center(self):
        return (0,0)
        #ie = self.image_extent
        #return ((ie[1] - ie[0])-1)/2. + 0.5, ((ie[3]-ie[2])-1)/2. + 0.5

    @property
    def is_valid(self):
        """
        True if all elements in the clipped ROI are valid, i.e.,
        no null pixels (as defined by the no data value (ndv)) are
        present.
        """
        print(self.ndv)
        if self.ndv == None:
            return True
        return np.isclose(self.ndv,self.array).all()
        

    @property
    def variance(self):
        return np.var(self.array)

    @property
    def array(self):
        """
        The clipped array associated with this ROI.
        """
        return self.clip()


    def clip(self, affine=None, dtype=None, mode="constant"):
        """
        Compatibility function that makes a call to the array property.
        Warning: The dtype passed in via this function resets the dtype attribute of this
        instance.
        Parameters
        ----------
        dtype : str
                The datatype to be used when reading the ROI information if the read
                occurs through the data object using the read_array method. When using
                this object when the data are a numpy array the dtype has not effect.
        Returns
        -------
         : ndarray
           The array attribute of this object.
        """
        self.dtype = dtype

        pixels = self.image_extent
        
        if (np.asarray(pixels) - self.buffer < 0).any():
            raise IndexError('Image coordinates plus read buffer are outside of the available data. Please select a smaller ROI and/or a smaller read buffer.')

        if isinstance(self.data, np.ndarray): # 33x33 buffer = 3
            data = self.data[pixels[2]-self.buffer:pixels[3]+1+self.buffer,  #(10-3): (43 + 3) 39x39 of real data, non-interpolated 
                             pixels[0]-self.buffer:pixels[1]+1+self.buffer]
        else:
            # Have to reformat to [xstart, ystart, xnumberpixels, ynumberpixels]
            # TODO: I think this will result in an incorrect obj.center when the passed data is a GeoDataset
            pixels = [pixels[0]-self.buffer, 
                      pixels[2]-self.buffer, 
                      pixels[1]-pixels[0]+1+self.buffer, 
                      pixels[3]-pixels[2]+1+self.buffer]
            data = self.data.read_array(pixels=pixels)
        
        # Now that the whole pixel array has been read, interpolate the array to align pixel edges
        xi = np.linspace(self._remainder_x, 
                         (self.buffer*2) + self._remainder_x + (self.size_x*2), 
                         (self.size_x*2+1)+(self.buffer*2)) 
        yi = np.linspace(self._remainder_y, 
                         (self.buffer*2) + self._remainder_y + (self.size_y*2), 
                         (self.size_y*2+1)+(self.buffer*2))

        pixel_locked = ndimage.map_coordinates(data, 
                                       np.meshgrid(xi, yi, indexing='ij'),
                                       mode='constant',
                                       cval=0.0,
                                       order=3)

        if affine:
            # if array_to_warp.shape != ((self.size_y * 2) + 1, (self.size_x * 2) + 1):
            #     raise ValueError("Unable to enlarge Roi to apply affine transformation." +
            #                      f" Was only able to extract {array_to_warp.shape}, when " +
            #                      f"{((self.size_y * 2) + 1, (self.size_x * 2) + 1)} was asked for. Select, " +
            #                      "a smaller region of interest" )

            pixel_locked = tf.warp(pixel_locked, affine.inverse, order=3, mode=mode, cval=0)

        if self.buffer != 0:
            return img_as_float32(pixel_locked[self.buffer:-self.buffer, 
                                       self.buffer:-self.buffer])
        else:
            return img_as_float32(pixel_locked)
