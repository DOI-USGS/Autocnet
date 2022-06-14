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

    left_x : int
             The left pixel coordinate in image space

    right_x : int
              The right pixel coordinage in image space

    top_y : int
            The top image coordinate in image space

    bottom_y : int
               The bottom image coordinate in imge space
    """
    def __init__(self, data, x, y, ndv=None, ndv_threshold=0.5, buffer=5):
        self.data = data
        self.x = x
        self.y = y
        self.ndv = ndv
        self._ndv_threshold = ndv_threshold
        self.buffer = buffer

    @property
    def center(self):
        return (self.x, self.y)

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
        ie = self.image_extent
        return ((ie[1] - ie[0])-1)/2. + 0.5, ((ie[3]-ie[2])-1)/2. + 0.5

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


    def clip(self, size_x, size_y, affine=None, dtype=None, mode="reflect"):
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
        Returns
        -------
         : ndarray
           The array attribute of this object.
        """
        

        pixels = self.image_extent
        if (np.asarray(pixels) - self.buffer < 0).any():
            raise IndexError('Image coordinates plus read buffer are outside of the available data. Please select a smaller ROI and/or a smaller read buffer.')

        
        if isinstance(self.data, np.ndarray):
            data = self.data[pixels[2]-self.buffer:pixels[3]+1+self.buffer, 
                             pixels[0]-self.buffer:pixels[1]+1+self.buffer]
        else:
            # Have to reformat to [xstart, ystart, xnumberpixels, ynumberpixels]
            # TODO: I think this will result in an incorrect obj.center when the passed data is a GeoDataset
            '''pixels = [pixels[0]-self.buffer, 
                      pixels[2]-self.buffer, 
                      pixels[1]-pixels[0]+(self.buffer*2)+1, 
                      pixels[3]-pixels[2]+(self.buffer*2)+1]'''
            pixels = map(floor, [self.x-size_x, self.y-size_y, size_x*2+1, size_y*2+1])
            data = self.data.read_array(pixels=pixels, dtype=dtype)
            return data
        if affine:
            # The cval is being set to the mean of the array,
            af = tf.warp(data, 
                                   affine, #.inverse, 
                                   order=3, 
                                   mode='constant',
                                   cval=0.1)

            array_center =  (np.array(data.shape)[::-1] - 1) / 2.0
            rmatrix = np.linalg.inv(affine.params[0:2, 0:2])
            new_center = np.dot(rmatrix, array_center)
            
            af = af[floor(new_center[0])-self.size_y:floor(new_center[0])+self.size_y+1,
                      floor(new_center[1])-self.size_x:floor(new_center[1])+self.size_x+1]
            
            return af
        """if affine:
            # The cval is being set to the mean of the array,
            d2 = tf.warp(data, 
                        affine,# .inverse, 
                        order=3, 
                        mode=mode)
            
            if self.buffer != 0:
                pixel_locked = d2[self.buffer:-self.buffer, 
                                  self.buffer:-self.buffer]

                return img_as_float32(pixel_locked)
            return d2
        else:
            return data"""
        # Now that the whole pixel array has been read, interpolate the array to align pixel edges
        xi = np.linspace(self._remainder_x, 
                         ((self.buffer*2) + self._remainder_x + (self.size_x*2)), 
                         (self.size_x*2+1)+(self.buffer*2)) 
        yi = np.linspace(self._remainder_y, 
                         ((self.buffer*2) + self._remainder_y + (self.size_y*2)), 
                         (self.size_y*2+1)+(self.buffer*2))

        # the xi, yi are intentionally handed in backward, because the map_coordinates indexes column major
        # Maybe this operates in place?
        pixel_locked = ndimage.map_coordinates(data, 
                                       np.meshgrid(yi, xi, indexing='ij'),
                                       mode=mode,
                                       order=3)
   
        if affine:
            # The cval is being set to the mean of the array,
            pixel_locked = tf.warp(data, 
                                   affine.inverse, 
                                   order=3, 
                                   mode=mode)

        # UL, LR, C and then compute 

        if self.buffer != 0:
            pixel_locked = pixel_locked[self.buffer:-self.buffer, 
                                self.buffer:-self.buffer]
        # Ohh is buffer doing something here? Yeah, post clip, so we should be safe.
        # Buffers don't matter - wtf
        return img_as_float32(pixel_locked)
