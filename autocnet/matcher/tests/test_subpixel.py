from skimage import transform as tf
from scipy.ndimage.interpolation import rotate
from skimage import color 
from skimage import data

import pytest

import numpy as np
from imageio import imread

from autocnet.examples import get_path
import autocnet.matcher.subpixel as sp
from autocnet.transformation import roi

def rot(image, xy, angle):
    """
    This function rotates an image about the center and also computes the new
    pixel coordinates of the previous image center.


    This function is intentionally external to the AutoCNet API so that
    it can be used to generate test data without any dependence on AutoCNet.
    """
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    
    org = xy - org_center
    a = np.deg2rad(angle)  # negative so the rotation is CCW
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
                    -org[0]*np.sin(a) + org[1]*np.cos(a)])

    return im_rot, new, new+rot_center

@pytest.fixture
def iris_pair(): 
    angle = 200
    scale = 1.4
    shiftr = 30
    shiftc = 10

    image = color.rgb2gray(data.retina())
    translated = image[shiftr:, shiftc:]
    rotated = tf.rotate(translated, angle)
    rescaled = tf.rescale(rotated, scale)
    sizer, sizec = image.shape
    rts_image = rescaled[:sizer, :sizec] 
    return image, rts_image

@pytest.fixture
def apollo_subsets():
    # These need to be geodata sets or just use mocks...
    arr1 = imread(get_path('AS15-M-0295_SML(1).png'))[100:201, 123:224]
    arr2 = imread(get_path('AS15-M-0295_SML(2).png'))[235:336, 95:196]
    return arr1, arr2

@pytest.mark.parametrize("nmatches, nstrengths", [(10,1), (10,2)])
def test_prep_subpixel(nmatches, nstrengths):
    arrs = sp._prep_subpixel(nmatches, nstrengths=nstrengths)
    assert len(arrs) == 5
    assert arrs[2].shape == (nmatches, nstrengths)
    assert arrs[0][0] == 0

delta_xs = [.0, 1.0,  1.4, .6, 0.75, 0.99, 2.2]
delta_ys = [.0, 1.0, 1.4, .6, 0.75, 0.99, 2.2]
buffer = [10, 15]
rotation_angles = [0, -1, 1, -5.5, 5.5, -10.25, 10.25, -45, 45, -90, 90]
@pytest.mark.parametrize("delta_x", delta_xs)
@pytest.mark.parametrize("delta_y", delta_ys)
@pytest.mark.parametrize("rotation_angle", rotation_angles)
def test_subpixel_transformed_template(apollo_subsets, delta_x, delta_y, rotation_angle):
    reference_image = apollo_subsets[0]
    moving_image = apollo_subsets[0]

    # The reference image needs to be rotated if the moving image is going to be
    # artifically rotated and then a match attempted.

    x = 50
    y = 51
    x1 = 50 + delta_x  
    y1 = 51 + delta_y  

    # Artifically rotate the b array by an arbitrary rotation angle.
    rotated_array, new, (rx1, ry1) = rot(moving_image, (x1, y1), rotation_angle)
    _, new, expected = rot(moving_image, (x, y), rotation_angle) 
    
    # Since this is not testing the quality of the matcher given different
    # reference and moving image sizes, just hard code to be something large, 
    # and something a fair bit smaller.
    # A large buffer is needed because of the rotation angles that are approaching 45 degrees.
    ref_roi = roi.Roi(reference_image, x, y, 39, 39)
    buffer = 10
    moving_roi = roi.Roi(rotated_array, rx1, ry1, 21, 21, buffer=buffer)

    # Compute the affine transformation. This is how the affine is computed in the code, so replicated here.
    # If the AutoCNet code changes, this will highlight a breaking change.
    t_size = moving_roi.array.shape[:2][::-1]
    shift_y = (t_size[0] - 1 + (buffer*2)) / 2.
    shift_x = (t_size[1] - 1 + (buffer * 2)) / 2.

    tf_rotate = tf.AffineTransform(rotation=np.radians(rotation_angle))
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
    # The '+' overload that is '@' matrix multiplication is read or applied left to right.
    affine = tf_shift + (tf_rotate + tf_shift_inv)

    # Apply the subpixel matcher to get back an affine that is just translation to apply.
    new_affine, strength, corrmap = sp.subpixel_template(ref_roi, moving_roi, affine, upsampling=16)

    new_x, new_y = new_affine((moving_roi.x,
                               moving_roi.y))[0]
    
    assert pytest.approx(new_x, abs=1/16) == expected[0]
    assert pytest.approx(new_y, abs=1/16) == expected[1]
    

def test_estimate_logpolar_transform(iris_pair):
    img1, img2 = iris_pair 
    affine = sp.estimate_logpolar_transform(img1, img2) 

    assert pytest.approx(affine.scale, 0.1) == 0.71
    assert pytest.approx(affine.rotation, 0.1) == 0.34 
    assert pytest.approx(affine.translation[0], 0.1) == 283.68
    assert pytest.approx(affine.translation[1], 0.1) == -198.62 


"""def test_fourier_mellen(iris_pair):
    img1, img2 = iris_pair 

    ref_roi = roi.Roi(img1, img1.shape[1]/2, img1.shape[0]/2, 401, 401)
    moving_roi = roi.Roi(img2, img2.shape[1]/2, img2.shape[0]/2, 401, 401)
    nx, ny, error = sp.fourier_mellen(ref_roi, moving_roi, phase_kwargs = {"size":(401, 401)}) 
    
    assert pytest.approx(nx, 0.01) == 996.39 
    assert pytest.approx(ny, 0.01) ==  984.912 
    assert pytest.approx(error, 0.01) == 0.0422""" 


@pytest.mark.parametrize("convergence_threshold, expected", [(2.0, (50.52, 48.6, -9.5e-20))])
def test_iterative_phase(apollo_subsets, convergence_threshold, expected):
    reference_image = apollo_subsets[0]
    walking_image = apollo_subsets[1]
    image_size = (31, 31)

    ref_x, ref_y = reference_image.shape[0]/2, reference_image.shape[1]/2
    walk_x, walk_y = walking_image.shape[0]/2, walking_image.shape[1]/2

    reference_roi = roi.Roi(reference_image, ref_x, ref_y, size_x=image_size[0], size_y=image_size[1])
    walking_roi = roi.Roi(walking_image, walk_x, walk_y, size_x=image_size[0], size_y=image_size[1])
    affine, error, diffphase = sp.iterative_phase(reference_roi,
                                                  walking_roi,
                                                  convergence_threshold=convergence_threshold,
                                                  upsample_factor=100)
    ref_to_walk = affine.inverse((ref_x, ref_y))[0]
    assert ref_to_walk[0] == expected[0]
    assert ref_to_walk[1] == expected[1]
    if expected[2] is not None:
        assert pytest.approx(error,6) == expected[2]

@pytest.mark.parametrize("data, expected", [
    ((21,21), (10, 10)),
    ((20,20), (10,10))
])
def test_check_image_size(data, expected):
    assert sp.check_image_size(data) == expected

@pytest.mark.parametrize("x, y, x1, y1, image_size, expected",[
    (4, 3, 3, 2, (1,1), (3,2)),
    (4, 3, 3, 2, (2,2), (3,2)),  # Increase the search image size
    (4, 3, 3, 2, (2,2), (3,2)), # Increase the template size
    (4, 3, 2, 2, (2,2), (3,2)), # Move point in the x-axis
    (4, 3, 4, 3, (2,2), (3,2)), # Move point in the other x-direction
    (4, 3, 3, 3, (2,2), (3,2))  # Move point positive in the y-axis

])
def test_subpixel_phase_cooked(x, y, x1, y1, image_size, expected):
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 0, 0, 0, 0, 1, 0),
                           (0, 0, 0, 1, 1, 1, 0, 1, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 1, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 1, 0, 1, 0, 0, 1, 0, 1),
                           (0, 1, 1, 1, 0, 0, 1, 0, 1),
                           (0, 0, 0, 0, 0, 0, 1, 1, 1)), dtype=np.uint8)

    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 1, 1, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)

    reference_roi = roi.Roi(test_image, x, y, size_x=image_size[0], size_y=image_size[1], buffer=0)
    moving_roi = roi.Roi(t_shape, x1, y1, size_x=image_size[0], size_y=image_size[1], buffer=0)

    affine, metrics, _ = sp.subpixel_phase(reference_roi, moving_roi)
    dx, dy = affine((moving_roi.x, moving_roi.y))[0]

    assert dx == expected[0]
    assert dy == expected[1]
