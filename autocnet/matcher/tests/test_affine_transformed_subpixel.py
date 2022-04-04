import itertools

import pytest
import numpy as np
from skimage import transform as tf
from scipy.ndimage.interpolation import rotate

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
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new, new+rot_center


@pytest.fixture
def reference_image():
    # This is the reference image. The correct location for the letter 'T' in the test image
    # is 5.5, 4.5 (x, y; pixel center)
    test_image = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0),
                        (0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0),
                        (0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0),
                        (0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0)), dtype=np.uint8)
    return test_image

@pytest.fixture
def moving_image():
    # This is the moving image. The correct solution where the letter T in the moving image is
    # 7.5,5.5 (x,y; pixel center) 
    t_shape = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), dtype=np.uint8)
    return t_shape


delta_xs = [0, 0.5, -0.5, 1, -1, 1.5, -1.5]
delta_ys = [0, 0.5, -0.5, 1, -1, 1.5, -1.5]
rotation_angles = [0, 1, -1, -90, 90, -70, 70, -60, 60, -45.23, 45.23, -33, 33, -15.1, 15.1]

@pytest.mark.parametrize('delta_x', delta_xs)
@pytest.mark.parametrize('delta_y', delta_ys)
@pytest.mark.parametrize('rotation_angle', rotation_angles)
def test_canned_affine_transformations(reference_image, moving_image, delta_x, delta_y, rotation_angle):
    image_size = (4,4)
    template_size = (3,3)
    x = 5
    y = 5
    x1 = 7 + delta_x  # 7 is correct
    y1 = 5 + delta_y  # 5 is correct

    # Rotate the moving_image by an arbitrary amount for the test. Rotations are CCW.
    rotated_array, new, (rx1, ry1) = rot(moving_image, (x1, y1), rotation_angle)

    # Since the moving_image array is rotated, it is necessary to compute the updated 'expected'
    # correct answer.
    _, _, expected = rot(moving_image, (7, 5), rotation_angle)

    # Get the reference ROIs from the full images above. The reference is a straight clip.
    # The moving array is taken from the rotated array with x1, y1 being the updated x1, y1 coordinates
    # after rotation.
    ref_roi = roi.Roi(reference_image, x, y, *image_size)
    moving_roi = roi.Roi(rotated_array, rx1, ry1, *template_size)  #rx1, rx2 in autocnet would be the a priori coordinates. Yup!

    # Compute the affine transformation. This is how the affine is computed in the code, so replicated here.
    # If the AutoCNet code changes, this will highlight a breaking change.
    t_size = moving_roi.array.shape[:2][::-1]
    shift_y = (t_size[0] - 1) / 2.
    shift_x = (t_size[1] - 1) / 2.

    tf_rotate = tf.AffineTransform(rotation=np.radians(rotation_angle))
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
    # The '+' overload that is '@' matrix multiplication is read or applied left to right.
    trans = tf_shift + (tf_rotate + tf_shift_inv)

    # Pretend that the matcher has found a shift in the x and a shift in the y.
    matcher_shift_x = -delta_x
    matcher_shift_y = -delta_y
        
    # Apply the shift to the center of the moving roi to the center of the reference ROI in index space. One pixel == one index (unitless).
    new_affine_transformed_center_x = moving_roi.center[0] + matcher_shift_x  #Center is indices.
    new_affine_transformed_center_y = moving_roi.center[1] + matcher_shift_y

    # Invert the affine transformation of the new center. This result is plotted in the second figure as a red dot.
    inverse_transformed_affine_center_x, inverse_transformed_affine_center_y = trans.inverse((new_affine_transformed_center_x, new_affine_transformed_center_y))[0]
    #print(inverse_transformed_affine_center_x, inverse_transformed_affine_center_y)

    # Take the original x,y (moving_roi.x, moving_roi.y) and subtract the delta between the original ROI center and the newly computed center.
    new_x = moving_roi.x - (moving_roi.center[0] - inverse_transformed_affine_center_x) 
    new_y = moving_roi.y - (moving_roi.center[1] - inverse_transformed_affine_center_y)

    # If testing this in a notebook, uncomment for visualization.
    """fig, axes = plt.subplots(1,4)    
    axes[0].imshow(ref_roi.clip(), cmap='Greys')
    axes[0].set_title('Reference')
    axes[0].plot(4,4, 'ro')
    axes[1].imshow(moving_roi.clip(), cmap='Greys')
    axes[1].plot(inverse_transformed_affine_center_x, inverse_transformed_affine_center_y, 'ro')
    axes[1].set_title('Moving, no affine')

    vis_arr = moving_roi.clip(trans, mode='constant')

    axes[2].imshow(vis_arr, cmap='Greys')
    axes[2].plot(moving_roi.center[0], moving_roi.center[1], 'r*')  # Plot the center attribute of the roi obj. Note that I modified the property in the roi.py code to add 0.5 to each value.  
    axes[2].set_title('Moving, affine')
    axes[3].imshow(rotated_array, cmap='Greys')
    axes[3].plot(expected[0], expected[1], 'y*', markersize=10)  
    axes[3].plot(new_x, new_y, 'ro')
    show()"""
    
    # Approx is accurate to 1e-6. The affine introduces errors on the order of 1e-10 some of the time.
    assert new_x == pytest.approx(expected[0])
    assert new_y == pytest.approx(expected[1])