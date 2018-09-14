from unittest import mock

import numpy as np
import pytest

from autocnet.matcher import cpu_ring_matcher as rm

@pytest.mark.parametrize('arr, expected', [
                         (np.array([[1,0],[1,1], [2,3]]), (1,2)),
                         (np.array([[0,0], [1,1], [2,2]]), (3,2)
                        )])
def test_check_pidx_duplicates(arr, expected):
    print(arr)
    pidx = rm.check_pidx_duplicates(arr)
    print(pidx)
    assert pidx.shape == expected

@pytest.mark.parametrize("a, b, threshold, expected", [
                         # Tests standard call
                         (np.array([1,2,3]), 
                          np.array([[1,2,3], [4,5,6], [7,8,9]]),
                          1.5, 
                          np.array([0])),
                          # Tests call where distances are too close
                        (np.array([1,2,3]),
                         np.array([[7,8,9], [1,2,4], [1,2,4.1]]),
                         1.5, 
                         None),
                         # Tests call with close distances where the threshold is low
                        (np.array([1,2,3]),
                         np.array([[7,8,9], [1,2,4], [1,2,4.1]]),
                         1., 
                         1),
                         # Tests call when np.argmin will fail
                        (np.array([np.nan, np.nan]),
                         np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                         1.5,
                         None),
                         # Tests call where descriptors are identical
                        (np.array([1,2,3]),
                         np.array([[1,2,3], [1,2,3], [1,2,3]]),
                         1.5,
                         None)
])
def test_sift_match(a, b, threshold, expected):
    assert rm.sift_match(a, b, thresh=threshold) == expected    

@pytest.mark.parametrize("x,y, eidx",[(np.array([[1,1],[2,2],[3,3], [4,4], [5,5]]),
                                       np.array([[1.1,1.0],[1.9,1.95],[3,3], [-4,-4], [5,5]]),
                                       np.array([[0,1,2,4]])),
                                      (np.array([[1,1], [5,5]]),
                                       np.array([[1,1], [3,3]]),
                                       [])
                                      ])
def test_ransac_permut(x, y, eidx):
    xp, yp, idx = rm.ransac_permute(x, y, 0.2, 2)
    np.testing.assert_array_equal(idx, eidx)


def test_add_correspondences():
    func = 'autocnet.matcher.cpu_ring_matcher.ring_match_one'
    with mock.patch(func, return_value=1):
        in_feats = np.array([[1,1], [2,2]])
        ref_feats = np.array([[1,1],[2,2],[3,3], [4,4], [5,5]])
        tar_feats = np.array([[1.1,1.0],[1.9,1.95],[3,3], [-4,-4], [5,5]])
        
        rm.add_correspondences(in_feats, ref_feats, tar_feats, None, None,
                               (0,6), (0,6),(0,1))


