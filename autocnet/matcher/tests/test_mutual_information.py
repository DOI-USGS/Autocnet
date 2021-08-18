import math
import os
import sys
import unittest
from unittest.mock import patch

import pytest
import numpy as np

from .. import mutual_information

def test_mutual_information():
    d_template = np.array([[i for i in range(50, 100)] for j in range(50)])
    s_image = np.ones((100, 100))

    s_image[25:75, 25:75] = d_template

    x_offset, y_offset, max_corr, corr_map = mutual_information.mutual_information_match(d_template, s_image, bins=20)
    assert x_offset == 0.010530473741837909
    assert y_offset == 0.0
    assert max_corr == 2.0
    assert corr_map.shape == (51, 51)
    assert np.min(corr_map) >= 0.0
