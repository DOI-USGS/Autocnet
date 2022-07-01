import math
import os
import sys
import unittest
from unittest.mock import patch

import pytest
import numpy as np

from .. import mutual_information

def test_good_mi():
    test_image1 = np.array([[i for i in range(50)] for j in range(50)])
    corrilation = mutual_information.mutual_information(test_image1, test_image1)
    assert corrilation == pytest.approx(2.30258509299404)

def test_bad_mi():
    test_image1 = np.array([[i for i in range(50)] for j in range(50)])
    test_image2 = np.ones((50, 50))
    corrilation = mutual_information.mutual_information(test_image1, test_image2)
    assert corrilation == pytest.approx(0)

