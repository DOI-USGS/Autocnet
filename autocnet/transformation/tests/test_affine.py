from plio.io.io_gdal import GeoDataset
import pytest

from autocnet.transformation import affine

def test_estimate_affine_transformation():
    gd_base = GeoDataset('test/test_subpixel_match/B08_012650_1780_XN_02S046W.l1.cal.destriped.crop.cub')
    gd_match = GeoDataset('test/test_subpixel_match/J04_046447_1777_XI_02S046W.l1.cal.destriped.crop.cub')
    affine_transform = affine.estimate_affine_transformation(gd_base,gd_match, 150, 150)
    assert affine_transform == 'foo'