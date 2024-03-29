import pytest

import contextlib
import os
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

import autocnet.camera.sensor_model as sm
from autocnet.spatial.surface import GdalDem
from autocnet.examples import get_path
from autocnet.camera import sensor_model as sm
from autocnet.spatial.surface import EllipsoidDem, GdalDem

@pytest.fixture
def ctx_path():
    return get_path('G02_019154_1800_XN_00N133W.crop.cub')

@pytest.fixture
def remote_mola_height_dem():
    path = '/vsicurl/https://asc-mars.s3.us-west-2.amazonaws.com/basemaps/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif'
    return path

@pytest.fixture
def isis_mola_radius_dem():
    dem = None
    isisdata = os.environ.get('ISISDATA', None)
    if isisdata:
        path = os.path.join(isisdata, 'base/dems/molaMarsPlanetaryRadius0005.cub')
        dem = GdalDem(path, 3396190, 3396190, dem_type='radius')
    return dem

@pytest.fixture
def ellipsoid():
    return EllipsoidDem(semi_major=3396190, semi_minor=3396190)

@pytest.fixture
def ctx_isis_sensor(ctx_path, isis_mola_radius_dem):
    return sm.ISISSensor(ctx_path, isis_mola_radius_dem)

@pytest.fixture
def ctx_csm_sensor(ctx_path, isis_mola_radius_dem):
    return sm.CSMSensor(ctx_path, isis_mola_radius_dem)

@pytest.fixture
def base_sensor(ellipsoid):
    return sm.BaseSensor(None,ellipsoid)

class TestBaseSensor():

    def test__check_args(self, base_sensor):
        x = 1
        y = 1  
        xc, yc = base_sensor._check_arg(x, y)
        assert [x] == xc
        assert [y] == yc

        x = np.arange(5)
        y = np.arange(5)
        xc, yc = base_sensor._check_arg(x, y)
        np.testing.assert_array_equal(x, xc)
        np.testing.assert_array_equal(y, yc)

        x = range(5)
        y = range(5)
        xc, yc = base_sensor._check_arg(x, y)
        assert x == xc
        assert y == yc

        x = y = z = 1
        xc, yc, zc = base_sensor._check_arg(x, y, z)
        assert [x] == xc
        assert [y] == yc
        assert [z] == zc

        x = np.arange(5)
        y = np.arange(5)
        z = np.arange(5)
        xc, yc, zc = base_sensor._check_arg(x, y, z)
        np.testing.assert_array_equal(x, xc)
        np.testing.assert_array_equal(y, yc)
        np.testing.assert_array_equal(z, zc)

        x = range(5)
        y = range(5)
        z = range(5)
        xc, yc, zc = base_sensor._check_arg(x, y, z)
        assert x == xc
        assert y == yc
        assert z == zc       

    def test__check_args_raises(self, base_sensor):
        with pytest.raises(IndexError):
            base_sensor._check_arg(np.array([1, 2, 3, 4, 5]),np.array([1, 2, 3, 4]))
            base_sensor._check_arg(np.array([[1, 2], [3, 4]]),np.array([1, 2, 3, 4]))
            base_sensor._check_arg([10, 20, 30],[10, 20])
        with pytest.raises(TypeError):
            base_sensor._check_arg({10, 20}, {10, 20})


class TestIsisSensor():

    def test_raise_bad_coord_type(self, ctx_isis_sensor):
        with pytest.raises(ValueError):
            ctx_isis_sensor._point_info(10, 10, point_type='bogus')

    def test_sampline2lonlat(self, ctx_isis_sensor):
        lon, lat = ctx_isis_sensor.sampline2lonlat(10.0, 10.0)
        assert lon == 226.76892358441
        assert lat == -0.31770729411217

    def test_sampline2xyz(self, ctx_isis_sensor):
        x, y, z = ctx_isis_sensor.sampline2xyz(10.0, 10.0)
        assert x == pytest.approx(-2327023.0983832)
        assert y == pytest.approx(-2475336.0552312)
        assert z == pytest.approx(-18838.904973497)

    def test_lonlat2sampline(self, ctx_isis_sensor):
        samp, line = ctx_isis_sensor.lonlat2sampline(226.8, -0.25)
        assert samp == pytest.approx(450.47864761698)
        assert line == pytest.approx(638.5458457207)

    def test_xyz2sampline(self, ctx_isis_sensor):
        x = -2327023.0983832
        y = -2475336.0552312
        z = -18838.904973497
        samp, line = ctx_isis_sensor.xyz2sampline(x,y,z)
        assert samp == pytest.approx(10.0,6)
        assert line == pytest.approx(10.0,6)

    def test_lonlat2xyz(self, ctx_isis_sensor):
        x, y, z = ctx_isis_sensor.lonlat2xyz(226.76892358441, -0.31770729411217)
        assert x == pytest.approx(-2327023.0983832)
        assert y == pytest.approx(-2475336.0552312)
        assert z == pytest.approx(-18838.904973497)


class TestISIS(unittest.TestCase):

    def setUp(self) -> None:
        self.resourcedir = Path("test-resources")
        self.red50img = self.resourcedir / "PSP_010502_2090_RED5_0.img"
        self.red51img = self.resourcedir / "PSP_010502_2090_RED5_1.img"

        if not all((self.red50img.exists(), self.red51img.exists())):
            self.skipTest(
                f"One or more files is missing from the "
                f"{self.resourcedir.resolve()} directory. "
                f"Tests on real files skipped."
            )

        self.cube = self.red50img.with_suffix(".TestISIS.cub")
        isis.hi2isis(self.red50img, to=self.cube)
        isis.spiceinit(self.cube)

        self.map = self.cube.with_suffix(".map.cub")
        isis.cam2map(self.cube, to=self.map)

        self.si_nomap = sm.ISISSensor(self.cub, EllipsoidDem(3396190, 3396190))
        self.si_map = sm.ISISSensor(self.map, EllipsoidDem(3396190, 3396190))

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            self.cube.unlink()
            self.map.unlink()
            Path("print.prt").unlink()

    def test_point_info(self):
        self.assertRaises(
            ValueError, self.si_nomap._point_info, -10, -10, "image"
        )

        d = self.si_nomap._point_info(10, 10, "image")
        self.assertEqual(10, d["Sample"])
        self.assertEqual(10, d["Line"])
        self.assertEqual(28.537311831691, d["PlanetocentricLatitude"].value)
        self.assertEqual(274.14960455269, d["PositiveEast360Longitude"].value)

        x_sample = 20
        x_lon = 274.14948072713
        y_line = 20
        y_lat = 28.537396673529

        d2 = self.si_nomap._point_info([10, x_sample], [10, y_line], "image")
        self.assertEqual(x_sample, d2[1]["Sample"])
        self.assertEqual(y_line, d2[1]["Line"])
        self.assertEqual(y_lat, d2[1]["PlanetocentricLatitude"].value)
        self.assertEqual(x_lon, d2[1]["PositiveEast360Longitude"].value)

        self.assertEqual(d, d2[0])

        d3 = self.si_nomap._point_info(x_lon, y_lat, "ground")
        self.assertEqual(20.001087366213, d3["Sample"])
        self.assertEqual(20.004109124452, d3["Line"])

        d4 = self.si_map._point_info(10, 10, "image")
        d5 = self.si_map._point_info([10, x_sample], [10, y_line], "image")
        self.assertEqual(d4, d5[0])

        d6 = self.si_map._point_info(x_lon, y_lat, "ground")
        self.assertEqual(961.20490394075, d6["Sample"])
        self.assertEqual(3959.4515093358, d6["Line"])

    def test_image_to_ground(self):
        lon, lat = self.si_nomap._image_to_ground( 20, 20)
        x = 274.14948
        y = 28.537396
        self.assertAlmostEqual(x, lon, places=5)
        self.assertAlmostEqual(y, lat, places=5)

        lons, lats = self.si_nomap.sampline2lonlat(
            np.array([10, 20]), np.array([10, 20])
        )
        npt.assert_allclose(np.array([274.14961, x]), lons, rtol=1e-05)
        npt.assert_allclose(np.array([28.53731, y]), lats, rtol=1e-05)

        lon, lat = self.si_map.sampline2lonlat(20, 20)
        x = 274.13914
        y = 28.57541
        self.assertAlmostEqual(x, lon, places=5)
        self.assertAlmostEqual(y, lat, places=5)

        lons, lats = self.si_map.sampline2lonlat(np.array([10, 20]), np.array([10, 20]))
        npt.assert_allclose(np.array([274.13903, x]), lons, rtol=1e-05)
        npt.assert_allclose(np.array([28.57551, y]), lats, rtol=1e-05)

    def test_ground_to_image(self):
        lon = 274.14948072713
        lat = 28.537396673529
        goal_samp = 20.001087
        goal_line = 20.004109
        sample, line = self.si_nomap.lonlat2sampline(lon, lat)
        self.assertAlmostEqual(goal_samp, sample, places=6)
        self.assertAlmostEqual(goal_line, line, places=6)

        samples, lines = self.si_nomap.lonlat2sampline(np.array([lon, 274.1495]),np.array([lat, 28.5374]))
        npt.assert_allclose(np.array([goal_samp, 18.241668]), samples)
        npt.assert_allclose(np.array([goal_line, 20.145382]), lines)

        lon = 274.13903475
        lat = 28.57550764
        goal_samp = 10.5001324
        goal_line = 10.49999466
        sample, line = self.si_map.lonlat2sampline(lon, lat)
        self.assertAlmostEqual(goal_samp, sample)
        self.assertAlmostEqual(goal_line, line)

        samples, lines = self.si_map.lonlat2sampline(np.array([lon, 274.14948072713]),np.array([lat, 28.57541113]))
        npt.assert_allclose(np.array([goal_samp, 961.03569217]), samples)
        npt.assert_allclose(np.array([goal_line, 20.50009032]), lines)

class TestCsmSensor():
    def test_sampline2lonlat(self, ctx_csm_sensor):
        lon, lat = ctx_csm_sensor.sampline2lonlat(10.0, 10.0)
        assert lon == 226.76892358441
        assert lat == -0.31770729411217

    def test_sampline2xyz(self, ctx_csm_sensor):
        x, y, z = ctx_csm_sensor.sampline2xyz(10.0, 10.0)
        assert x == pytest.approx(-2327023.0983832)
        assert y == pytest.approx(-2475336.0552312)
        assert z == pytest.approx(-18838.904973497)

    def test_lonlat2sampline(self, ctx_csm_sensor):
        samp, line = ctx_csm_sensor.lonlat2sampline(226.8, -0.25)
        assert samp == pytest.approx(450.47864761698)
        assert line == pytest.approx(638.5458457207)

    def test_xyz2sampline(self, ctx_csm_sensor):
        x = -2327023.0983832
        y = -2475336.0552312
        z = -18838.904973497
        samp, line = ctx_csm_sensor.xyz2sampline(x,y,z)
        print(samp, line)
        assert samp == pytest.approx(10.0,6)
        assert line == pytest.approx(10.0,6)
