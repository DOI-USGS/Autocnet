"""
A set of classes that represent a sensor mode. Each class implements the
calculate_sample_line and sampline2xyz methods for computing image coordinate location
from lon/lat and xyz location from sample/line, respectively.
"""

import abc
import collections
import logging
from subprocess import CalledProcessError
from numbers import Number

from autocnet.transformation.spatial import og2oc, og2xyz, xyz2og, xyz2oc
from autocnet.io import isis

import numpy as np
import pvl

try:
    import kalasiris as isis
except Exception as exception:
    from autocnet.utils.utils import FailedImport
    isis = FailedImport(exception)

from knoten.csm import create_csm, generate_ground_point, generate_image_coordinate
from csmapi import csmapi

# set up the logger file
log = logging.getLogger(__name__)

class BaseSensor:
    def __init__(self, data_path, dem):
        self.data_path = data_path
        self.dem = dem
        self.semi_major = self.dem.a
        self.semi_minor = self.dem.c

    def _check_arg(self, x, y, z=None):
        if isinstance(x, collections.abc.Sequence) and isinstance(y, collections.abc.Sequence) and (isinstance(z, collections.abc.Sequence) or z is None):
            if len(x) != len(y) or (z is not None and len(x) != len(y) != len(z)):
                raise IndexError(
                    f"Sequences given to x and y must be of the same length."
                )
            x_coords = x
            y_coords = y
            if z is not None:
                z_coords = z
        elif isinstance(x, Number) and isinstance(y, Number) and (isinstance(z, Number) or z is None):
            x_coords = [x, ]
            y_coords = [y, ]
            if z is not None:
                z_coords = [z, ]
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and (isinstance(z, np.ndarray) or z is None):
            if not all((x.ndim == 1, y.ndim == 1)) and (z is not None and all((x.ndim ==1, y.ndim == 1, z.ndim == 1))):
                if z is None:
                    raise IndexError(
                        f"If they are numpy arrays, x and y must be one-dimensional, "
                        f"they were: {x.ndim} and {y.ndim}"
                    )
                else:
                    raise IndexError(
                        f"If they are numpy arrays, x and y must be one-dimensional, "
                        f"they were: {x.ndim}, {y.ndim} and {z.ndim}"
                    )     
            if x.shape != y.shape:
                raise IndexError(
                    f"Numpy arrays given to x and y must be of the same shape."
                )
            x_coords = x
            y_coords = y
            if z is not None:
                z_coords = z
        else:
            raise TypeError(
                f"The values of x and y were neither Sequences nor individual numbers"
                f"numbers, they were: {x} and {y}"
            )
        
        if z is not None:
            return x_coords, y_coords, z_coords
        else:
            return x_coords, y_coords
        
    def _flatten_results(self, x, results):
        if isinstance(x, collections.abc.Sequence):
            # Maintain the return type, if a sequence is passed,
            # return a sequence.
            if isinstance(results, np.ndarray):
                return results.tolist()
            return results
        elif isinstance(x, np.ndarray):
            return np.asarray(results)
        else:
            return results[0]
        
    @abc.abstractmethod
    def lonlat2xyz(self, lon, lat, height):
        return 
    
    @abc.abstractmethod
    def xyz2sampline(self, x, y, z):
        return
    
    @abc.abstractmethod
    def lonlat2sampline(self, lon, lat):
        return
    
    @abc.abstractmethod
    def sampline2xyz(self, sample, line):
        return

    @abc.abstractmethod
    def sampline2lonlat(sample, line):
        return


class ISISSensor(BaseSensor):
    """
    ISIS defaults to ocentric latitudes for everything.
    """
    sensor_type = "isis"

    def _point_info(
            self,
            x,
            y,
            point_type: str,
            allowoutside=False
    ):
        """
        Returns a pvl.collections.MutableMappingSequence object or a
        Sequence of MutableMappingSequence objects which contain keys
        and values derived from the output of ISIS campt or mappt on
        the *cube_path*.

        If x and y are single numbers, then a single MutableMappingSequence
        object will be returned.  If they are Sequences or Numpy arrays, then a
        Sequence of MutableMappingSequence objects will be returned,
        such that the first MutableMappingSequence object of the returned
        Sequence will correspond to the result of *x[0]* and *y[0]*,
        etc.

        Raises subprocess.CalledProcessError if campt or mappt have failures.
        May raise ValueError if campt completes, but reports errors.

        Parameters
        ----------
        cube_path : os.PathLike
                    Path to the input cube.

        x : Number, Sequence of Numbers, or Numpy Array
            Point(s) in the x direction. Interpreted as either a sample
            or a longitude value determined by *point_type*.

        y : Number, Sequence of Numbers, or Numpy Array
            Point(s) in the y direction. Interpreted as either a line
            or a latitude value determined by *point_type*.

        point_type : str
                    Options: {"image", "ground"}
                    Pass "image" if  x,y are in image space (sample, line) or
                    "ground" if in ground space (longitude, latitude)

        allowoutside: bool
                    Defaults to False, this parameter is passed to campt
                    or mappt.  Please read the ISIS documentation to
                    learn more about this parameter.

        """
        point_type = point_type.casefold()
        valid_types = {"image", "ground"}
        if point_type not in valid_types:
            raise ValueError(
                f'{point_type} is not a valid point type, valid types are '
                f'{valid_types}'
            )

        x_coords, y_coords = self._check_arg(x, y)
        results = []
        if pvl.load(self.data_path).get("IsisCube").get("Mapping"):
            # We have a projected image, and must use mappt
            mappt_common_args = dict(allowoutside=allowoutside, type=point_type)

            for xx, yy in zip(x_coords, y_coords):
                mappt_args = {
                    "ground": dict(
                        longitude=xx,
                        latitude=yy,
                        coordsys="UNIVERSAL"
                    ),
                    "image": dict(
                        # Convert PLIO pixels to ISIS pixels
                        sample=xx+0.5,
                        line=yy+0.5
                    )
                }
                for k in mappt_args.keys():
                    mappt_args[k].update(mappt_common_args)
                mapres = pvl.loads(isis.mappt(self.data_path, **mappt_args[point_type]).stdout)["Results"]
                
                # convert from ISIS pixels to PLIO pixels
                mapres['Sample'] = mapres['Sample'] - 0.5
                mapres['Line'] = mapres['Line'] - 0.5

                results.append(mapres)
        else:
            # Not projected, use campt
            if point_type == "ground":
                # campt uses lat, lon for ground but sample, line for image.
                # So swap x,y for ground-to-image calls
                p_list = [f"{lat}, {lon}" for lon, lat in zip(x_coords, y_coords)]
            else:
                p_list = [
                    f"{samp+0.5}, {line+0.5}" for samp, line in zip(x_coords, y_coords)
                ]

            # ISIS's campt needs points in a file
            with isis.fromlist.temp(p_list) as f:
                cp = isis.campt(
                    self.data_path,
                    coordlist=f,
                    allowoutside=allowoutside,
                    usecoordlist=True,
                    coordtype=point_type
                )

            camres = pvl.loads(cp.stdout)
            if 'campt' in camres.keys():
                camres = camres['campt']
            for r in camres.getall("GroundPoint"):
                if r['Error'] is None:
                    # convert all pixels to PLIO pixels from ISIS
                    r["Sample"] -= .5
                    r["Line"] -= .5
                    
                    results.append(r)
                else:
                    r["Sample"] = None
                    r["Line"] = None
                    results.append(r)

        return self._flatten_results(x, results)

    def _get_value(self, obj):
        """Returns *obj*, unless *obj* is of type pvl.collections.Quantity, in
        which case, the .value component of the object is returned."""
        if isinstance(obj, pvl.collections.Quantity):
            return obj.value
        else:
            return obj
    
    def xyz2sampline(self, x, y, z):
        # No ocentric conversion here. That conversion is handled in lonlat2sampline
        lon, lat = xyz2og(x, y, z, self.semi_major, self.semi_minor)
        return self.lonlat2sampline(lon, lat)

    def lonlat2sampline(self, lon, lat, allowoutside=False):
        """
        Returns a two-tuple of numpy arrays or a two-tuple of floats, where
        the first element of the tuple is the sample(s) and the second
        element are the lines(s) that represent the coordinate(s) of the
        input *lon* and *lat* in *cube_path*.

        If *lon* and *lat* are single numbers, then the returned two-tuple
        will have single elements. If they are Sequences, then the returned
        two-tuple will contain numpy arrays.

        Raises the same exceptions as _point_info().

        Parameters
        ----------
        lon: Number or Sequence of Numbers
            Longitude coordinate(s).

        lat: Number or Sequence of Numbers
            Latitude coordinate(s).

        """
        # ISIS is expecting ocentric latitudes. Convert from ographic before passing.
        lonoc, latoc = og2oc(lon, lat, self.semi_major, self.semi_minor)
        res = self._point_info(lonoc, latoc, "ground", allowoutside=allowoutside)
        if isinstance(lon, (collections.abc.Sequence, np.ndarray)):
            samples, lines = np.asarray([[r["Sample"], r["Line"]] for r in res]).T
        else:
            samples, lines = res["Sample"], res["Line"]

        return samples, lines

    def sampline2xyz(self, sample, line):
        """
        Convert a line and sample into and x,y,z point for isis camera model

        Parameters
        ----------
        node: obj
            autocnet object containing image information

        sample: int
            sample of point

        line: int
            lint of point
        
        Returns
        -------
        x,y,z : int(s)
            x,y,z coordinates of the point
        """

        try:
            p = self._point_info(sample, line, point_type='image')
        except CalledProcessError as e:
            if 'Requested position does not project in camera model' in e.stderr:
                log.debug(f"Image coordinates {sample}, {line} do not project fot image {self.data_path}.")

        try:
            x, y, z = p["BodyFixedCoordinate"].value
        except:
            x, y, z = p["BodyFixedCoordinate"]

        if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
            x = x * 1000
            y = y * 1000
            z = z * 1000

        return x,y,z          
    
    def sampline2lonlat(
            self,
            sample,
            line,
            lontype="PositiveEast360Longitude",
            lattype="PlanetographicLatitude",
            allowoutside=False
    ):
        """
        Returns a two-tuple of numpy arrays or a two-tuple of floats, where
        the first element of the tuple is the longitude(s) and the second
        element are the latitude(s) that represent the coordinate(s) of the
        input *sample* and *line* in *cube_path*.

        If *sample* and *line* are single numbers, then the returned two-tuple
        will have single elements. If they are Sequences, then the returned
        two-tuple will contain numpy arrays.

        Raises the same exceptions as point_info().

        Parameters
        ----------
        cube_path : os.PathLike
                    Path to the input cube.

        sample : Number or Sequence of Numbers
            Sample coordinate(s).

        line : Number or Sequence of Numbers
            Line coordinate(s).

        lontype: str
            Name of key to query in the campt or mappt return to get the returned
            longitudes. Defaults to "PositiveEast360Longitude", but other values
            are possible. Please see the campt or mappt documentation.

        lattype: str
            Name of key to query in the campt or mappt return to get the returned
            latitudes. Defaults to "PlanetocentricLatitude", but other values
            are possible.  Please see the campt or mappt documentation.

        """
        res = self._point_info(sample, line, "image", allowoutside=allowoutside)
        if isinstance(sample, (collections.abc.Sequence, np.ndarray)):
            lon_list = list()
            lat_list = list()
            for r in res:
                lon_list.append(self._get_value(r[lontype]))
                lat_list.append(self._get_value(r[lattype]))

            lons = np.asarray(lon_list)
            lats = np.asarray(lat_list)
        else:
            lons = self._get_value(res[lontype])
            lats = self._get_value(res[lattype])
        return lons, lats   

    def lonlat2xyz(self, lon, lat):
        sample, line = self.lonlat2sampline(lon, lat)
        return self.sampline2xyz(sample, line)

    
class CSMSensor(BaseSensor):
    """
    The CSM sensor model works in ographic latitudes.
    """
    sensor_type = "csm"
    
    def __init__(self, data_path, dem):
        super().__init__(data_path,dem)
        # Create a sensor model object using knoten here.
        self.sensor = create_csm(data_path)
        self.dem = dem

    def xyz2sampline(self, x, y, z):
        x_coords, y_coords, z_coords = self._check_arg(x, y, z)
        results = np.empty((len(x_coords),2))
        for i, coord in enumerate(zip(x_coords, y_coords, z_coords)):
            imagept = generate_image_coordinate(coord, self.sensor)
            results[i,0] = imagept.samp
            results[i,1] = imagept.line
        return (self._flatten_results(x, results[:,0]),
                self._flatten_results(x, results[:,1]))

    def lonlat2sampline(self, lon, lat, **kwargs):
        """
        Calculate the sample and line for an csm camera sensor

        Parameters
        ----------
        node: obj
            autocnet object containing image information

        lon: int
            longitude of point
        
        lat: int
            latitude of point

        Returns
        -------
        sample: int
            sample of point
        line: int
            lint of point
        """
        x_coords, y_coords = self._check_arg(lon, lat)
        heights = []
        for coord in zip(x_coords, y_coords):
            # get_height needs lat, lon ordering
            heights.append(self.dem.get_height(coord[1], coord[0]))
        x,y,z = og2xyz(x_coords, y_coords, heights, self.semi_major, self.semi_minor)
        
        return self.xyz2sampline(self._flatten_results(lon, x),
                                 self._flatten_results(lon, y),
                                 self._flatten_results(lon, z))
    
    def sampline2xyz(self, sample, line):
        """
        Convert a line and sample into and x,y,z point for csm camera model

        Parameters
        ----------
        node: obj
            autocnet object containing image information

        sample: int
            sample of point

        line: int
            lint of point

        kwargs: dict
            Contain information to be used if the csm sensor model is passed
            csm_kwargs = {
                'semi_major': int,
                'semi_minor': int,
                'ncg': obj,
            }
        
        Returns
        -------
        x,y,z : int(s)
            x,y,z coordinates of the point
        """
        csample, cline = self._check_arg(sample, line)
        results = np.empty((len(csample),3))
        # CSMAPI using line/sample ordering. Swap here.
        for i, coord in enumerate(zip(cline, csample)):
            # self.dem is an autocnet surface model dem that has a GeoDataset attribute
            bcbf = generate_ground_point(self.dem, coord, self.sensor)
            results[i,0] = bcbf.x
            results[i,1] = bcbf.y
            results[i,2] = bcbf.z
        return (self._flatten_results(sample, results[:,0]),
                self._flatten_results(sample, results[:,1]),
                self._flatten_results(sample, results[:,2]))
    
    def sampline2lonlat(self, sample, line, **kwargs):
        # sampline2xyz handles the CSM iteration, just pass the array through
        x,y,z = self.sampline2xyz(sample, line)
        lons, lats = xyz2og(x, y, z, self.semi_major, self.semi_minor)
        return lons, lats
    
    def lonlat2xyz(self, lon, lat):
        # All size checking and response type code is handled in self.lonlat2sampline
        sample, line = self.lonlat2sampline(lon, lat)
        return self.sampline2xyz(sample, line)
    

def create_sensor(sensor_type, cam_path, dem=None):
    sensor_type = sensor_type.lower()
    sensor_classes = {
        "isis": ISISSensor,
        "csm": CSMSensor
    }

    if sensor_type in sensor_classes:
        return sensor_classes[sensor_type](cam_path, dem)
    else:
        raise Exception(f"Unsupported sensor type: {sensor_type}, accept 'isis' or 'csm'")
    