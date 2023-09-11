from autocnet.transformation.spatial import reproject, og2oc, oc2xyz
from autocnet.spatial import isis
from autocnet.spatial import overlap
import csmapi
import logging

# set up the logger file
log = logging.getLogger(__name__)

class ISISSensor:
    def __init__(self):
        self.sensor_type = "ISISSensor"

    def calculate_sample_line(self, node, lon, lat, **kwargs):
        """
        Calculate the sample and line for an isis camera sensor

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
        sample, line = isis.ground_to_image(node["image_path"], lon, lat)
        return sample, line

    def linesamp2xyz(node, sample, line, **kwargs):
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
        cube_path = node['image_path']

        try:
            p = isis.point_info(cube_path, sample, line, point_type='image')
        except:
            log.debug(f"Image coordinates {sample}, {line} do not project fot image {cube_path}.")

        try:
            x, y, z = p["BodyFixedCoordinate"].value
        except:
            x, y, z = p["BodyFixedCoordinate"]

        if getattr(p["BodyFixedCoordinate"], "units", "None").lower() == "km":
            x = x * 1000
            y = y * 1000
            z = z * 1000

        return x,y,z        

class CSMSensor:
    def __init__(self):
        self.sensor_type = "CSMSensor"

    def calculate_sample_line(self, node, lon, lat, **kwargs):
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
        
        kwargs: dict
            Contain information to be used if the csm sensor model is passed
            csm_kwargs = {
                'semi_major': int,
                'semi_minor': int,
                'height': int,
                'ncg': obj,
                'needs_projection': bool
            }
        Returns
        -------
        sample: int
            sample of point
        line: int
            lint of point
        """
        semi_major = kwargs.get('semi_major', None)
        semi_minor = kwargs.get('semi_minor', None)
        needs_projection = kwargs.get('needs_projection', None)
        
        # TODO: Take this projection out of the CSM model and work it into the point
        if needs_projection:
            height = kwargs.get('height', None)
            x,y,z = oc2xyz(lon, lat, semi_major, semi_minor, height)
            # The CSM conversion makes the LLA/ECEF conversion explicit
            gnd = csmapi.EcefCoord(x, y, z)
        else:
            ncg = kwargs.get('ncg', None)
            height = ncg.dem.get_height(lat, lon)
            # Get the BCEF coordinate from the lon, lat
            x, y, z = reproject([lon, lat, height],
                                semi_major, semi_minor, 'latlon', 'geocent')
            gnd = csmapi.EcefCoord(x, y, z)

        image_coord = node.camera.groundToImage(gnd)
        sample, line = image_coord.samp, image_coord.line
        return sample,line
    
    def linesamp2xyz(node, sample, line, **kwargs):
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
        semi_major = kwargs.get('semi_major', None)
        semi_minor = kwargs.get('semi_minor', None)
        ncg = kwargs.get('ncg', None)

        image_coord = csmapi.ImageCoord(sample, line)
        pcoord = node.camera.imageToGround(image_coord)
        # Get the BCEF coordinate from the lon, lat
        # TODO: Take this projection out of the CSM model and work it into the point
        updated_lon_og, updated_lat_og, _ = reproject([pcoord.x, pcoord.y, pcoord.z],
                                                        semi_major, semi_minor, 'geocent', 'latlon')
        updated_lon, updated_lat = og2oc(updated_lon_og, updated_lat_og, semi_major, semi_minor)

        updated_height = ncg.dem.get_height(updated_lat, updated_lon)


        # Get the BCEF coordinate from the lon, lat
        x, y, z = reproject([updated_lon_og, updated_lat_og, updated_height],
                            semi_major, semi_minor, 'latlon', 'geocent')

        return x,y,z

def create_sensor(sensor_type):
    sensor_type = sensor_type.lower()
    sensor_classes = {
        "isis": ISISSensor,
        "csm": CSMSensor
    }

    if sensor_type in sensor_classes:
        return sensor_classes[sensor_type]()
    else:
        raise Exception(f"Unsupported sensor type: {sensor_type}, accept 'isis' or 'csm'")
    