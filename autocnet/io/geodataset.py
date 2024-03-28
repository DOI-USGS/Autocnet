import json
import os

from plio.io.io_gdal import GeoDataset
import pvl

from autocnet.camera.sensor_model import create_sensor
from autocnet.spatial.surface import EllipsoidDem, GdalDem

class AGeoDataset(GeoDataset):
    def __init__(self, filename, sensortype='isis', dem=None):
        super().__init__(filename)
        if sensortype not in ['csm', 'isis']:
            raise ValueError('Argument sensortype must be either "csm" or "isis".')
        self.sensortype = sensortype
        self.dem = dem

    def _parse_radii_from_isd(self, isd):
        with open(isd, 'r') as stream:
            isd = json.load(stream)
        radii = isd['radii']
        if radii['unit'] == 'km':
            multiplier = 1000
        else:
            multiplier = 1
        semimajor = radii['semimajor'] * multiplier
        semiminor = radii['semiminor'] * multiplier
        return semimajor, semiminor
    
    def _make_dem_from_isd(self, isd):
        # Read the semi-major / semi-minor from the ISD
        semimajor, semiminor = self._parse_radii_from_isd(isd)
        if self.dem is None:
            # Create an EllipsoidDem
            dem = EllipsoidDem(semi_major=semimajor, semi_minor=semiminor)
        else:
            # Create a GdalDem
            dem = GdalDem(self.dem, semi_major=semimajor, semi_minor=semiminor)
        self.dem = dem
    
    def _parse_radii_from_label(self, label):
        bodycode = label['NaifKeywords']['BODY_CODE']
        radii_triplet = label['NaifKeywords'][f'BODY{bodycode}_RADII']
        semimajor = radii_triplet[0] * 1000  # Implicit km to m conversion
        semiminor = radii_triplet[1] * 1000
        return semimajor, semiminor
    
    def _parse_dem_from_label(self, label):
        return label['IsisCube']['Kernels']['ShapeModel']
    
    def _make_dem_from_isis(self):
        label = pvl.load(self.file_name)
        semimajor, semiminor, = self._parse_radii_from_label(label)
        
        # Check for ISISDATA environment variable
        if os.environ.get('ISISDATA', None):
            # Create a GdalDem
            dempath = self._parse_dem_from_label(label)
            dempath = dempath.replace('$base', os.path.join(os.environ['ISISDATA'], 'base'))
            dem = GdalDem(dempath, semi_major=semimajor, semi_minor=semiminor, dem_type='radius')
        else:
            dem = EllipsoidDem(semimajor, semiminor)
        
        self.dem = dem

    @property
    def sensormodel(self):
        if not hasattr(self, '_sensormodel'):
            if self.sensortype == 'csm':
                cam_path = self.file_name.replace('.cub', '.json')
                self._make_dem_from_isd(cam_path)
            else:
                cam_path = self.file_name
                self._make_dem_from_isis()
            self._sensormodel = create_sensor(self.sensortype, cam_path, dem=self.dem)
        return self._sensormodel