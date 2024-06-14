# Changelog

All changes that impact users of this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!---
This document is intended for users of the applications and API. Changes to things
like tests should not be noted in this document.

When updating this file for a PR, add an entry for your change under Unreleased
and one of the following headings:
 - Added - for new features.
 - Changed - for changes in existing functionality.
 - Deprecated - for soon-to-be removed features.
 - Removed - for now removed features.
 - Fixed - for any bug fixes.
 - Security - in case of vulnerabilities.

If the heading does not yet exist under Unreleased, then add it as a 3rd heading,
with three #.


When preparing for a public release candidate add a new 2nd heading, with two #, under
Unreleased with the version number and the release date, in year-month-day
format. Then, add a link for the new version at the bottom of this document and
update the Unreleased link so that it compares against the latest release tag.


When preparing for a bug fix release create a new 2nd heading above the Fixed
heading to indicate that only the bug fixes and security fixes are in the bug fix
release.
-->
## [Unreleased]
### Added
- Debug logging to `place_points_in_overlap` and `distribute_points_in_geom` to make debugging issues easier.

### Fixed
- Error in `find_interesting_feature` that was mis-using the ROI API. This bug was introduced in 1.2.0 when the ROI API updated.

## [1.2.0]
### Added
- Ability to choose whether to compute overlaps for a network candidate graph
- Integration tests for end-to-end of an equatorial CTX pair and a mid-latitude CTX trio. These write out ISIS control networks that can be visually inspected in `qnet` in case changes exceed the test tolerances.
- AGeoDataset class that abstracts the plio GeoDataset class. The AGeoDataset includes sensor models on the GeoDataset object. The object automatically instantiates an autocnet surface model using either an EllipsoidDem or a GdalDem. ISIS sensor models use the DEM defined on the cube. CSM sensor models must have a DEM explicitly passed.
- Full abstraction of the sensor model. The underlying sensor model can now be either USGSCSM or ISIS. The Network* objects track which sensor type was used, which DEM was used, if any, and whether the DEM is in radius or height.
- Multiple cropped ISIS cubes and associated CSM sensor models for testing. These cubes account for the offset issue that the `ale` `isd_generate` script has with generating CSM ISDs from cropped observations.

### Changed
- `cluster_submit_single` now uses a global retry in addition to a pre-ping on the pool. This accounts for database disconnects that are especially prevelent when the back, serverless, database is attempting to provision additional capacity. Each session access (e.g. session.query) will retry up to 5 times with a 300s timeout. Aurora should provision additional capacity in less than 300s.
- Smart subpixel matcher now re-uses the clip call on the ROI object instead of re-instantiating an ROI object with each different parameter set. The result is a measurable performance increase.
- Subpixel template now takes two arrays and return computed offsets. The caller is responsible for applying any transformations. For example, if an affine transformed moving template is passed, the caller of subpixel_template must apply the inverse transform.
- CI on the library now uses a mocked sqlalchemy connection. All tests can now run locally without the need for a supplemental postgres container. This changed removed some non-optimal tests that were testing datbase triggers and database instantiation handled by SQLAlchemy.

### Fixed
- Affine transformations are now properly accounting for data reflection.
- Error in subpixel ROI extraction when an affine transformation is provided. The code was using the same translation, regardless of the size of the data pulled into the ROI. This caused the center code to fail with large swawthes of bad data. The fix computes the size of the read data, including buffer, and computes the proper translation.
- Errors when importing sensor model in `overlap.py`
- Dealt with None values trying to be converted to a shapely point in `centroids.py`
- Dealt with Key Error when finding file paths in the network nodes.
- Removed depreciated function from `spatial.py` and replaced
- Fixed Errors in `network.py`
- Finding points in polar areas that avoids dividing by zero errors
- Small errors in `overlap.py` to get overlap function working.
- Raise an exception if the shear is too high when trying to find the baseline affine
- Correctly find the number of points without producing errors in `centroids.py`

## [1.1.0]
### Added
- Ability to place points in centroids instead of overlaps.
- Ability to find points in centroids focused around a plantary body
- Functionality to insert message into database
- Reformated `test_model.py` and added more test for overlays and points
- Functionality to create a point with reference measure
- Functionality to add measure to a point
- Functionality to convert from sample, line (x,y) pixel coordinates to Body-Centered, Body-Fixed (BCBF) coordinates in meters.
- Functionality to test for valid input images.
- Refactored place_points_in_overlap to make it easier to understand and work with
- Created `sensor.py` that creates a class for either a 'csm' camera sensor or an 'isis' camera sensor. This removes confusing and wordy code. Now just need to create a sensor object based on the input 'csm' or 'isis'. Code inside classes will figure out the rest.
- Fuctionality to convert between oc2xyz and xyz2oc in `spatial.py`

### Fixed
- string injection via format with sqlalchemy text() object.

## [1.0.2]
### Fixed
- API updates for numpy changing types and SQLAlchemy2.0.
- Tests updated or marked xfail for API changes

## [1.0.1]
### Added
- Logging to `affine` transformation that warns when the absolute value of the shear on the affine transformation is greater than 1e-2. High shear in the transformation matric was observed attempting to match nadir LROC-NAC to high slew LROC-NAC data. These high slew images do not match well to nadir images. Additionally, the `x_read_length` and `y_read_length` variables in the `Roi` class (`roi.py`) have a hard coded read length of two times the passed size. This read size is insufficient as the affine shear increases. A candidate enhancement would be to automatically compute the read size based on the affine transformation. This was not done in this addition as matching between high slew and nadir images using a correlation coefficient based approach is quite poor.
- Image check into `place_points_in_overlap` that ensures that the candidate reference image exists on disk. If it does not, the algorithm skips attempting to place points in that image and attempts to use the next image. This was added for LROC NAC control as some images may fail to download.
### Fixed
- `place_points_in_overlap` bug where if any of the points in the overlap failed to project into an image, all points in the overlap were lost. This was caused by #580, which allowed for multiple (a list) of inputs. The error handling was removed from the `image_to_ground` call so the `except` in `place_points_in_overlap` was never called.
## [1.0.0-rc2]

### Changed
- Redis queue population pushed to a background thread for non-blocking data processing.

### Fixed
- clip_center property in the roi object was checking for existence using getattr which fails on ndarrays that expect to use all() or any() for boolean checks. Fixed to use only tuples for clip_center.


## [1.0.0-rc1]

### Added
- [`pool_pre_ping`](https://docs.sqlalchemy.org/en/14/core/pooling.html#disconnect-handling-pessimistic) to the sqlalchemy engine connection to handle instances where hundreds of connections are simultaneously connecting to the database.
- verbose option to the smart subpixel matcher that will visualize the reference and moving ROIs in order to better support single point visualization.
- Debug logging to place_points_in_overlap

### Changed
- Estimation of the affine transformation no longer needs to use points completely within the destination (moving) image. Negative values are still valid for affine estimation and the sensor model is not constrained to within the image.
- to_isis method on the network candidate graph returns both the dataframe (existing functionality) and the filelist (new functionality).

### Fixed
- Fixed connection issues where too many connections to AWS RDW were causing connetions failures by adding an exponential sleep over five retries.
- Fixed missing import in place points in overlap that was causing a failure when attempting to throw a warning.

### Removed
- Ciratefi matcher from subpixel.py as the matcher is seldom used and better alternatives for scale and rotation invariance exist in the library.

## [0.7.0]()

### Added
- Added a mutual information matcher [#559](https://github.com/USGS-Astrogeology/autocnet/pull/559)
- Added residual column information to the Points model

### Changed
- `geom_match_simple` defaults to a 3rd order warp for interpolation
- Speed improvements for place_points_from_cnet dependent on COPY method instead of ORM update
- License from custom to CC0. Fixes [#607](https://github.com/USGS-Astrogeology/autocnet/issues/607)
- `place_points_in_overlap` now properly handles ISIS sensor exceptions
- Complex geometries that failed to find valid, in geometry points now fallback
  to using a random point distribution method to ensure points are added.
- Point and Image deletion from the DB now CASCADE to the measures table making
  modifications via measures easier to manage. 

### Fixed
- `update_from_jigsaw` failures due to stale code. Now uses a conntext on the engine to ensure closure
- Fixes errors where reference measure index was being incorrectly tracked when placing measures would fail [#606](https://github.com/USGS-Astrogeology/autocnet/issues/606)
-  Fixed #584 where importing autocnet fails on kalasiris imports by wrapping the import in a try accept.

## [0.6.0]

### Added
- Abstract DEM interface that supports ellipsoids #576
- Fourier-Mellon subpixel registration #558

### Changed
- Updates center of gravity style subpixel matcher to use scipy.ndimage.center_of_mass of determining subpixel shifts
- Use `kalisiris` instead of `pysis` #573
- Moved `acn_submit` cluster submission script into the library and added tests for cluster submission #567
- Point identifier no longer needs to be a unique string

### Fixed
- Image to ground to support multiple input types with proper output type handling #580
- Support for ISIS special pixels in image data #577
- Fix for no correlation map returned from `geom_match_simple` #556


