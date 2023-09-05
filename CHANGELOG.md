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
- Ability to place points in centroids instead of overlaps.
- Ability to find points in centroids focused around a plantary body
- Functionality to insert message into database
- Reformated `test_model.py` and added more test for overlays and points
- Functionality to create a point with reference measure
- Functionality to add measure to a point
- Functionality to convert from sample, line (x,y) pixel coordinates to Body-Centered, Body-Fixed (BCBF) coordinates in meters.
- Functionality to test for valid input images.

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


