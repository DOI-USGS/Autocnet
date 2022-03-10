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


