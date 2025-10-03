"""
Version information for GaitSetPy.

This file serves as the single source of truth for the package version.
All other files should import the version from here.
"""

__version__ = "0.2.2"

def _parse_version(version_string):
    """Parse version string into tuple of integers, ignoring non-numeric suffixes."""
    parts = []
    for part in version_string.split('.'):
        # Extract leading digits only, stop at first non-digit
        numeric = ''
        for c in part:
            if c.isdigit():
                numeric += c
            else:
                break
        if numeric:
            parts.append(int(numeric))
    return tuple(parts)

__version_info__ = _parse_version(__version__)

# Version metadata
VERSION = __version__
VERSION_INFO = __version_info__

# Release information
RELEASE_DATE = "2025-10-03"
RELEASE_NOTES = "Added concurrent downloading support"

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get the current version as a tuple of integers."""
    return __version_info__

def get_release_info():
    """Get release information."""
    return {
        'version': __version__,
        'version_info': __version_info__,
        'release_date': RELEASE_DATE,
        'release_notes': RELEASE_NOTES
    }
