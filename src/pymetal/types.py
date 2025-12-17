"""
pymetal.types - Metal utility types and structures

This module provides utility types used across Metal operations:
- Origin, Size, Range for specifying regions
- ClearColor for render pass clearing
"""

from ._pymetal import (
    # Utility structures
    Origin,
    Size,
    Range,
    ClearColor,
)

__all__ = [
    "Origin",
    "Size",
    "Range",
    "ClearColor",
]
