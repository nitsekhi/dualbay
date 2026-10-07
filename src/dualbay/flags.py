"""NASA ocean-color Level-2 flags (l2_flags) as used by PACE OCI, MODIS, VIIRS.

Bit layout follows the NASA Ocean Biology Processing Group convention
(https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/). Two masks are provided:

  STANDARD_MASK  the OBPG Level-3 mask: pixels the agency itself would not trust
  WATER_MASK     the minimum: land, clouds, navigation failures

Why this matters for the project: the fraction of pixels flagged ABSAER (absorbing aerosol),
LOWLW (low water-leaving radiance) and STRAYLIGHT per zone is a direct diagnostic of the
atmospheric-correction problem described for Tokyo Bay by Higa et al. (2024).
"""

from __future__ import annotations

import numpy as np

L2_FLAG_BITS = {
    "ATMFAIL": 0, "LAND": 1, "PRODWARN": 2, "HIGLINT": 3, "HILT": 4, "HISATZEN": 5,
    "COASTZ": 6, "STRAYLIGHT": 8, "CLDICE": 9, "COCCOLITH": 10, "TURBIDW": 11,
    "HISOLZEN": 12, "LOWLW": 14, "CHLFAIL": 15, "NAVWARN": 16, "ABSAER": 17,
    "MAXAERITER": 19, "MODGLINT": 20, "CHLWARN": 21, "ATMWARN": 22, "SEAICE": 24,
    "NAVFAIL": 25, "FILTER": 26, "BOWTIEDEL": 28, "HIPOL": 29, "PRODFAIL": 30,
}

STANDARD_MASK = ["ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "STRAYLIGHT", "CLDICE",
                 "COCCOLITH", "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "ABSAER",
                 "MAXAERITER", "ATMWARN", "NAVFAIL"]
WATER_MASK = ["LAND", "CLDICE", "NAVFAIL", "ATMFAIL"]
# Flags worth reporting per zone as diagnostics (not necessarily masked)
DIAGNOSTIC_FLAGS = ["ABSAER", "LOWLW", "STRAYLIGHT", "HIGLINT", "TURBIDW", "ATMWARN", "COASTZ", "CLDICE"]


def bitmask(names) -> int:
    """Integer mask with the bits of the named flags set."""
    m = 0
    for n in names:
        m |= 1 << L2_FLAG_BITS[n]
    return m


def is_set(flags: np.ndarray, name: str) -> np.ndarray:
    """Boolean array: flag `name` set."""
    return (np.asarray(flags).astype(np.int64) & (1 << L2_FLAG_BITS[name])) != 0


def any_set(flags: np.ndarray, names) -> np.ndarray:
    """Boolean array: any of the named flags set."""
    return (np.asarray(flags).astype(np.int64) & bitmask(names)) != 0


def good_pixels(flags: np.ndarray | None, shape, level: str = "standard") -> np.ndarray:
    """Boolean array of pixels that pass the chosen mask ('standard' or 'water').

    With no flags array (older or custom files) everything is accepted.
    """
    if flags is None:
        return np.ones(shape, bool)
    names = STANDARD_MASK if level == "standard" else WATER_MASK
    return ~any_set(flags, names)


def flag_fractions(flags: np.ndarray, where: np.ndarray, names=DIAGNOSTIC_FLAGS) -> dict:
    """Fraction of pixels inside `where` carrying each diagnostic flag."""
    n = int(where.sum())
    if n == 0:
        return {f"frac_{k.lower()}": np.nan for k in names}
    return {f"frac_{k.lower()}": float((is_set(flags, k) & where).sum() / n) for k in names}
