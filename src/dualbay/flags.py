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
    "ATMFAIL": 0,
    "LAND": 1,
    "PRODWARN": 2,
    "HIGLINT": 3,
    "HILT": 4,
    "HISATZEN": 5,
    "COASTZ": 6,
    "STRAYLIGHT": 8,
    "CLDICE": 9,
    "COCCOLITH": 10,
    "TURBIDW": 11,
    "HISOLZEN": 12,
    "LOWLW": 14,
    "CHLFAIL": 15,
    "NAVWARN": 16,
    "ABSAER": 17,
    "MAXAERITER": 19,
    "MODGLINT": 20,
    "CHLWARN": 21,
    "ATMWARN": 22,
    "OPSHAL": 23,
    "SEAICE": 24,
    "NAVFAIL": 25,
    "FILTER": 26,
    "BOWTIEDEL": 28,
    "HIPOL": 29,
    "PRODFAIL": 30,
}
# Verified against the PACE OCI L2 flag_meanings string (NASA oceandata-notebooks, 2026):
# ATMFAIL LAND PRODWARN HIGLINT HILT HISATZEN COASTZ SPARE STRAYLIGHT CLDICE COCCOLITH TURBIDW HISOLZEN SPARE
# LOWLW CHLFAIL NAVWARN ABSAER SPARE MAXAERITER MODGLINT CHLWARN ATMWARN OPSHAL SEAICE NAVFAIL FILTER SPARE
# BOWTIEDEL HIPOL PRODFAIL SPARE

# NASA L3 mask minus ABSAER and ATMWARN: absorbing-aerosol pixels are the research object in Tokyo Bay
# (Higa et al. 2024), so they are kept and reported, not silently removed. Add them back via
# config/sensors.yaml `mask_extra` for a conservative run.
STANDARD_MASK = [
    "ATMFAIL",
    "LAND",
    "HIGLINT",
    "HILT",
    "HISATZEN",
    "STRAYLIGHT",
    "CLDICE",
    "COCCOLITH",
    "HISOLZEN",
    "LOWLW",
    "CHLFAIL",
    "NAVWARN",
    "MAXAERITER",
    "NAVFAIL",
    "FILTER",
]
WATER_MASK = ["LAND", "CLDICE", "NAVFAIL", "ATMFAIL"]
# Flags worth reporting per zone as diagnostics (not necessarily masked)
DIAGNOSTIC_FLAGS = [
    "ABSAER",
    "LOWLW",
    "STRAYLIGHT",
    "HIGLINT",
    "TURBIDW",
    "ATMWARN",
    "COASTZ",
    "CLDICE",
    "OPSHAL",
]


def from_file_attrs(flag_meanings: str | None, flag_masks=None) -> dict[str, int]:
    """Bit table from a file's own l2_flags attributes (preferred over the hardcoded table).

    flag_meanings is the space-separated name list; flag_masks the matching powers of two.
    SPARE entries are skipped. Falls back to L2_FLAG_BITS when nothing is given.
    """
    if not flag_meanings:
        return dict(L2_FLAG_BITS)
    names = str(flag_meanings).split()
    out = {}
    for i, n in enumerate(names):
        if n.upper() == "SPARE":
            continue
        if flag_masks is not None and i < len(flag_masks):
            m = int(flag_masks[i])
            bit = (m & 0xFFFFFFFF).bit_length() - 1
        else:
            bit = i
        out[n] = bit
    return out


def bitmask(names, table: dict[str, int] | None = None) -> int:
    """Integer mask with the bits of the named flags set."""
    table = table or L2_FLAG_BITS
    m = 0
    for n in names:
        if n in table:
            m |= 1 << table[n]
    return m


def is_set(flags: np.ndarray, name: str, table: dict[str, int] | None = None) -> np.ndarray:
    """Boolean array: flag `name` set."""
    table = table or L2_FLAG_BITS
    return (np.asarray(flags).astype(np.int64) & (1 << table[name])) != 0


def any_set(flags: np.ndarray, names, table: dict[str, int] | None = None) -> np.ndarray:
    """Boolean array: any of the named flags set."""
    return (np.asarray(flags).astype(np.int64) & bitmask(names, table)) != 0


def good_pixels(
    flags: np.ndarray | None, shape, level: str = "standard", table: dict[str, int] | None = None, extra=()
) -> np.ndarray:
    """Boolean array of pixels that pass the chosen mask ('standard' or 'water') plus `extra` flag names.

    With no flags array (older or custom files) everything is accepted.
    """
    if flags is None:
        return np.ones(shape, bool)
    names = list(STANDARD_MASK if level == "standard" else WATER_MASK) + list(extra)
    return ~any_set(flags, names, table)


def flag_fractions(
    flags: np.ndarray, where: np.ndarray, names=DIAGNOSTIC_FLAGS, table: dict[str, int] | None = None
) -> dict:
    """Fraction of pixels inside `where` carrying each diagnostic flag."""
    table = table or L2_FLAG_BITS
    n = int(where.sum())
    if n == 0:
        return {f"frac_{k.lower()}": np.nan for k in names}
    return {
        f"frac_{k.lower()}": float((is_set(flags, k, table) & where).sum() / n) for k in names if k in table
    }
