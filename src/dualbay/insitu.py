"""Parse downloaded in-situ water quality files into one long table.

Target format (one row per observation):
    station | time (Asia/Tokyo) | layer | depth_m | param | value | unit | qc | source_file

Works with the CSV and Excel exports typical for Japanese monitoring sites:
Shift-JIS or UTF-8 encoding, title rows above the header, half-width katakana,
missing-value markers such as 欠測 (missing), and hour 24 meaning midnight of the next day.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TZ = "Asia/Tokyo"
MISSING = {"", "-", "--", "---", "*", "**", "///", "欠測", "欠", "調整中", "点検中", "nan", "NaN", "null"}
# Layer words used in Japanese monitoring tables → English labels
LAYERS = {
    "表層": "surface",
    "上層": "upper",
    "中層": "middle",
    "下層": "lower",
    "底層": "bottom",
    "底": "bottom",
}
ENCODINGS = ("utf-8-sig", "cp932", "euc_jp")


def norm(text) -> str:
    """NFKC-normalise (half-width katakana → full-width, full-width ASCII → ASCII) and lower-case."""
    return unicodedata.normalize("NFKC", str(text)).strip().lower()


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV/Excel file with no header assumption; every cell as string."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, header=None, dtype=str)
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {path} with encodings {ENCODINGS}")
    lines = [ln for ln in text.splitlines()]
    # Delimiter = the one most common in the busiest lines (title rows above the table have none)
    delim = max((",", "\t", ";"), key=lambda d: sorted((ln.count(d) for ln in lines), reverse=True)[:5])
    rows = list(csv.reader(lines, delimiter=delim))
    width = max((len(r) for r in rows), default=0)
    rows = [r + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, dtype=object)


def _keywords(cols_cfg: dict) -> list[tuple[str, str]]:
    """Flat list of (role, keyword) in priority order: time/meta roles first, then parameters."""
    pairs = []
    for role in ("datetime", "date", "hour", "depth", "station"):
        pairs += [(role, norm(k)) for k in cols_cfg[role]["keywords"]]
    for param, spec in cols_cfg["params"].items():
        pairs += [(param, norm(k)) for k in spec["keywords"]]
    return pairs


def find_header_row(raw: pd.DataFrame, cols_cfg: dict, scan: int = 30) -> int:
    """Index of the row that looks most like a header (most cells matching a keyword)."""
    kws = [k for _, k in _keywords(cols_cfg)]
    best, best_hits = 0, -1
    for i in range(min(scan, len(raw))):
        hits = sum(any(k in norm(c) for k in kws) for c in raw.iloc[i].tolist() if isinstance(c, str))
        if hits > best_hits:
            best, best_hits = i, hits
    return best


def classify_columns(headers: list[str], cols_cfg: dict) -> dict[int, tuple[str, str | None]]:
    """Map column index → (role, layer). Each column gets at most one role."""
    pairs = _keywords(cols_cfg)
    out: dict[int, tuple[str, str | None]] = {}
    for i, h in enumerate(headers):
        h_n = norm(h)
        if not h_n or h_n == "nan":
            continue
        for role, kw in pairs:
            if kw and kw in h_n:
                layer = next((v for k, v in LAYERS.items() if k in h_n), None)
                out[i] = (role, layer)
                break
    return out


def _to_float(series: pd.Series) -> pd.Series:
    """Convert a text column to numbers: Japanese/English missing markers → NaN, quality flags such as '12.3*' stripped."""
    s = series.astype(str).map(norm)
    s = s.where(~s.isin({norm(m) for m in MISSING}))
    s = s.str.replace(r"[^0-9eE+\-.]", "", regex=True)  # drop flags such as "12.3*"
    return pd.to_numeric(s, errors="coerce")


def _parse_time(df: pd.DataFrame, roles: dict[int, tuple[str, str | None]]) -> pd.Series:
    """Build one timezone-aware JST timestamp column from the date/time columns found in the file (24:00 → next day 00:00)."""
    by_role = {r: i for i, (r, _) in roles.items()}
    if "datetime" in by_role:
        raw = df.iloc[:, by_role["datetime"]].astype(str).map(norm)
        # hour 24 → 00 of the next day
        is24 = raw.str.contains(r"\b24:00")
        t = pd.to_datetime(raw.str.replace(r"\b24:00", "00:00", regex=True), errors="coerce")
        t = t + pd.to_timedelta(is24.astype(int), unit="D")
    elif "date" in by_role:
        d = pd.to_datetime(df.iloc[:, by_role["date"]].astype(str).map(norm), errors="coerce")
        h = _to_float(df.iloc[:, by_role["hour"]]) if "hour" in by_role else pd.Series(0, index=df.index)
        t = d + pd.to_timedelta(h.fillna(0), unit="h")
    else:
        raise ValueError("No date/time column recognised; add its header to config/insitu_columns.yaml")
    return t.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")


def parse_file(path: str | Path, station: str | None = None, cols_cfg: dict | None = None) -> pd.DataFrame:
    """Parse one file into the long format. ``station`` is used when the file has no station column."""
    cols_cfg = cols_cfg or config.load("insitu_columns")
    raw = read_raw(path)
    h = find_header_row(raw, cols_cfg)
    headers = [str(x) for x in raw.iloc[h].tolist()]
    df = raw.iloc[h + 1 :].reset_index(drop=True)
    df = df.dropna(how="all")
    roles = classify_columns(headers, cols_cfg)
    params = {i: (r, lay) for i, (r, lay) in roles.items() if r in cols_cfg["params"]}
    if not params:
        raise ValueError(f"No parameter columns recognised in {path}; headers were: {headers}")

    time = _parse_time(df, roles)
    by_role = {r: i for i, (r, _) in roles.items()}
    stn = (
        df.iloc[:, by_role["station"]].map(norm)
        if "station" in by_role
        else pd.Series(station, index=df.index)
    )
    depth = (
        _to_float(df.iloc[:, by_role["depth"]]) if "depth" in by_role else pd.Series(np.nan, index=df.index)
    )

    frames = []
    for i, (param, layer) in params.items():
        spec = cols_cfg["params"][param]
        vals = _to_float(df.iloc[:, i])
        lo, hi = spec["valid_range"]
        qc = np.where(vals.between(lo, hi), "ok", "out_of_range")
        frames.append(
            pd.DataFrame(
                {
                    "station": stn,
                    "time": time,
                    "layer": layer,
                    "depth_m": depth,
                    "param": param,
                    "value": vals,
                    "unit": spec["unit"],
                    "qc": qc,
                    "source_file": Path(path).name,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["time", "value"])
    if out["station"].isna().all():
        raise ValueError("Station unknown: pass station=... for files without a station column")
    return out.sort_values(["station", "time", "param"]).reset_index(drop=True)


def parse_many(paths, station_from_filename: bool = True) -> pd.DataFrame:
    """Parse several files; if a file has no station column, the file stem is used as station id."""
    frames = []
    for p in paths:
        p = Path(p)
        frames.append(parse_file(p, station=re.sub(r"\W+", "_", p.stem) if station_from_filename else None))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
