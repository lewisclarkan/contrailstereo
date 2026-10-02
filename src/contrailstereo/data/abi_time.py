"""Per-pixel scan times for ABI, from the ABI Time Model look-up tables."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import netCDF4
import numpy as np
import pandas as pd
from pyproj import Proj

from ..config import ABI_TIME_VERSION, R_EQ, R_POL
from . import goes

DIR_FMT = "ABI_Time_Model_{version}"
DEFAULT_VERSION = ABI_TIME_VERSION

# Detector offsets within a scan [s] (README of the Time Model), relative to
# which the tables are computed (Band 2).
BAND_OFFSET_S = {
    1: 0.179, 2: -0.055, 3: 0.402, 4: 0.642, 5: -0.359, 6: -0.642, 7: 0.535,
    8: 0.267, 9: 0.000, 10: -0.267, 11: -0.535, 12: -0.542, 13: 0.551,
    14: 0.319, 15: -0.256, 16: 0.579}

# Timelines (file names inside the version folder).
F_M6_G16 = "ABI-Timeline05B_Mode 6A_20190612-183017.nc"     # G16, from 2019-06-12 18:30:17
F_M6_G16_OLD = "Timeline05F_Mode6_20180828-092941.nc"       # G16 Mode 6 before that
F_M6_G17 = "340_Timeline_05M_Mode6_v2.7.nc"                 # G17 normal
F_M3 = "Timeline03C_Mode3_20180828-092953.nc"               # Mode 3 (has CONUS1-3)
F_M4 = "ABI-Timeline04A_Mode 4_20181219-104006.nc"
G16_6A_START = pd.Timestamp("2019-06-12 18:30:17")
EAST, WEST = (16, 19), (17, 18)


def scan_mode(ds) -> int:
    """ABI scan mode (3, 4 or 6) from the product's own names.

    Looks at dataset_name, filename, id (e.g. ``OR_ABI-L2-CMIPC-M6C13_G16_s...``)
    and timeline_id (``ABI Mode 6``), as variables or attributes -- goes2go
    moves attributes into variables, so both are searched."""
    tried = []
    for name in ("dataset_name", "filename", "id", "timeline_id"):
        vals = []
        if name in ds.variables:
            vals.append(np.asarray(ds[name].values).ravel()[0])
        if name in ds.attrs:
            vals.append(ds.attrs[name])
        for v in vals:
            v = v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)
            tried.append(f"{name}={v!r}")
            m = re.search(r"-M(\d)C\d\d", v) or re.search(r"Mode\s*(\d)", v)
            if m:
                return int(m.group(1))
    raise LookupError("cannot tell the ABI scan mode: none of dataset_name / "
                      f"filename / id / timeline_id says it (saw {tried or 'none'})")


def lut_filename(sat: int, mode: int, domain: str, start) -> str:
    """The Time Model table for this product.

    CONUS (domain "C"): every Mode 6 timeline shipped has an identical CONUS
    table, so the choice is immaterial there (G18 and G19 included). Mode 3 uses its own (3 CONUS
    instances). The G17 cooling timelines have no CONUS table, and G17
    full-disk Mode 3 is ambiguous between the normal and cooling timelines.
    Mode 4: contingency mode
    Anything else raises LookupError; pass ``abi_time_file`` to choose."""
    start = pd.Timestamp(start)
    opts = f"sat={sat} mode={mode} domain={domain!r}"
    if domain not in ("C", "F"):
        raise LookupError(f"no Time Model for {opts} (only C and F are supported)")
    if mode == 6:
        if domain == "F" and sat not in (16, 17):
            raise LookupError(f"{opts}: the full-disk tables shipped are for G16 "
                              "and G17 only; set abi_time_file")
        if sat in EAST:
            old = domain == "F" and start < G16_6A_START
            return F_M6_G16_OLD if old else F_M6_G16
        if sat in WEST:
            return F_M6_G17
    elif mode == 3:
        if domain == "C":
            return F_M3
        if sat in EAST:
            return F_M3
        raise LookupError(f"{opts}: full-disk Mode 3 on the West satellite may be "
                          "the normal or the cooling timeline; set abi_time_file")
    elif mode == 4:
        return F_M4
    raise LookupError(f"no Time Model table known for {opts}; set abi_time_file")


def lut_path(abi_dir, version: str, filename: str) -> Path:
    p = Path(abi_dir) / DIR_FMT.format(version=version) / filename
    if not p.exists():
        raise FileNotFoundError(
            f"ABI Time Model table not found: {p}\n"
            f"Expected the unzipped ABI_Time_Model_{version} folder under "
            f"{Path(abi_dir)} (set [paths] abi_time, or CONTRAILSTEREO_ABI_TIME).")
    return p


@lru_cache(maxsize=6)
def _read(path: str, var: str) -> np.ndarray:
    with netCDF4.Dataset(path) as d:
        if var not in d.variables:
            have = [v for v in d.variables if v.endswith("_pixel_times")]
            raise LookupError(f"{Path(path).name} has no {var} (has {have})")
        d.set_auto_mask(False)
        a = np.ascontiguousarray(np.asarray(d[var][:], dtype=np.float32).T)
    a.setflags(write=False)
    return a                                      # [row, col]


def n_conus(path) -> int:
    with netCDF4.Dataset(str(path)) as d:
        return sum(1 for v in d.variables if re.fullmatch(r"CONUS\d+_pixel_times", v))


def load_lut(path, domain: str, start) -> np.ndarray:
    """[row, col] seconds after the first product pixel (Band 2).

    CONUS: instance k of the timeline is picked from the scan's start minute
    (5 min apart: Mode 6 has two, Mode 3 three; identical in Mode 6)."""
    path = str(path)
    if domain == "F":
        return _read(path, "FD_pixel_times")
    n = n_conus(path)
    if n == 0:
        raise LookupError(f"{Path(path).name} has no CONUS table (cooling timelines "
                          "are full disk only)")
    k = (pd.Timestamp(start).minute % (5 * n)) // 5 + 1
    return _read(path, f"CONUS{k}_pixel_times")


def crop_to_product(fd: np.ndarray, ds) -> np.ndarray:
    """The part of a full-disk table covered by a product cut from that scan.

    The full-disk fixed grid is centred on the sub-satellite point, so the
    product's x/y axes give its offset in the table. Raises if the offset is
    not a whole number of pixels or the product does not fit in the table."""
    ny, nx = fd.shape
    xs, ys = (np.asarray(ds[k].values, float) for k in ("x", "y"))
    dx, dy = (xs[-1] - xs[0]) / (xs.size - 1), (ys[-1] - ys[0]) / (ys.size - 1)
    c0, r0 = xs[0] / dx + (nx - 1) / 2, ys[0] / dy + (ny - 1) / 2
    if abs(c0 - round(c0)) > 0.05 or abs(r0 - round(r0)) > 0.05:
        raise ValueError(f"product grid is not aligned to the full-disk table "
                         f"(offset {r0:.2f} rows, {c0:.2f} cols)")
    r0, c0 = int(round(r0)), int(round(c0))
    h, w = ds["y"].size, ds["x"].size
    if r0 < 0 or c0 < 0 or r0 + h > ny or c0 + w > nx:
        raise ValueError(f"product ({h}x{w}) at row {r0}, col {c0} falls outside "
                         f"the {ny}x{nx} full-disk table")
    return np.ascontiguousarray(fd[r0:r0 + h, c0:c0 + w])


def band_shift_s(channels) -> float:
    """Mean detector offset of the channels minus Band 2's, added to the table."""
    return float(np.mean([BAND_OFFSET_S[int(c)] for c in channels]) - BAND_OFFSET_S[2])


# ----------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PixelClock:
    """Scan time of every pixel of one product.

    start      : product start (time_bounds[0]).
    lut        : [row, col] seconds after start, Band 2 (shared, read-only).
    shift_s    : channel offset added to the table (see band_shift_s).
    since_start(lat, lon) : seconds after ``start`` at which the pixel that
                 images the APPARENT position (lat, lon) was scanned; NaN off
                 the sector.
    info       : provenance (file, mode, window vs table span)."""

    start: pd.Timestamp
    lut: np.ndarray
    shift_s: float
    _locate: Callable
    info: dict

    def since_start(self, lat, lon, clip=False) -> np.ndarray:
        row, col = self._locate(lat, lon)
        ny, nx = self.lut.shape
        ok = np.isfinite(row) & np.isfinite(col)
        if not clip:
            ok &= (row > -0.5) & (row < ny - 0.5) & (col > -0.5) & (col < nx - 0.5)
        i = np.clip(np.rint(np.where(ok, row, 0)), 0, ny - 1).astype(int)
        j = np.clip(np.rint(np.where(ok, col, 0)), 0, nx - 1).astype(int)
        return np.where(ok, self.lut[i, j].astype(float) + self.shift_s, np.nan)

    def time_at(self, lat, lon, clip=False):
        """Scan time as Timestamps (scalar input) -- for diagnostics."""
        s = float(np.atleast_1d(self.since_start(lat, lon, clip))[0])
        return self.start + pd.to_timedelta(s, unit="s")


def _locator(ds) -> Callable:
    """(lat, lon) -> fractional (row, col) on the product's fixed grid."""
    pj = ds["goes_imager_projection"]
    H = float(pj.attrs["perspective_point_height"])
    p = Proj(proj="geos", h=H, lon_0=float(pj.attrs["longitude_of_projection_origin"]),
             sweep=pj.attrs.get("sweep_angle_axis", "x"), a=R_EQ, b=R_POL)
    xs, ys = (np.asarray(ds[k].values, float) for k in ("x", "y"))
    dx, dy = (xs[-1] - xs[0]) / (xs.size - 1), (ys[-1] - ys[0]) / (ys.size - 1)

    def locate(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        xm, ym = p(lon.ravel(), lat.ravel(), errcheck=False)
        xm, ym = np.asarray(xm, float), np.asarray(ym, float)
        bad = ~(np.isfinite(xm) & np.isfinite(ym) & (np.abs(xm) < 1e20) & (np.abs(ym) < 1e20))
        col = np.where(bad, np.nan, (xm / H - xs[0]) / dx)
        row = np.where(bad, np.nan, (ym / H - ys[0]) / dy)
        return row.reshape(lat.shape), col.reshape(lat.shape)

    return locate


def pixel_clock(ds, sat: int, domain: str, channels, abi_dir, version=DEFAULT_VERSION,
                filename: str = "") -> PixelClock:
    """The PixelClock of one CMIP product (any one of the channels; they share
    a scan). Raises if the scan window, mode or table cannot be established
    or the table does not fit the product."""
    win = goes.scan_window(ds)
    if win is None:
        raise RuntimeError("no scan window (time_bounds / time_coverage_*) in the "
                           "product; refusing to guess the pixel times")
    start, end = win
    mode = scan_mode(ds)
    must_match = False
    try:
        fname = filename or lut_filename(sat, mode, domain, start)
    except LookupError:
        # G17 full-disk Mode 3 is the normal timeline or a cooling one: take the
        # normal table, but only if the product's own scan window agrees with it
        if filename or not (sat in WEST and mode == 3 and domain == "F"):
            raise
        fname, must_match = F_M3, True
    path = lut_path(abi_dir, version, fname)
    cut_from_fd = domain == "C" and n_conus(path) == 0      # e.g. Mode 4: no CONUS table
    lut = load_lut(path, "F" if cut_from_fd else domain, start)
    span = float(lut.max() - lut.min())
    if cut_from_fd:
        lut = crop_to_product(lut, ds)
    ny, nx = ds["y"].size, ds["x"].size
    if lut.shape != (ny, nx):
        raise ValueError(f"{fname}: table is {lut.shape} but the G{sat} product is "
                         f"{(ny, nx)} ({domain!r}, mode {mode})")
    window_s = (end - start).total_seconds()
    if must_match and abs(window_s - span) > 10.0:
        raise LookupError(f"G{sat} full-disk Mode 3: the product window ({window_s:.0f} s) "
                          f"does not match the normal table ({span:.0f} s), so this is "
                          "probably a cooling timeline; set abi_time_file")
    if abs(window_s - span) > 10.0:
        warnings.warn(f"G{sat}: product window is {window_s:.0f} s but the table "
                      f"spans {span:.0f} s -- wrong timeline?")
    info = dict(file=fname, version=version, mode=mode, window_s=window_s,
                table_span_s=span, cut_from_full_disk=cut_from_fd)
    return PixelClock(start=start, lut=lut, shift_s=band_shift_s(channels),
                      _locate=_locator(ds), info=info)