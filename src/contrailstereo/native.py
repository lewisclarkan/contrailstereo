"""Matching on the reference view's own pixels (cfg.grid_kind = "native").

The lat/lon-grid matcher resamples BOTH views onto a common grid, so the
reference view is interpolated too. Here the reference view A stays as
measured and only the other view B is resampled:

    for each trial height h
      A pixel -> its true position for a cloud at h        (inverse parallax)
              -> advected to the time B scans that cloud    (wind(h) x dt)
              -> parallax into B -> bilinear sample of B
      high-pass, local correlation in a window of native pixels whose
      ground area is cfg.win_km^2                            (geometry.NativeGrid.filters)
    -> height map on A's pixels                              (retrieve.height_map)

The map is then placed on the case's lat/lon grid (``native_to_grid``) at the
reference time, so validation and everything downstream see the same
Result as in the lat/lon path.

Times. With time_model "lut" each A pixel is at its own scan time and B is
read at the scan time of the pixel that images the advected cloud; with
"nominal" both are one number per product. The times enter as seconds
relative to the reference frame's time (Frame.time), which is also the time
the output map is valid at.

Not supported here (raise): prep other than "hp", tracking mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import StereoConfig, DEFAULT
from .geometry import (KM_PER_DEG, NativeGrid, advect_latlon,
                       apparent_surface_latlon, true_latlon_from_apparent)
from .prep import highpass
from .retrieve import (QC_REF_H_KM, height_map, height_scan, local_corr,
                       other_key, ref_key)
from .types import BBox, Frame, Grid, NativeMap, NativeRaster

PAD_REF_KM = 15.0        # crop margin around the reference view's footprint
PAD_OTHER_KM = 40.0      # ... and the other view's: it must also cover the advection
TOL_FRAC = 0.75          # placement tolerance, x the larger pixel spacing


# ======================================================================
# Inputs
# ======================================================================
def native_raster(da, db, sat, bbox, cfg: StereoConfig, clock, s0, is_ref) -> NativeRaster:
    """BTD crop of one view on its fixed grid, and the scan time of every
    pixel relative to the frame's time.

    da, db : the two channels' datasets (already destriped)
    clock  : PixelClock, or None for time_model "nominal" (times all 0)
    s0     : the frame time's offset from clock.start [s]"""
    bbox = BBox(*bbox)
    grid, (isl, jsl) = NativeGrid.from_dataset(
        da, sat, bbox, (cfg.h_lo_km, cfg.h_hi_km),
        pad_km=PAD_REF_KM if is_ref else PAD_OTHER_KM)
    btd = (da["CMI"].isel(y=isl, x=jsl).values.astype(np.float32)
           - db["CMI"].isel(y=isl, x=jsl).values.astype(np.float32))
    if clock is None:
        t_s = np.zeros(btd.shape, np.float32)
    else:
        t_s = (clock.lut[isl, jsl].astype(float) + clock.shift_s - s0)
    return NativeRaster(grid=grid, btd=btd, t_s=t_s)


# ======================================================================
# Sampling and projection
# ======================================================================
def sample_bilinear(Z, row, col):
    """Bilinear sample of Z at fractional (row, col); NaN off the raster or
    where a corner is NaN."""
    ny, nx = Z.shape
    ok = (np.isfinite(row) & np.isfinite(col) & (row >= 0) & (row <= ny - 1)
          & (col >= 0) & (col <= nx - 1))
    r, c = np.where(ok, row, 0.0), np.where(ok, col, 0.0)
    i0 = np.minimum(np.floor(r).astype(int), ny - 2)
    j0 = np.minimum(np.floor(c).astype(int), nx - 2)
    fr, fc = r - i0, c - j0
    out = ((1 - fr) * (1 - fc) * Z[i0, j0] + (1 - fr) * fc * Z[i0, j0 + 1]
           + fr * (1 - fc) * Z[i0 + 1, j0] + fr * fc * Z[i0 + 1, j0 + 1])
    return np.where(ok, out, np.nan)


def _t_lookup(tg, row, col, fill):
    """Time of the pixel nearest fractional (row, col); `fill` where undefined."""
    ny, nx = tg.shape
    ok = np.isfinite(row) & np.isfinite(col)
    i = np.clip(np.rint(np.where(ok, row, 0.0)), 0, ny - 1).astype(int)
    j = np.clip(np.rint(np.where(ok, col, 0.0)), 0, nx - 1).astype(int)
    return np.where(ok, tg[i, j], fill)

def project_to_other(tlat, tlon, h_km, t_ref, B: NativeGrid, t_oth, sat_lon_oth,
                     wind, per_pixel, n_read=2):
    """Fractional (row, col) in the other view's raster of a cloud at h_km
    that the reference view imaged at its TRUE position (tlat, tlon) at time
    t_ref.

    The cloud is advected to the time its other-view pixel is scanned
    (dt = t_oth - t_ref), then parallax-projected -- the cloud moves, then
    parallax, as in retrieve.project. That pixel depends on where the
    advected cloud appears, so with per_pixel the time is re-read at the
    projected position (n_read times; it only changes near a swath boundary).

    t_ref : scalar or array like tlat [s, relative to the reference time]
    t_oth : the other raster's per-pixel times (per_pixel) or a scalar
    Returns (row, col, dt)."""
    t_ref = np.asarray(t_ref, float)
    if wind is None:
        row, col = B.rc(*apparent_surface_latlon(tlat, tlon, h_km * 1e3, sat_lon_oth))
        return row, col, 0.0
    u, v = wind.at(h_km)
    fill = float(np.nanmedian(t_oth)) if per_pixel else float(t_oth)
    dt = fill - t_ref
    for it in range(n_read + 1):
        glat, glon = advect_latlon(tlat, tlon, u, v, dt)
        row, col = B.rc(*apparent_surface_latlon(glat, glon, h_km * 1e3, sat_lon_oth))
        if not per_pixel or it == n_read:
            break
        dt_new = _t_lookup(t_oth, row, col, fill) - t_ref
        if np.array_equal(dt_new, dt):
            break
        dt = dt_new
    return row, col, dt


# ======================================================================
# Matching
# ======================================================================
def match_native(ref: Frame, oth: Frame, cfg: StereoConfig = DEFAULT, wind=None,
                 bbox=None):
    """Height map on the reference view's native pixels.

    ref, oth : frames loaded with grid_kind "native" (Frame.raster set)
    bbox     : the case bbox; the overlap fraction is judged inside it
    Returns (NativeMap, valid_frac): valid_frac is the share of reference
    pixels inside bbox where both views are valid, at the scan height
    nearest QC_REF_H_KM (as retrieve.match_snapshot)."""
    if cfg.prep != "hp":
        raise NotImplementedError("grid_kind='native' supports prep='hp' only")
    A, B = ref.raster, oth.raster
    if A is None or B is None:
        raise ValueError("frames have no native raster: load them with "
                         "cfg.grid_kind='native'")
    per_pixel = ref.clock is not None
    sig, win, J = A.grid.filters(cfg)
    Ah = highpass(A.btd, sig)
    hs = height_scan(cfg)
    k_mid, k_ref = hs.size // 2, int(np.argmin(np.abs(hs - QC_REF_H_KM)))
    t_A = A.t_s.astype(float)                         # relative to ref.time
    dt_frames = (oth.time - ref.time).total_seconds()
    t_B = B.t_s.astype(float) + dt_frames if per_pixel else dt_frames

    Rc = np.full((hs.size,) + A.btd.shape, np.nan, np.float32)
    amp, valid = None, 0.0
    for k, h in enumerate(hs):
        tlat, tlon = A.grid.true_at(h)
        row, col, _ = project_to_other(tlat, tlon, h, t_A, B.grid, t_B,
                                       oth.sat_lon, wind, per_pixel)
        Bh = highpass(sample_bilinear(B.btd, row, col), sig)
        Rc[k], a = local_corr(Ah, Bh, win)
        if k == k_mid:
            amp = a
        if k == k_ref:
            both = np.isfinite(Ah) & np.isfinite(Bh)
            if bbox is not None:
                la0, la1, lo0, lo1 = bbox
                both = both[(tlat >= la0) & (tlat <= la1)
                            & (tlon >= lo0) & (tlon <= lo1)]
            valid = float(both.mean()) if both.size else 0.0
    H, rmax = height_map(hs, Rc, amp, cfg)
    return NativeMap(grid=A.grid, height=H.astype(np.float32),
                     r=rmax.astype(np.float32), amp=amp.astype(np.float32),
                     t_s=t_A.astype(np.float32), win_px=tuple(win)), valid


# ======================================================================
# Placement on the output grid
# ======================================================================
def native_to_grid(nm: NativeMap, grid: Grid, wind=None, tol_km=None) -> dict:
    """Place a native map on the lat/lon grid at the reference time.

    Each pixel goes to its TRUE position for its OWN retrieved height (no
    height is assumed), is carried from its scan time to the reference time
    by the wind at that height, and every grid point takes the nearest
    placed pixel within tol_km (default TOL_FRAC x the larger pixel spacing).
    Returns {"height", "r", "amp"} on the grid."""
    g = nm.grid
    ok = np.isfinite(nm.height)
    out = {k: np.full(grid.shape, np.nan) for k in ("height", "r", "amp")}
    if not ok.any():
        return out
    H = nm.height[ok].astype(float)
    tlat, tlon, _ = true_latlon_from_apparent(g.alat[ok], g.alon[ok], H * 1e3,
                                              g.sat_lon)
    if wind is not None:
        u, v = np.interp(H, wind.h, wind.u), np.interp(H, wind.h, wind.v)
        tlat, tlon = advect_latlon(tlat, tlon, u, v, -nm.t_s[ok].astype(float))
    lat0, lon0 = grid.bbox.center
    c = np.cos(np.deg2rad(lat0))
    tree = cKDTree(np.c_[(tlon - lon0) * KM_PER_DEG * c, (tlat - lat0) * KM_PER_DEG])
    q = np.c_[((grid.lon - lon0) * KM_PER_DEG * c).ravel(),
              ((grid.lat - lat0) * KM_PER_DEG).ravel()]
    if tol_km is None:
        tol_km = TOL_FRAC * max(g.jacobian_km()[1])
    d, idx = tree.query(q, distance_upper_bound=tol_km)
    hit = np.isfinite(d)
    for name, F in (("height", nm.height), ("r", nm.r), ("amp", nm.amp)):
        o = np.full(q.shape[0], np.nan)
        o[hit] = F[ok][idx[hit]]
        out[name] = o.reshape(grid.shape)
    return out


def diag(nm: NativeMap, cfg: StereoConfig = DEFAULT) -> dict:
    """Scalar diagnostics of a native map for Result.diag."""
    J, (si, sj) = nm.grid.jacobian_km()
    return dict(native_ny=int(nm.height.shape[0]), native_nx=int(nm.height.shape[1]),
                win_px_i=int(nm.win_px[0]), win_px_j=int(nm.win_px[1]),
                win_area_frac=float(nm.win_px[0] * nm.win_px[1]
                                    * abs(np.linalg.det(J)) / cfg.win_km ** 2),
                native_px_km_i=si, native_px_km_j=sj)