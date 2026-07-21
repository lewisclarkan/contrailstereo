"""Correlation module"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from ..config import StereoConfig, DEFAULT, SAT_LON
from ..geometry import apparent_surface_latlon
from .fields import hp_km

def project_pair(hkm, glat, glon, samplers, sig, cfg: StereoConfig = DEFAULT,
                 winds=None, offset_s=0.0):
    """High-passed field pair at assumed height hkm.

    The later satellite's sampling grid is displaced by wind(hkm)*offset
    BEFORE parallax projection (advection of the cloud, not of its ground
    projection -- order matters at oblique VZA).
    """
    if winds is not None and offset_s != 0.0:
        u, v = winds[0](hkm), winds[1](hkm)
        g2lat = glat + v * offset_s / 111.0e3
        g2lon = glon + u * offset_s / (111.0e3 * np.cos(np.deg2rad(glat)))
    else:
        g2lat, g2lon = glat, glon

    alat, alon = apparent_surface_latlon(glat, glon, hkm * 1000.0,
                                         SAT_LON[cfg.sat_east])
    A = hp_km(samplers[cfg.sat_east](alat, alon), sig)
    alat, alon = apparent_surface_latlon(g2lat, g2lon, hkm * 1000.0,
                                         SAT_LON[cfg.sat_west])
    B = hp_km(samplers[cfg.sat_west](alat, alon), sig)
    return A, B


def scene_correlation(A, B, winmask, min_px):
    """(scene-wide r, window r, n window px, finite fraction)."""
    m = np.isfinite(A) & np.isfinite(B)
    sc = np.corrcoef(A[m], B[m])[0, 1] if m.sum() > 100 else np.nan
    ml = m & winmask
    lo = np.corrcoef(A[ml], B[ml])[0, 1] if ml.sum() >= min_px else np.nan
    return sc, lo, int(ml.sum()), float(m.mean())


def local_corr(A, B, win):
    """Moving-window Pearson r at every pixel (NaN-aware via weights).

    Returns (r map, local G16 amplitude map). Windows with <70% valid
    support return NaN.
    """
    wgt = (np.isfinite(A) & np.isfinite(B)).astype(float)
    A0, B0 = np.nan_to_num(A), np.nan_to_num(B)
    w = np.maximum(uniform_filter(wgt, win), 1e-9)
    mean = lambda x: uniform_filter(x, win) / w
    mA, mB = mean(A0), mean(B0)
    cAB = mean(A0 * B0) - mA * mB
    vA, vB = mean(A0**2) - mA**2, mean(B0**2) - mB**2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cAB / np.sqrt(vA * vB)
    r[uniform_filter(wgt, win) < 0.7] = np.nan
    return r, np.sqrt(np.maximum(vA, 0))