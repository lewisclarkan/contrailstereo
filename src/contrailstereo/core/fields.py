"""Fields processing: BTW sampling and spatial filtering"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Proj
from scipy.ndimage import gaussian_filter

from ..config import R_EQ, R_POL


def make_sampler(ds14, ds15):
    """Bilinear sampler of the C14-C15 BTD on one satellite's grid
    
    Returns f(glat, glon) -> BTD [K]; NaN outside the sector"""

    pj = ds14["goes_imager_projection"]
    sat_lon = float(pj.attrs["longitude_of_projection_origin"])
    H = float(pj.attrs["perspective_point_height"])

    p = Proj(proj="geos", h=H, lon_0=sat_lon, sweep=pj.attrs.get("sweep_angle_axis", "x"), a=R_EQ, b=R_POL)

    d14 = ds14["CMI"].assign_coords(x=ds14.x * H, y=ds14.y * H)
    d15 = ds15["CMI"].assign_coords(x=ds15.x * H, y=ds15.y * H)

    def f(glat, glon):
        xm, ym = p(glon, glat)
        xm = xr.DataArray(xm, dims=("j", "i"))
        ym = xr.DataArray(ym, dims=("j", "i"))
        return d14.interp(x=xm, y=ym).values - d15.interp(x=xm, y=ym).values
    
    return f


def hp_km(f, sig):
    """NaN-aware high pass"""

    f = np.asarray(f, float)
    m = np.isfinite(f)
    sm = gaussian_filter(np.where(m, f, 0.0), sig)
    nm = gaussian_filter(m.astype(float), sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = sm / nm
    out = f - sm
    out[~m] = np.nan
    return out


def make_bandpass(sig_lo, sig_hi):
    """Band-pass prep: low-pass at sig_lo (kills pixel-scale grain), then
    high-pass at sig_hi. sig_lo=(0,0) reduces exactly to hp_km.
    """
    
    def prep(f):
        f = np.asarray(f, float)
        if max(sig_lo) > 0:
            m = np.isfinite(f)
            sm = gaussian_filter(np.where(m, f, 0.0), sig_lo)
            nm = gaussian_filter(m.astype(float), sig_lo)
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(m, sm / nm, np.nan)
        return hp_km(f, sig_hi)
    return prep