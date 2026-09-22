""" Field preparation module 

1. Sampling - ``btd_view`` gives a callable that samples the BTD
2. Conditioning - ``make_preps`` returns the function that turns 
   a sampled field into the field that is correlated:
   
   "hp": Gaussian high-pass
   "psf_iso": blur each view to a common isotropic footprint, then high-pass
   "psf_minmtf": blur each view to the pointwise minimym MTF of the pair 
                 (anisotropic), then high-pass.
   
"""

from __future__ import annotations
 
from typing import Callable
 
import numpy as np
import xarray as xr
from pyproj import Proj
from scipy.ndimage import convolve, gaussian_filter
 
from .config import R_EQ, R_POL, SAT_LON, StereoConfig, DEFAULT
from .geometry import km_filters, view_geometry
from .types import Grid, View
 
ABI_IFOV = 56.0e-6          # rad; ~2 km at nadir for the ABI IR bands
 
Prep = Callable[[np.ndarray], np.ndarray]

# ---------- Sampling ---------- 

def btd_view(ds_a: xr.Dataset, ds_b: xr.Dataset) -> View:
    """Bilinear sampler of CMI(a) - CMI(b) on one satellite's fixed grid.
 
    ds_a, ds_b : ABI L2 CMIP datasets from the same scan (see
                 data.goes.fetch_channels).
    Returns f(lat, lon) -> BTD [K], same shape as the inputs; NaN outside
    the sector or where either channel is masked.
    """
    pj = ds_a["goes_imager_projection"]
    H = float(pj.attrs["perspective_point_height"])
    p = Proj(proj="geos", h=H,
             lon_0=float(pj.attrs["longitude_of_projection_origin"]),
             sweep=pj.attrs.get("sweep_angle_axis", "x"), a=R_EQ, b=R_POL)
    da = ds_a["CMI"].assign_coords(x=ds_a.x * H, y=ds_a.y * H)
    db = ds_b["CMI"].assign_coords(x=ds_b.x * H, y=ds_b.y * H)
 
    def view(lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        shape = lat.shape
        xm, ym = p(lon.ravel(), lat.ravel(), errcheck=False)
        xm = np.where(np.isfinite(xm) & (np.abs(xm) < 1e20), xm, np.nan)
        ym = np.where(np.isfinite(ym) & (np.abs(ym) < 1e20), ym, np.nan)
        xs = xr.DataArray(xm, dims="p")
        ys = xr.DataArray(ym, dims="p")
        out = (da.interp(x=xs, y=ys).values - db.interp(x=xs, y=ys).values)
        return out.reshape(shape)
 
    return view


# ---------- High pass ----------

def highpass(f, sig_px) -> np.ndarray:
    """NaN-aware Gaussian high-pass: f minus its mask-normalised Gaussian
    smooth at sig_px (per-axis sigma in pixels). NaNs stay NaN and do not
    bleed into their neighbours."""
    f = np.asarray(f, float)
    m = np.isfinite(f)
    sm = gaussian_filter(np.where(m, f, 0.0), sig_px)
    nm = gaussian_filter(m.astype(float), sig_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = f - sm / nm
    out[~m] = np.nan
    return out


# ---------- Footprint model (experimental) ----------


def _rot_cov(sig_along, sig_across, azi_deg):
    """Covariance [km^2] in a local (north, east) frame."""
    a = np.deg2rad(azi_deg)
    u = np.array([np.cos(a), np.sin(a)])          # along line of sight
    v = np.array([-np.sin(a), np.cos(a)])
    return sig_along**2 * np.outer(u, u) + sig_across**2 * np.outer(v, v)
 
 
def footprint(lat, lon, sat_lon, psf_factor=0.45, dz_km=0.0, ifov=ABI_IFOV):
    """Effective PSF of one view at one point.
 
    psf_factor : sigma / GSD. ~0.29 is a box IFOV; ~0.45 is a reasonable
                 ABI MTF + fixed-grid resampling value. Mostly the DIFFERENCE
                 between the views matters; this sets the absolute blur cost.
    dz_km      : optional layer thickness, adding a slant-path smear
                 dz * tan(VZA) (uniform-equivalent sigma = smear / sqrt 12)
                 along the line of sight.
 
    Returns (cov, info): cov [km^2] in (north, east); info has vza_deg,
    azi_deg, slant_km, gsd_along_km, gsd_across_km.
    """
    g = view_geometry(float(lat), float(lon), sat_lon)
    if g["vza_deg"] > 85.0:
        raise ValueError(f"VZA {g['vza_deg']:.1f} > 85 deg: along-view "
                         "footprint is unbounded")
    across = ifov * g["slant_km"]
    along = across / np.cos(np.deg2rad(g["vza_deg"]))
    s_al, s_ac = psf_factor * along, psf_factor * across
    if dz_km > 0:
        smear = dz_km * np.tan(np.deg2rad(g["vza_deg"]))
        s_al = np.hypot(s_al, smear / np.sqrt(12.0))
    info = dict(g, gsd_along_km=float(along), gsd_across_km=float(across))
    return _rot_cov(s_al, s_ac, g["azi_deg"]), info
 
 
def gauss_kernel(cov, px_km, nsig=3.0) -> np.ndarray:
    """Normalised N(0, cov) sampled on the grid's (i=north, j=east) pixels;
    a 1x1 delta if cov is negligible."""
    cov = np.asarray(cov, float)
    w = np.linalg.eigvalsh(cov)
    if w.max() <= 1e-6:
        return np.ones((1, 1))
    if w.min() < -1e-8:
        raise ValueError("kernel covariance is not positive semi-definite")
    cov = cov + 1e-6 * np.eye(2)
    hi = int(max(1, np.ceil(nsig * np.sqrt(cov[0, 0]) / px_km[0])))
    hj = int(max(1, np.ceil(nsig * np.sqrt(cov[1, 1]) / px_km[1])))
    N, E = np.meshgrid(np.arange(-hi, hi + 1) * px_km[0],
                       np.arange(-hj, hj + 1) * px_km[1], indexing="ij")
    P = np.linalg.inv(cov)
    K = np.exp(-0.5 * (P[0, 0] * N**2 + 2 * P[0, 1] * N * E + P[1, 1] * E**2))
    return K / K.sum()
 
 
def nan_convolve(f, K) -> np.ndarray:
    """Mask-normalised convolution; NaNs stay NaN (no data is invented)."""
    f = np.asarray(f, float)
    if K.shape == (1, 1):
        return f.copy()
    m = np.isfinite(f)
    num = convolve(np.where(m, f, 0.0), K, mode="nearest")
    den = convolve(m.astype(float), K, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out
 
 
def min_mtf_filters(shape, px_km, covs) -> dict:
    """Fourier filters taking each view to the pointwise-minimum MTF.
 
    With Gaussian footprints MTF_i(k) = exp(-k^T Sigma_i k / 2), so
    F_i = min_j MTF_j / MTF_i = exp(-(max_j q_j - q_i) / 2) <= 1: a pure
    blur for every view. {sat: filter on the FFT grid of `shape`}.
    """
    ky = np.fft.fftfreq(shape[0], d=px_km[0]) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=px_km[1]) * 2 * np.pi
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    q = {s: C[0, 0] * KY**2 + 2 * C[0, 1] * KY * KX + C[1, 1] * KX**2
         for s, C in covs.items()}
    qmax = np.maximum.reduce(list(q.values()))
    return {s: np.exp(-0.5 * (qmax - q[s])) for s in q}
 
 
def fft_filter(f, F, pad) -> np.ndarray:
    """Mask-normalised Fourier filter. The field is NaN-padded by `pad`
    pixels per side first, so the FFT's circular wrap cannot carry one
    edge's structure onto the other; F must match the padded shape."""
    f = np.pad(np.asarray(f, float), pad, constant_values=np.nan)
    m = np.isfinite(f)
    num = np.fft.ifft2(np.fft.fft2(np.where(m, f, 0.0)) * F).real
    den = np.fft.ifft2(np.fft.fft2(m.astype(float)) * F).real
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.where(np.abs(den) > 1e-3, den, np.nan)
    out[~m] = np.nan
    return out[pad[0]:out.shape[0] - pad[0], pad[1]:out.shape[1] - pad[1]]

# How much height information


def observability(field, grid: Grid, sats, cfg: StereoConfig = DEFAULT,
                  deriv_sigma_px=1.5) -> dict:
    """How much height information the local structure carries for a pair.
 
    Stereo measures height only through displacement along the disparity
    axis k (km of separation per km of height). A linear feature -- a
    contrail -- parallel to k looks the same after that displacement, so
    its correlation-vs-height curve is flat: high r, no height information.
    Only the component of k across the feature counts.
 
    Local orientation and linearity come from the structure tensor of the
    high-passed field, averaged over ~the matching window:
        coherence : 0 isotropic texture ... 1 perfectly linear
        angle_deg : angle between the local feature and the disparity axis
        s_eff     : effective sensitivity [km of cross-feature shift per km
                    of height] = |k| sqrt(coh cos^2(phi - psi) + (1 - coh)/2),
                    phi = gradient direction, psi = axis direction.
                    |k| for a feature perpendicular to the axis, 0 parallel,
                    |k|/sqrt 2 for isotropic texture.
 
    field : 2-D field on grid (e.g. the reference view's BTD).
    sats  : (sat_a, sat_b) defining the disparity axis.
    Returns dict of 2-D arrays (NaN where field is NaN) plus k_km_per_km
    and axis_deg (bearing, clockwise from north).
    """
    from .geometry import disparity_per_km
    if grid.kind != "latlon":
        raise NotImplementedError("observability assumes north/east grid axes")
    sig_px, win_px = km_filters(grid, cfg)
    f = highpass(field, sig_px)
    m = np.isfinite(f)
    f0 = np.where(m, f, 0.0)
    gn = gaussian_filter(f0, deriv_sigma_px, order=(1, 0)) / grid.px_km[0]
    ge = gaussian_filter(f0, deriv_sigma_px, order=(0, 1)) / grid.px_km[1]
    s = (win_px[0] / 4.0, win_px[1] / 4.0)
    Jee, Jnn, Jen = (gaussian_filter(a, s) for a in (ge * ge, gn * gn, ge * gn))
    tr = Jee + Jnn
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.sqrt((Jee - Jnn) ** 2 + 4 * Jen ** 2) / tr
    phi = 0.5 * np.arctan2(2 * Jen, Jee - Jnn)          # gradient, from east
    lat0, lon0 = grid.bbox.center
    kn, ke = disparity_per_km(lat0, lon0, *sats)
    k, psi = float(np.hypot(kn, ke)), float(np.arctan2(kn, ke))
    c2 = np.cos(phi - psi) ** 2
    s_eff = k * np.sqrt(np.clip(coh * c2 + (1 - coh) * 0.5, 0, None))
    # sin(feature-axis angle) = |cos(gradient-axis angle)|
    ang = np.degrees(np.arcsin(np.clip(np.sqrt(c2), 0, 1)))
    out = dict(s_eff=s_eff, coherence=coh, angle_deg=ang)
    for key in out:
        out[key] = np.where(m, out[key], np.nan)
    out.update(k_km_per_km=k, axis_deg=float(np.degrees(np.arctan2(ke, kn)) % 180))
    return out
 
 
# ---------- Entrypoint ----------

def make_preps(grid: Grid, sats, cfg: StereoConfig = DEFAULT):
    """Per-satellite conditioning functions for one case.
 
    grid : the case's retrieval grid (PSF geometry is evaluated at its
           centre; its px_km sets the kernel sampling).
    sats : satellite ids, e.g. (cfg.sat_east, cfg.sat_west).
 
    Returns (preps, info):
        preps : {sat: f(field) -> field}, applied to every sampled field
        info  : flat, JSON-serialisable diagnostics for Result.diag
                (empty for "hp"; footprint geometry and target for psf_*).
    """
    sig_px, _ = km_filters(grid, cfg)
    if cfg.prep == "hp":
        hp = lambda f: highpass(f, sig_px)
        return {s: hp for s in sats}, {}
 
    if grid.kind != "latlon":
        raise NotImplementedError(
            "PSF kernels assume grid axes point north/east (latlon grid)")
    lat0, lon0 = grid.bbox.center
    covs, info = {}, {}
    for s in sats:
        covs[s], g = footprint(lat0, lon0, SAT_LON[s], cfg.psf_factor,
                               cfg.psf_dz_km)
        for k in ("vza_deg", "azi_deg", "gsd_along_km", "gsd_across_km"):
            info[f"psf_{k}_{s}"] = g[k]
 
    if cfg.prep == "psf_iso":
        s_t = cfg.psf_inflate * max(
            float(np.sqrt(np.linalg.eigvalsh(C).max())) for C in covs.values())
        CT = s_t**2 * np.eye(2)
        kernels = {s: gauss_kernel(CT - covs[s], grid.px_km) for s in sats}
        info["psf_sigma_t_km"] = s_t
        return ({s: (lambda f, K=K: highpass(nan_convolve(f, K), sig_px))
                 for s, K in kernels.items()}, info)
 
    if cfg.prep == "psf_minmtf":
        s_max = max(float(np.sqrt(np.linalg.eigvalsh(C).max()))
                    for C in covs.values())
        pad = tuple(int(np.ceil(4 * s_max / p)) for p in grid.px_km)
        shape = tuple(n + 2 * p for n, p in zip(grid.shape, pad))
        F = min_mtf_filters(shape, grid.px_km, covs)
        info["psf_pad_px"] = list(pad)
        return ({s: (lambda f, Fs=Fs: highpass(fft_filter(f, Fs, pad), sig_px))
                 for s, Fs in F.items()}, info)
 
    raise ValueError(f"unknown prep {cfg.prep!r}")