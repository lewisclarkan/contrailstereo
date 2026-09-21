"""Anisotropic footprint model and PSF homogenisation for oblique pairs.

Why this exists
---------------
Both views are resampled onto a common ~1 km lat/lon grid and then given
the SAME isotropic high-pass (``hp_km``).  That silently assumes the two
images carry the same effective resolution.  They do not.  A geostationary
IFOV projects to a ground footprint that is stretched by 1/cos(VZA) ALONG
the view azimuth:

    GSD_across = ifov * D(VZA)              (~2.0-2.2 km)
    GSD_along  = ifov * D(VZA) / cos(VZA)   (2.0 km at nadir, 6.5 km at 70 deg)

and the two satellites stretch along DIFFERENT azimuths (east-looking vs
west-looking).  At a 70/56 deg pair the along-view sample sizes are 6.5 and
3.8 km in near-opposing directions, so each view has smeared away structure
the other still resolves.  Linear correlation between them is then bounded
well below 1 no matter how good the height hypothesis is.

The fix (PSF homogenisation / matching kernels)
-----------------------------------------------
Model each footprint as a Gaussian with covariance Sigma_i (km^2) in a
local north/east frame.  Choose a common ISOTROPIC target Sigma_T =
sigma_t^2 I with sigma_t at least the worst along-view sigma of the pair.
Then convolve view i with the matching kernel

    M_i = N(0, Sigma_T - Sigma_i)

so that both views end up with PSF exactly Sigma_T.  Because the target is
larger than both footprints in every direction, Sigma_T - Sigma_i is
positive definite and every matching kernel is a pure blur -- no
deconvolution, no noise amplification, no striping enhancement.

Cost: the pair's effective resolution drops to sigma_t, so the correlation
peak broadens.  That is the trade to measure, not to assume.

Optionally the layer's own vertical extent can be folded in: a deck of
thickness dz seen at VZA smears by dz*tan(VZA) along the same azimuth,
which is the same anisotropy with a larger amplitude (2.7 km per km of
depth at 70 deg).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

from ..config import R_EQ, SAT_R

ABI_IFOV = 56.0e-6          # rad; 2 km at nadir for the IR bands
R_MEAN_KM = 6371.0


# ------------------------------------------------------------------ geometry

def _central_angle(lat, lon, sat_lon):
    """Earth-centre angle between a ground point and the sub-satellite point."""
    la, dlon = np.deg2rad(lat), np.deg2rad(np.asarray(lon) - sat_lon)
    return np.arccos(np.clip(np.cos(la) * np.cos(dlon), -1.0, 1.0))


def view_geometry(lat, lon, sat_lon, ifov=ABI_IFOV):
    """Viewing geometry of a geostationary sensor at a ground point.

    Returns
    -------
    dict with
        vza_deg   : viewing zenith angle
        azi_deg   : view azimuth, degrees clockwise from north, pointing
                    from the ground point TOWARD the sub-satellite point.
                    This is the direction the footprint is stretched along.
        slant_km  : satellite-to-point range
        gsd_along : ground sample distance along azimuth [km]
        gsd_across: ground sample distance across azimuth [km]
    """
    gam = _central_angle(lat, lon, sat_lon)
    r, R = SAT_R / 1e3, R_EQ / 1e3
    elev = np.arctan2(np.cos(gam) - R / r, np.sin(gam))
    vza = 0.5 * np.pi - elev
    slant = np.sqrt(R**2 + r**2 - 2 * R * r * np.cos(gam))

    la, dlon = np.deg2rad(lat), np.deg2rad(np.asarray(lon) - sat_lon)
    azi = np.degrees(np.arctan2(np.sin(-dlon), -np.sin(la) * np.cos(dlon)))

    if np.degrees(vza) > 85.0:
        raise ValueError("point is at VZA > 85 deg for this satellite: "
                         "the along-view footprint is unbounded")
    across = ifov * slant
    return dict(vza_deg=float(np.degrees(vza)), azi_deg=float(azi % 360.0),
                slant_km=float(slant), gsd_across=float(across),
                gsd_along=float(across / np.cos(vza)))


# ---------------------------------------------------------------- covariances

def _rot_cov(sig_along, sig_across, azi_deg):
    """Covariance [km^2] in a local (north, east) frame."""
    a = np.deg2rad(azi_deg)
    u = np.array([np.cos(a), np.sin(a)])          # along-view unit vector
    v = np.array([-np.sin(a), np.cos(a)])
    return sig_along**2 * np.outer(u, u) + sig_across**2 * np.outer(v, v)


def footprint_cov(lat, lon, sat_lon, psf_factor=0.45, dz_km=0.0,
                  ifov=ABI_IFOV):
    """Effective PSF covariance [km^2] of one view at one point.

    psf_factor : sigma / GSD.  0.29 is the box-IFOV equivalent, ~0.45 is a
                 reasonable ABI MTF + fixed-grid resampling value.  Only the
                 pair's DIFFERENCE in anisotropy really matters, but this
                 sets how much absolute blur homogenisation costs.
    dz_km      : optional cloud-layer geometric thickness.  Adds a slant-path
                 smear dz*tan(VZA) (uniform-equivalent sigma = smear/sqrt12)
                 along the same azimuth.
    """
    g = view_geometry(lat, lon, sat_lon, ifov)
    s_al = psf_factor * g["gsd_along"]
    s_ac = psf_factor * g["gsd_across"]
    if dz_km > 0:
        smear = dz_km * np.tan(np.deg2rad(g["vza_deg"]))
        s_al = np.hypot(s_al, smear / np.sqrt(12.0))
    return _rot_cov(s_al, s_ac, g["azi_deg"]), g


def common_target(covs, inflate=1.05):
    """Smallest isotropic PSF both views can be blurred UP to.

    sigma_t = inflate * max over views of the largest footprint axis.
    """
    smax = max(float(np.sqrt(np.linalg.eigvalsh(C).max())) for C in covs)
    s = inflate * smax
    return s**2 * np.eye(2), s


# ------------------------------------------------------------------- kernels

def gauss_kernel(cov, px, nsig=3.0, min_half=1):
    """Sample N(0, cov) on a (lat, lon) pixel grid.

    px : (px_lat_km, px_lon_km) as returned by geometry.build_grid.
    Returns a normalised 2-D kernel; a 1x1 delta if cov is negligible.
    """
    cov = np.asarray(cov, float)
    w, _ = np.linalg.eigh(cov)
    if w.max() <= 1e-6:
        return np.ones((1, 1))
    if w.min() < -1e-8:
        raise ValueError("matching kernel covariance is not positive "
                         "semi-definite -- raise the target inflate factor")
    cov = cov + 1e-6 * np.eye(2)

    hi = int(max(min_half, np.ceil(nsig * np.sqrt(cov[0, 0]) / px[0])))
    hj = int(max(min_half, np.ceil(nsig * np.sqrt(cov[1, 1]) / px[1])))
    di = np.arange(-hi, hi + 1) * px[0]                 # north [km]
    dj = np.arange(-hj, hj + 1) * px[1]                 # east  [km]
    N, E = np.meshgrid(di, dj, indexing="ij")

    P = np.linalg.inv(cov)
    q = P[0, 0] * N**2 + 2 * P[0, 1] * N * E + P[1, 1] * E**2
    K = np.exp(-0.5 * q)
    return K / K.sum()


def matching_kernels(lat, lon, sat_lons, px, psf_factor=0.45, dz_km=0.0,
                     inflate=1.05, sigma_t_km=None):
    """Per-view kernels that map both footprints onto one isotropic PSF.

    Parameters
    ----------
    sat_lons : {sat_id: sub-satellite longitude}
    px       : (px_lat_km, px_lon_km)
    sigma_t_km : override the automatic common target (must exceed both
                 along-view sigmas, else a ValueError is raised).

    Returns
    -------
    kernels : {sat_id: 2-D kernel}
    info    : {sat_id: view_geometry dict} plus "sigma_t_km"
    """
    covs, info = {}, {}
    for sat, slon in sat_lons.items():
        covs[sat], info[sat] = footprint_cov(lat, lon, slon, psf_factor,
                                             dz_km)
    if sigma_t_km is None:
        CT, s_t = common_target(list(covs.values()), inflate)
    else:
        s_t = float(sigma_t_km)
        CT = s_t**2 * np.eye(2)
    info["sigma_t_km"] = s_t
    return ({sat: gauss_kernel(CT - covs[sat], px) for sat in covs}, info)


# -------------------------------------------------------------- convolutions

def nan_convolve(f, K):
    """NaN-aware convolution: normalise by the convolved validity mask."""
    if K.shape == (1, 1):
        return np.asarray(f, float)
    f = np.asarray(f, float)
    m = np.isfinite(f)
    num = convolve(np.where(m, f, 0.0), K, mode="nearest")
    den = convolve(m.astype(float), K, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan                      # never invent data over gaps
    return out


def make_preps(lat, lon, sat_lons, px, hp_sig, psf_factor=0.45, dz_km=0.0,
               inflate=1.05, sigma_t_km=None):
    """Per-satellite prep callables: homogenise PSF, then band-pass.

    Drop-in replacement for the single ``hp_km(f, sig)`` currently applied
    to both views inside ``project_pair``.

    Returns
    -------
    preps : {sat_id: f(array) -> array}
    info  : geometry/diagnostics dict from matching_kernels
    """
    from .fields import hp_km

    kernels, info = matching_kernels(lat, lon, sat_lons, px, psf_factor,
                                     dz_km, inflate, sigma_t_km)

    def _mk(K):
        def prep(f):
            return hp_km(nan_convolve(f, K), hp_sig)
        return prep

    return {sat: _mk(K) for sat, K in kernels.items()}, info


# ------------------------------------------------- minimum-MTF (Fourier) form

def _freq_grid(shape, px):
    ky = np.fft.fftfreq(shape[0], d=px[0]) * 2 * np.pi      # rad / km, north
    kx = np.fft.fftfreq(shape[1], d=px[1]) * 2 * np.pi      # rad / km, east
    return np.meshgrid(ky, kx, indexing="ij")


def min_mtf_filters(shape, px, covs):
    """Fourier filters taking every view onto the pointwise-min MTF.

    Isotropic homogenisation is wasteful: it degrades BOTH views to the
    worst axis of the worst view in EVERY direction.  The minimal common
    PSF is the one whose MTF is min_i MTF_i(k) pointwise -- anisotropic,
    and still sharp in directions where both views happen to be sharp.

    With Gaussian footprints MTF_i(k) = exp(-0.5 k^T Sigma_i k), so

        F_i(k) = min_j MTF_j(k) / MTF_i(k) = exp(-0.5 max_j(q_j - q_i))

    which is <= 1 everywhere: a pure blur for every view, never a
    deconvolution.  Returns {sat: filter array} on the FFT grid.
    """
    KY, KX = _freq_grid(shape, px)
    q = {}
    for sat, C in covs.items():
        q[sat] = (C[0, 0] * KY**2 + 2 * C[0, 1] * KY * KX + C[1, 1] * KX**2)
    qmax = np.maximum.reduce(list(q.values()))
    return {sat: np.exp(-0.5 * (qmax - q[sat])) for sat in q}


def fft_filter(f, F):
    """Apply a Fourier filter to a field with NaNs (mask-normalised)."""
    f = np.asarray(f, float)
    m = np.isfinite(f)
    num = np.fft.ifft2(np.fft.fft2(np.where(m, f, 0.0)) * F).real
    den = np.fft.ifft2(np.fft.fft2(m.astype(float)) * F).real
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.where(np.abs(den) > 1e-3, den, np.nan)
    out[~m] = np.nan
    return out


def make_preps_minmtf(lat, lon, sat_lons, shape, px, hp_sig,
                      psf_factor=0.45, dz_km=0.0):
    """make_preps, but homogenising to the pointwise-min MTF."""
    from .fields import hp_km

    covs, info = {}, {}
    for sat, slon in sat_lons.items():
        covs[sat], info[sat] = footprint_cov(lat, lon, slon, psf_factor,
                                             dz_km)
    F = min_mtf_filters(shape, px, covs)

    def _mk(Fi):
        def prep(f):
            return hp_km(fft_filter(f, Fi), hp_sig)
        return prep

    return {sat: _mk(F[sat]) for sat in F}, info
