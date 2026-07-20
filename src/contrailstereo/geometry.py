"""Viewing geometry: parallax projection, zenith angles, retrieval grids"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from .config import R_EQ, R_POL, SAT_R, SAT_LON, StereoConfig, DEFAULT


_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_to_geod = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def apparent_surface_latlon(lat, lon, h_m, sat_lon):
    """Apparent ground position of an elevated point in geostationary view.

    A ray is cast from the satellite through the point at height h_m and
    intersected with the WGS84 ellipsoid; the satellite-facing root is the
    apparent (parallax-displaced) surface position.

    Parameters
    ----------
    lat, lon : array_like [deg]
        Geodetic position(s) of the elevated point.
    h_m : float
        Height above the ellipsoid [METRES].
    sat_lon : float [deg]
        Sub-satellite longitude.

    Returns
    -------
    glat, glon : ndarray [deg]
        Apparent surface position, same shape as inputs; NaN where the
        line of sight misses the ellipsoid.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    X, Y, Z = _to_ecef.transform(lon, lat, np.full_like(lat, float(h_m)))

    lam = np.deg2rad(sat_lon)
    Sx, Sy, Sz = SAT_R * np.cos(lam), SAT_R * np.sin(lam), 0.0
    vx, vy, vz = X - Sx, Y - Sy, Z - Sz

    A = (vx**2 + vy**2) / R_EQ**2 + vz**2 / R_POL**2
    B = 2.0 * ((Sx * vx + Sy * vy) / R_EQ**2 + Sz * vz / R_POL**2)
    Cq = (Sx**2 + Sy**2) / R_EQ**2 + Sz**2 / R_POL**2 - 1.0

    disc = B**2 - 4.0 * A * Cq
    disc = np.where(disc < 0, np.nan, disc)
    t = (-B - np.sqrt(disc)) / (2.0 * A)
    glon, glat, _ = _to_geod.transform(Sx + t * vx, Sy + t * vy, Sz + t * vz)
    return glat, glon


def vza_deg(lat_pt, lon_pt, sat_lon):
    """Viewing zenith angle [deg] at a ground point."""
    X, Y, Z = _to_ecef.transform(lon_pt, lat_pt, 0.0)
    lam = np.deg2rad(sat_lon)
    v = np.array([SAT_R * np.cos(lam) - X, SAT_R * np.sin(lam) - Y, -Z])
    up = np.array([X, Y, Z]) / np.linalg.norm([X, Y, Z])
    return float(np.degrees(np.arccos(np.dot(v / np.linalg.norm(v), up))))


def parallax_displacement_km(lat, lon, h_km, sat_lon):
    """Magnitude [km] of the parallax displacement at height h_km.
    Diagnostic helper (ground-point audits, error budgets, tests)."""
    glat, glon = apparent_surface_latlon(lat, lon, h_km * 1000.0, sat_lon)
    dlat = (np.asarray(glat) - lat) * 111.0
    dlon = (np.asarray(glon) - lon) * 111.0 * np.cos(np.deg2rad(lat))
    return np.hypot(dlat, dlon)


def build_grid(pr, cfg: StereoConfig = DEFAULT):
    """Track-extent retrieval grid at ~cfg.px_km isotropic resolution.

    Parameters
    ----------
    pr : DataFrame with .lat/.lon of the scene's collocated profiles.

    Returns
    -------
    glat, glon : 2-D ndarrays [deg]
    px : (px_lat_km, px_lon_km) actual pixel sizes -- feed to km_filters.
    """
    la0 = pr.lat.min() - cfg.pad_lat_deg
    la1 = pr.lat.max() + cfg.pad_lat_deg
    lo0 = pr.lon.min() - cfg.pad_lon_deg
    lo1 = pr.lon.max() + cfg.pad_lon_deg
    latm = 0.5 * (la0 + la1)
    n_la = int(np.clip((la1 - la0) * 111.0 / cfg.px_km,
                       cfg.min_npx, cfg.max_npx))
    n_lo = int(np.clip((lo1 - lo0) * 111.0
                       * np.cos(np.deg2rad(latm)) / cfg.px_km,
                       cfg.min_npx, cfg.max_npx))
    glat, glon = np.meshgrid(np.linspace(la0, la1, n_la),
                             np.linspace(lo0, lo1, n_lo), indexing="ij")
    px = ((la1 - la0) * 111.0 / n_la,
          (lo1 - lo0) * 111.0 * np.cos(np.deg2rad(latm)) / n_lo)
    return glat, glon, px


def km_filters(px, cfg: StereoConfig = DEFAULT):
    """Per-axis filter sizes (pixels) from physical scales (km)."""
    sig = (cfg.hp_sigma_km / px[0], cfg.hp_sigma_km / px[1])
    win = (max(int(cfg.win_km / px[0]) | 1, 3),
           max(int(cfg.win_km / px[1]) | 1, 3))
    return sig, win