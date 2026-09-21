"""Viewing geometry, domains, and retrieval grids"""

from __future__ import annotations

import numpy as np
from pyproj import Transformer

from .config import R_EQ, R_POL, SAT_LON, SAT_R, StereoConfig, DEFAULT
from .types import BBox, Grid

KM_PER_DEG = 111.0

_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_to_geod = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

def _sat_ecef(sat_lon):
    lam = np.deg2rad(sat_lon)
    return np.array([SAT_R * np.cos(lam), SAT_R * np.sin(lam), 0.0])

# ---------- Parallax ----------

def apparent_surface_latlon(lat, lon, h_m, sat_lon):
    """Apparent ground position of an elevated point in geostationary view.

    A ray is cast from the satellite through the point at height h_m and
    intersected with the WGS84 ellipsoid; the satellite-facing root is the
    apparent (parallax-displaced) surface position.

    lat, lon: array_like [deg]
    h_m: float [m]
    sat_lon: float [deg]

    Returns (glat, glon) [deg], same shape as inputs. 
    """

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    X, Y, Z = _to_ecef.transform(lon, lat, np.full_like(lat, float(h_m)))
    Sx, Sy, Sz = _sat_ecef(sat_lon)
    vx, vy, vz = X - Sx, Y - Sy, Z - Sz
 
    A = (vx**2 + vy**2) / R_EQ**2 + vz**2 / R_POL**2
    B = 2.0 * ((Sx * vx + Sy * vy) / R_EQ**2 + Sz * vz / R_POL**2)
    Cq = (Sx**2 + Sy**2) / R_EQ**2 + Sz**2 / R_POL**2 - 1.0
 
    disc = B**2 - 4.0 * A * Cq
    disc = np.where(disc < 0, np.nan, disc)
    t = (-B - np.sqrt(disc)) / (2.0 * A)
    glon, glat, _ = _to_geod.transform(Sx + t * vx, Sy + t * vy, Sz + t * vz)

    return glat, glon


def parallax_vector_km(lat, lon, h_km, sat_lon):
    """(north, east) displacement [km] of the apparent position of a 
    point at h_km, local equirectangular approximation."""
    
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)

    glat, glon = apparent_surface_latlon(lat, lon, h_km * 1000.0, sat_lon)

    dn = (np.asarray(glat) - lat) * KM_PER_DEG
    de = (np.asarray(glon) - lon) * KM_PER_DEG * np.cos(np.deg2rad(lat))

    return dn, de


def parallax_displacement_km(lat, lon, h_km, sat_lon):
    """Magnitude of the parallax displacement at h_km."""
    return np.hypot(*parallax_vector_km(lat, lon, h_km, sat_lon))


def disparity_per_km(lat, lon, sat_a, sat_b, h_km=11.0):
    """Differential parallax between two satellites, (north, east) km per
    km of height. The direction in which it is aligned is the one in which 
    stereo contains height information."""
    na, ea = parallax_vector_km(lat, lon, h_km, SAT_LON[sat_a])
    nb, eb = parallax_vector_km(lat, lon, h_km, SAT_LON[sat_b])
    return (na - nb) / h_km, (ea - eb) / h_km

# ---------- Viewing geometry ----------

def view_geometry(lat, lon, sat_lon):
    """Viewing geometry of a geostationary satellite at ground point.
    
    Returns:
        vza_deg : viewing zenith angle
        azi_deg : azimuth of the line of sight, clockwise from north,
                  pointing from the ground toward the satellite
        slant_km: ground-to-satellite range"""
    
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    X, Y, Z = _to_ecef.transform(lon, lat, np.zeros_like(lat))
    S = _sat_ecef(sat_lon)
    v = np.stack([S[0] - X, S[1] - Y, S[2] - Z])
    rng = np.linalg.norm(v, axis=0)
    v = v / rng
 
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)])
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo),
                      np.cos(la)])
    cu = np.clip((v * up).sum(0), -1.0, 1.0)
    azi = np.degrees(np.arctan2((v * east).sum(0), (v * north).sum(0))) % 360.0
 
    out = dict(vza_deg=np.degrees(np.arccos(cu)), azi_deg=azi,
               slant_km=rng / 1e3)
    if lat.ndim == 0:
        out = {k: float(x) for k, x in out.items()}
    return out


def vza_deg(lat, lon, sat_lon):
    """Viewing zenith angle [deg] at a ground point."""
    return view_geometry(lat, lon, sat_lon)["vza_deg"]


# ---------- Domains and grids ----------

def bbox_around(lat, lon, pad_lat=0.30, pad_lon=0.70) -> BBox:
    """Box enclosing points plus padding [deg]"""

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return BBox(float(lat.min() - pad_lat), float(lat.max() + pad_lat),
                float(lon.min() - pad_lon), float(lon.max() + pad_lon))


def make_grid(bbox, cfg: StereoConfig = DEFAULT) -> Grid:
    """Retrieval grid over bbox"""

    bbox = BBox(*bbox)

    if cfg.grid_kind == "latlon":
        return _latlon_grid(bbox, cfg)
    if cfg.grid_kind == "native":
        raise NotImplementedError("native satellite grid not implemented yet")
    raise ValueError("unknown grid_king {cfg.grid_kind!r}")


def _latlon_grid(bbox : BBox, cfg: StereoConfig) -> Grid:
    """~cfg.px_km isotropic lat/lon grid, pixel counts clipped to
    [min_npx, max_npx] per axis."""

    la0, la1, lo0, lo1 = bbox
    latm = 0.5 * (la0 + la1)
    coslat = np.cos(np.deg2rad(latm))
    n_la = int(np.clip((la1 - la0) * KM_PER_DEG / cfg.px_km,
                       cfg.min_npx, cfg.max_npx))
    n_lo = int(np.clip((lo1 - lo0) * KM_PER_DEG * coslat / cfg.px_km,
                       cfg.min_npx, cfg.max_npx))
    glat, glon = np.meshgrid(np.linspace(la0, la1, n_la),
                             np.linspace(lo0, lo1, n_lo), indexing="ij")
    px = ((la1 - la0) * KM_PER_DEG / n_la,
          (lo1 - lo0) * KM_PER_DEG * coslat / n_lo)
    return Grid(glat, glon, px, bbox, kind="latlon")


def locate(grid: Grid, lat, lon):
    """Fractional (i, j) pixel indices of points on the grid; NaN outside.
    Canonical method to go from coordinate to pixels (to support native grid)."""

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if grid.kind != "latlon":
        raise NotImplementedError(f"locate on {grid.kind!r} grid")
    la, lo = grid.lat1d, grid.lon1d
    fi = (lat - la[0]) / (la[-1] - la[0]) * (la.size - 1)
    fj = (lon - lo[0]) / (lo[-1] - lo[0]) * (lo.size - 1)
    out = (fi < 0) | (fi > la.size - 1) | (fj < 0) | (fj > lo.size - 1)
    return np.where(out, np.nan, fi), np.where(out, np.nan, fj)


def km_filters(grid: Grid, cfg: StereoConfig = DEFAULT):
    """Per-axis filter sizes (pixels) from physical scales (km)."""
    px = grid.px_km
    sig = (cfg.hp_sigma_km / px[0], cfg.hp_sigma_km / px[1])
    win = (max(int(cfg.win_km / px[0]) | 1, 3),
           max(int(cfg.win_km / px[1]) | 1, 3))
    return sig, win