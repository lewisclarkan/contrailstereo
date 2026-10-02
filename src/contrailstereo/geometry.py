"""Viewing geometry, domains, and retrieval grids"""

from __future__ import annotations

import itertools

import numpy as np
from pyproj import Proj, Transformer

from .config import R_EQ, R_POL, SAT_LON, SAT_R, StereoConfig, DEFAULT
from .types import BBox, Grid

KM_PER_DEG = 111.0
ABI_STEP_RAD = 56e-6 # ABI fixed-grid step, 2km at nadir

_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
_to_geod = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

def _sat_ecef(sat_lon):
    lam = np.deg2rad(sat_lon)
    return np.array([SAT_R * np.cos(lam), SAT_R * np.sin(lam), 0.0])

# ---------- Advection ----------

def advect_latlon(lat, lon, u, v, dt_s):
    """Advect a feature at (lat, lon) to a new position dt_s later
    with wind of (u, v) [m/s]"""

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    m = KM_PER_DEG * 1e3
    dt = np.asarray(dt_s, float)
    return (lat + np.asarray(v, float) * dt / m,
            lon + np.asarray(u, float) * dt / (m * np.cos(np.deg2rad(lat))))

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


def true_latlon_from_apparent(alat, alon, h_m, sat_lon, n_iter=6):
    """Inverse of apparent_surface_latlon."""

    alat = np.asarray(alat, float)
    alon = np.asarray(alon, float)

    h_m = np.broadcast_to(np.asarray(h_m, float), alat.shape)
    with np.errstate(invalid="ignore"):
        P = np.stack(_to_ecef.transform(alon, alat, np.zeros_like(alat)))
        S = _sat_ecef(sat_lon).reshape((3,) + (1,) * alat.ndim)
        d = P - S
        la, lo = np.deg2rad(alat), np.deg2rad(alon)
        up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                       np.sin(la)])
        t = 1.0 + h_m / (d * up).sum(0)    
        for _ in range(n_iter):
            X = S + t * d
            lon_t, lat_t, h_t = _to_geod.transform(X[0], X[1], X[2])
            lt, ln = np.deg2rad(lat_t), np.deg2rad(lon_t)
            nrm = np.stack([np.cos(lt) * np.cos(ln), np.cos(lt) * np.sin(ln),
                            np.sin(lt)])
            t = t - (h_t - h_m) / (d * nrm).sum(0)
        X = S + t * d
        lon_t, lat_t, h_t = _to_geod.transform(X[0], X[1], X[2])
    return lat_t, lon_t, h_t - h_m


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
    """Retrieval (output) grid over bbox"""

    bbox = BBox(*bbox)

    if cfg.grid_kind in ("latlon", "native"):
        return _latlon_grid(bbox, cfg)
    raise NotImplementedError("native satellite grid not implemented yet")


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

# ---------- Native (fixed-grid) pixels ----------

def _footprint_xy(p, H, sat, bbox, h_range_km):
    """Scan angles [rad] of the bbox seen from 'sat' at every height
    in h_range_km. p is the geos Proj and H it its perspective height [m]"""

    la0, la1, lo0, lo1 = bbox
    LA, L0 = np.meshgrid(np.linspace(la0, la1, 9), np.linspace(lo0, lo1, 9), indexing="ij")
    xs, ys = [], []
    for h in h_range_km:
        alat, alon = apparent_surface_latlon(LA, L0, h*1e3, SAT_LON[sat])
        x, y = p(alon, alat, errcheck=False)
        xs.append(x / H)
        ys.append(y / H)
    return np.concatenate(xs), np.concatenate(ys)


class OutsideSector(ValueError):
    """The case bbox does not overlap the dataset's sector (e.g. off CONUS)."""


class NativeGrid:
    """One satellite's ABI fixed grid, cropped to a case.
    
    alat and alon are the apparent ground positions of the pixel centres
    true_at(h) turns them into the true position of a cloud at height h
    rc goes from an apparent position back to a fractional (row, col)"""

    def __init__(self, x_rad, y_rad, sat_lon, H, lon0, sweep="x"):
        self.sat_lon = float(sat_lon)
        self.H = float(H)
        self.lon0 = float(lon0)
        self.x_rad = np.asarray(x_rad, float)
        self.y_rad = np.asarray(y_rad, float)

        if self.x_rad.size < 3 or self.y_rad.size < 3:
            raise OutsideSector("bbox is outside this dataset's sector")

        
        self.dx = float(self.x_rad[1] - self.x_rad[0])
        self.dy = float(self.y_rad[1] - self.y_rad[0])
        self.p = Proj(proj="geos", h=self.H, lon_0=self.lon0, sweep=sweep,
                      a=R_EQ, b=R_POL)
        
        X, Y = np.meshgrid(self.x_rad * self.H, self.y_rad * self.H)
        lon, lat = self.p(X, Y, inverse=True, errcheck=False)

        bad = ~(np.isfinite(lon) & np.isfinite(lat)
                & (np.abs(lon) < 1e10) & (np.abs(lat) < 1e10))
        
        self.alat = np.where(bad, np.nan, lat)
        self.alon = np.where(bad, np.nan, lon)
        self.shape = X.shape

    @classmethod
    def from_dataset(cls, ds, sat, bbox, h_range_km, pad_km=10.0):
        """Crop an ABI to a bbox at every height in h_range_ with pad pad_km.
        
        Returns (NativeGrid, (row_slice, col_slice)) so the caller can crops its
        arrays in the same way."""

        pj = ds["goes_imager_projection"]
        H = float(pj.attrs["perspective_point_height"])
        lon0 = float(pj.attrs["longitude_of_projection_origin"])
        sweep = pj.attrs.get("sweep_angle_axis", "x")

        p = Proj(proj="geos", h=H, lon_0=lon0, sweep=sweep, a=R_EQ, b=R_POL)
        xs, ys = _footprint_xy(p, H, sat, bbox, h_range_km)
        pad = pad_km / 37000.0

        x, y = ds["x"].values, ds["y"].values
        jj = np.where((x >= xs.min() - pad) & (x <= xs.max() + pad))[0]
        ii = np.where((y >= ys.min() - pad) & (y <= ys.max() + pad))[0]

        if jj.size == 0 or ii.size == 0:
            raise ValueError("bbox is outside this dataset's sector")
        
        isl = slice(int(ii.min()), int(ii.max()) + 1)
        jsl = slice(int(jj.min()), int(jj.max()) + 1)

        return cls(x[jsl], y[isl], SAT_LON[sat], H, lon0, sweep), (isl, jsl)
    

    def true_at(self, h_km):
        """True (lat, lon) of every pixel for a cloud at h_km."""
        tlat, tlon, _ = true_latlon_from_apparent(self.alat, self.alon,
                                                  h_km * 1e3, self.sat_lon)
        return tlat, tlon
    

    def rc(self, lat, lon):
        """Fractional (row, col) of apparent ground positions; NaN off the
        projection."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        x, y = self.p(lon.ravel(), lat.ravel(), errcheck=False)
        x, y = np.asarray(x, float), np.asarray(y, float)
        bad = ~(np.isfinite(x) & np.isfinite(y)
                & (np.abs(x) < 1e20) & (np.abs(y) < 1e20))
        col = np.where(bad, np.nan, (x / self.H - self.x_rad[0]) / self.dx)
        row = np.where(bad, np.nan, (y / self.H - self.y_rad[0]) / self.dy)
        return row.reshape(lat.shape), col.reshape(lat.shape)
    

    def jacobian_km(self, h_km=11.0):
        """d(north, east)/d(row, col) [km per pixel] at the grid centre for 
        a cloud at h_km, and the pixel spacing [km] along the rows and columns."""

        ic, jc = self.shape[0] // 2, self.shape[1] // 2
        sl = np.s_[ic-1:ic+2, jc-1:jc+2]
        tlat, tlon, _ = true_latlon_from_apparent(
            self.alat[sl], self.alon[sl], h_km * 1e3, self.sat_lon)
        c = np.cos(np.deg2rad(tlat[1,1]))

        N = (tlat - tlat[1,1]) * KM_PER_DEG
        Ee = (tlon - tlon[1,1]) * KM_PER_DEG * c

        J = np.array([[(N[2,1] - N[0,1]) / 2, (N[1, 2] - N[1,0]) / 2],
                        [(Ee[2,1] - Ee[0,1]) / 2, (Ee[1,2] - Ee[1,0]) / 2]])
        
        return J, (float(np.hypot(J[0,0], J[1,0])), float(np.hypot(J[0,1], J[1,1])))
    

    def filters(self, cfg: StereoConfig = DEFAULT, h_km=11.0):
        """High pass sigma and correlation window in native pixels from the 
        km scales of cfg (cf.km_filters) plus the Jacobian"""

        J, (si, sj) = self.jacobian_km(h_km)
        sig = (cfg.hp_sigma_km / si, cfg.hp_sigma_km / sj)
        det = abs(np.linalg.det(J))
        g = 1.0 / np.sqrt(det / (si * sj))

        def odd_pair(t):
            lo = max(int(np.floor((t-1) / 2) * 2 + 1), 3)
            return lo, lo + 2
        
        win = min(itertools.product(odd_pair(g * cfg.win_km / si),
                                    odd_pair(g * cfg.win_km / sj)),
                key=lambda w: (abs(w[0] * w[1] * det / cfg.win_km ** 2 - 1),
                                abs(np.log(w[0] * si / (w[1] * sj)))))
        return sig, win, J
