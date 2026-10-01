"""Data structures for the contrailstereo repo.

The following conventions apply:

Times:      [pd.Timestamp] in UTC.
Heights:    km above the WGS84 ellipsoid unless a name says ``_m``.
Maps:       2D arrays on ``Grid`` indexed (lat, lon), NaN = no retrieval.
Truth:      DataFrame with at least ``lat``, ``lon``, ``top_km``; its index
            is the profile id (``pid``) carried through validation
            
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import GRID_KINDS, MODES

if TYPE_CHECKING:
    from .data.abi_time import PixelClock
    from .geometry import NativeGrid


# ---------- vocab ----------

QC_OK = "ok"
QC_NO_DATA = "no_data"
QC_COVERAGE = "coverage"
QC_OFFSET = "offset"
QC_ERROR = "error"

QC_VERDICTS = (QC_OK, QC_NO_DATA, QC_COVERAGE, QC_OFFSET, QC_ERROR)
QC_DATA_INVALID = (QC_NO_DATA, QC_COVERAGE)

TRUTH_COLUMNS = ("lat", "lon", "top_km")


def _utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts

# ---------- geometry ----------

class BBox(NamedTuple):
    """Latitude/longitude box [deg]. Longitudes in [-180, 180]."""
    lat0: float
    lat1: float
    lon0: float
    lon1: float

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.lat0 + self.lat1), 0.5 * (self.lon0 + self.lon1)
    

@dataclass(frozen=True, eq=False)
class Grid:
    """Retrieval grid: regular in lat/lon and isotropic in km.
    
    lat, lon : 2D (n_lat, n_lon) from meshgrid(indexing="ij").
    px_km    : actual (lat, lon) pixel size [km]."""

    lat: np.ndarray
    lon: np.ndarray
    px_km: tuple[float, float]
    bbox: BBox
    kind: str = "latlon"

    def __post_init__(self):
        object.__setattr__(self, "bbox", BBox(*self.bbox))
        if self.lat.shape != self.lon.shape or self.lat.ndim != 2:
            raise ValueError("lat and lon must be 2-D arrays of equal shape")
        if self.kind not in GRID_KINDS:
            raise ValueError(f"unknown grid kind {self.kind!r}")
 
    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape
 
    @property
    def lat1d(self) -> np.ndarray:
        return self.lat[:, 0]
 
    @property
    def lon1d(self) -> np.ndarray:
        return self.lon[0, :]
    
    def _require_latlon(self):
        if self.kind != "latlon":
            raise ValueError(f"1-D axes are undefined on a {self.kind!r} grid")
    
# ---------- inputs ----------

@dataclass(frozen=True, eq=False)
class Case:
    """One retrieval location and time as well as optional ground-truth.
    
    id      : unique string (for CALIOP cases, the GOES filename).
    time    : observation time (UTC).
    bbox    : retrieval domain.
    truth   : optional validation points
    meta    : metadata
    anchor  : lat, lon representative point to ancillary data"""

    id: str
    time: pd.Timestamp
    bbox: BBox
    truth: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)
    anchor: Optional[tuple[float, float]] = None

    def __post_init__(self):
        object.__setattr__(self, "time", _utc(self.time))
        object.__setattr__(self, "bbox", BBox(*self.bbox))
        object.__setattr__(self, "anchor", tuple(map(float, self.anchor))
                           if self.anchor is not None else self.bbox.center)
        if self.truth is not None:
            missing = set(TRUTH_COLUMNS) - set(self.truth.columns)
            if missing:
                raise ValueError(f"case {self.id}: truth lacks {sorted(missing)}")
            

@dataclass(frozen=True, eq=False)
class WindProfile:
    """Wind as a function of height at one location and time.
 
    h, u, v : 1-D arrays, h [km] ascending, u/v [m/s] (east/north).
    source  : provenance, e.g. "era5"."""

    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    source: str = ""
 
    def __post_init__(self):
        h, u, v = (np.asarray(x, float) for x in (self.h, self.u, self.v))
        if not (h.shape == u.shape == v.shape) or h.ndim != 1:
            raise ValueError("h, u, v must be 1-D arrays of equal length")
        o = np.argsort(h)
        for name, x in (("h", h), ("u", u), ("v", v)):
            object.__setattr__(self, name, x[o])
 
    def at(self, h_km: float) -> tuple[float, float]:
        """(u, v) [m/s] at h_km, linearly interpolated, clamped at the ends."""
        return (float(np.interp(h_km, self.h, self.u)),
                float(np.interp(h_km, self.h, self.v)))
    

View = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""(lat, lon) -> matching field (e.g. BTD [K]); NaN outside the sector."""


@dataclass(frozen=True, eq=False)
class NativeRaster:
    """A satellite's conditioned-input BTD on its own fixed grid, cropped to a case.

    grid   : the crop's NativeGrid (apparent positions of pixel centres).
    btd    : (ny, nx) float32 BTD [K]; NaN where masked (destriped rows).
    t_s    : (ny, nx) scan time of every pixel [s after Frame.time] under
             the frame's time model (constant for "nominal")."""

    grid: "NativeGrid"
    btd: np.ndarray
    t_s: np.ndarray




@dataclass(frozen=True, eq=False)
class Frame:
    """One satellite's image at one time.
 
    view   : samples the matching field at arbitrary (lat, lon).
    domain : "C" (CONUS) or "F" (full disk).
    stripe : destriping diagnostics, e.g. {"max_score": .., "n_rows": ..}
    clock  : per-pixel scan times (data.abi_time.PixelClock or None)
    s0     : seconds after clock.start at which time falls.
    raster : the native pixel input, grid kind native only"""

    sat: int
    sat_lon: float
    time: pd.Timestamp
    view: View
    domain: str = "?"
    stripe: dict = field(default_factory=dict)
    clock: Optional["PixelClock"] = None
    s0: float = 0.0
    raster: Optional[NativeRaster] = None
 
    def __post_init__(self):
        object.__setattr__(self, "time", _utc(self.time))

    def dt_at(self, lat, lon):
        """Seconds betwen this frame's time and the scane time of the pixel that 
        images the apparent position. NaN off the sector, 0 for a frame with no clock."""

        if self.clock is None:
            return np.zeros(np.shape(lat))
        return self.clock.since_start(lat, lon) - self.s0

# ---------- output ----------

@dataclass(frozen=True, eq=False)
class Result:
    """What retrieve() returns for a case.
    
    height  : height map [km]; NaN where masked.
    r       : peak local correlation at each pixel.
    amp     : local feature amplitude of the reference view [K].
    s_eff   :
    ref_time: time the maps are valid at (the reference frame's time)
    wind    : WindProfile used to advect
    native. : NativeMap on the reference view's own pixels
    diag    : scalar diagnostics."""

    case_id: str
    mode: str
    qc: str
    grid: Grid
    height: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    amp: Optional[np.ndarray] = None
    s_eff: Optional[np.ndarray] = None
    diag: dict = field(default_factory=dict)
    config_hash: str = ""
    ref_time: Optional[pd.Timestamp] = None
    wind: Optional[WindProfile] = None
    native : Optional[NativeMap] = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.qc not in QC_VERDICTS:
            raise ValueError(f"unknown qc verdict {self.qc!r}")
        for name in ("height", "r", "amp"):
            a = getattr(self, name)
            if a is not None and a.shape != self.grid.shape:
                raise ValueError(f"{name} shape {a.shape} != grid {self.grid.shape}")
            

@dataclass(frozen=True, eq=False)
class NativeMap:
    """Retrieval on the reference view's own pixels (before placement on the
    output grid). All arrays (ny, nx) on ``grid``.

    height, r, amp : as Result, per native pixel.
    t_s            : scan time of each pixel [s after Result.ref_time].
    win_px         : correlation window (rows, cols) in native pixels."""

    grid: "NativeGrid"
    height: np.ndarray
    r: np.ndarray
    amp: np.ndarray
    t_s: np.ndarray
    win_px: tuple