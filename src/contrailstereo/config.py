"""Configuration for contrailstereo"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

# --- Geometry constants ---
# WGS84 ellipsoid and GOES orbit radius [m]
R_EQ    = 6378137.0
R_POL   = 6356752.31414
SAT_R   = 42164160.0

# Satellite longitudes [deg]
SAT_LON = {
    16: -75.2,
    17: -137.2,
    18: -137.9,
    19: -75.2
}

@dataclass(frozen=True)
class StereoConfig:
    # Satellites
    sat_east: int = 16
    sat_west: int = 17
    match_channels: tuple = (13, 15) # BTD pair used for matching
                                     # (14,15) = testing candidate
                                     # (13,15) = default candidate

    # Grid 
    px_km: float = 1.0
    max_npx: int = 750
    min_npx: int = 60
    pad_lat_deg: float = 0.30
    pad_lon_deg: float = 0.70

    # Filter parameters
    hp_sigma_km: float = 8.0
    win_km: float = 25.0

    # Height scan 
    h_lo_km: float = 0.0
    h_hi_km: float = 16.0
    dh_km: float = 0.25
    hmin_cirrus_km: float = 8.0
    edge_km: float = 1.0

    # Scene-mode window
    local_deg: float = 0.35
    scene_window_deg: float = 1.0

    # Quality control thresholds
    r_min: float = 0.2
    min_local_px: int = 30
    min_valid_frac: float = 0.5
    prominence: float = 0.03
    no_data_s: float = 120.0
    vza_max_deg: float = 65.0 

    # Striping
    stripe_nsig: float = 4.0
    stripe_dilate: int = 1
    stripe_mask_primary: bool = True  # To be tested

    # ERA5 advection (v3.2, might be removed/replaced in tracker)
    wind_min_offset_s: float = 2.0
    wind_levels: tuple = ("100", "125", "150", "175", "200",
                          "225", "250", "300", "350", "400")
    wind_sigma_base: float = 2.5        # m/s error floor at cruise
    wind_sigma_frac: float = 0.08       # + fraction of local speed
    wind_resid_max_km: float = 0.25     # residual displacement -> "offset"

    # Map mode
    map_r_min: float = 0.35            # per-pixel quality mask floor
    map_amp_min_k: float = 0.25
    map_gates: tuple = (0.5, 0.6, 0.7) # r_map thresholds for quality/coverage tradeoff

    # Paths
    goes_cache: str = "data/goes_cache"
    era5_cache: str = "data/era5_cache"

    def config_hash(self) -> str:
        """Short stable hash of all settings, for stamping into outputs"""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]
    

DEFAULT = StereoConfig()