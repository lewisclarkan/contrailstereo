"""Truth-free QC verdicts"""

from __future__ import annotations

import numpy as np

from ..config import StereoConfig, DEFAULT


VERDICTS = ("no_data", "coverage", "no_peak", "low_edge", "edge",
            "neg_r", "offset", "low_r", "ok")


def qc_verdict(h_pk, r_pk, offset_s, coverage_ok, wind_applied,
               wind_resid_km, cfg: StereoConfig = DEFAULT):
    if not coverage_ok:
        return "coverage"
    if not np.isfinite(h_pk):
        return "no_peak"
    if h_pk < cfg.hmin_cirrus_km + cfg.edge_km:
        return "low_edge"
    if h_pk > cfg.h_hi_km - cfg.edge_km:
        return "edge"
    if not np.isfinite(r_pk) or r_pk <= 0:
        return "neg_r"
    if wind_applied:
        if wind_resid_km > cfg.wind_resid_max_km:
            return "offset"
    elif abs(offset_s) >= 8.0:
        return "offset"
    if r_pk <= cfg.r_min:
        return "low_r"
    return "ok"