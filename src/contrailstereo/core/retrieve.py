"""The retrieval: one scene in, scene record + height map + per-profile
samples out.

load_scene()  -- acquisition (network/cache): imagery, winds, profiles.
run_scene()   -- pure computation on a SceneData; offline-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..config import StereoConfig, DEFAULT, SAT_LON
from ..geometry import build_grid, km_filters, vza_deg
from ..data.goes import fetch_pair, row_artifact_scores, flag_rows, masked_copy
from ..data.era5 import fetch_era5_winds
from .fields import make_sampler
from .match import project_pair, scene_correlation, local_corr
from .peaks import cirrus_peak, refine_peak
from .qc import qc_verdict


# one place the CSV schema lives; every writer/reader imports it
SCENE_SCHEMA = [
    "scene", "qc", "offset_s", "dom_east", "dom_west", "vza_east",
    "vza_west", "g17_stripe", "n_rows_masked", "coverage_ok", "valid_frac",
    "h_coarse", "h_local", "r_local", "sigma_h", "n_low_peaks", "n_peaks",
    "wind_applied", "u_pk", "v_pk", "wind_disp_km", "wind_resid_km",
    "map_n", "map_bias", "map_scatter", "map_coverage", "feat_amp",
    "config_hash",
]


@dataclass
class SceneData:
    """Everything run_scene needs, acquisition-free."""
    when: object                      # datetime
    lat0: float
    lon0: float
    fields: dict                      # {sat: {14: ds, 15: ds}}
    domains: dict                     # {sat: "C"|"F"}
    offset_s: float
    profiles: pd.DataFrame            # scene's collocated profiles
    winds: Optional[tuple] = None     # (u_of_h, v_of_h) or None
    meta: dict = field(default_factory=dict)

def _patch_median(Z, i, j):
    """Median of the 5x5 patch at (i, j); NaN when the patch is fully
    masked (an unmapped profile is NaN by design, not by warning)."""
    patch = Z[i-2:i+3, j-2:j+3]
    return float(np.nanmedian(patch)) if np.isfinite(patch).any() else np.nan


def load_scene(row, prof_df, cfg: StereoConfig = DEFAULT) -> SceneData:
    """Acquire a scene's inputs (cache-first)."""
    when = row.time.floor("s").to_pydatetime()
    pr = prof_df[prof_df.goes_file == row.goes_file].sort_values("lat").copy()
    fields, domains, times = {}, {}, {}
    for sat in (cfg.sat_east, cfg.sat_west):
        fields[sat], domains[sat] = fetch_pair(when, sat, cfg)
        times[sat] = np.datetime64(fields[sat][14].t.values)
    offset = float((times[cfg.sat_west] - times[cfg.sat_east])
                   / np.timedelta64(1, "s"))
    winds = None
    if abs(offset) >= cfg.wind_min_offset_s and abs(offset) <= cfg.no_data_s:
        winds = fetch_era5_winds(when, float(row.lat), float(row.lon), cfg)
    return SceneData(when=when, lat0=float(row.lat), lon0=float(row.lon),
                     fields=fields, domains=domains, offset_s=offset,
                     profiles=pr, winds=winds,
                     meta=dict(goes_file=row.goes_file))


def run_scene(sd: SceneData, cfg: StereoConfig = DEFAULT, do_map=True):
    """Compute the retrieval.

    Returns
    -------
    rec : dict (SCENE_SCHEMA keys)
    pr : DataFrame with h_map / r_map columns (None if no map)
    grids : (glat, glon, Hq, rmax) or None -- for plotting
    """
    E, W = cfg.sat_east, cfg.sat_west
    rec = {k: np.nan for k in SCENE_SCHEMA}
    rec.update(qc="", offset_s=sd.offset_s,
               dom_east=sd.domains[E], dom_west=sd.domains[W],
               vza_east=vza_deg(sd.lat0, sd.lon0, SAT_LON[E]),
               vza_west=vza_deg(sd.lat0, sd.lon0, SAT_LON[W]),
               n_rows_masked=0, wind_applied=sd.winds is not None,
               u_pk=0.0, v_pk=0.0, wind_disp_km=0.0, wind_resid_km=0.0,
               map_n=0, map_coverage=0.0, config_hash=cfg.config_hash())

    if abs(sd.offset_s) > cfg.no_data_s:
        rec["qc"] = "no_data"
        return rec, None, None

    # striping: masked west view is primary when cfg says so
    score = row_artifact_scores(sd.fields[W][14], sd.fields[W][15])
    bad = flag_rows(score, cfg)
    rec.update(g17_stripe=float(np.nanmax(score)), n_rows_masked=int(bad.size))
    west14, west15 = sd.fields[W][14], sd.fields[W][15]
    if cfg.stripe_mask_primary and bad.size:
        west14, west15 = masked_copy(west14, bad), masked_copy(west15, bad)
    samplers = {E: make_sampler(sd.fields[E][14], sd.fields[E][15]),
                W: make_sampler(west14, west15)}

    glat, glon, px = build_grid(sd.profiles, cfg)
    sig, win = km_filters(px, cfg)

    # scene-mode window: v3.2-compatible scope (near the scene median)
    pr = sd.profiles
    near = ((np.abs(pr.lat - sd.lat0) < cfg.scene_window_deg)
            & (np.abs(pr.lon - sd.lon0) < 2 * cfg.scene_window_deg))
    dmin = np.full(glat.shape, np.inf)
    for la, lo in zip(pr.lat[near].values, pr.lon[near].values):
        dmin = np.minimum(dmin, np.hypot(
            glat - la, (glon - lo) * np.cos(np.deg2rad(la))))
    winmask = dmin < cfg.local_deg

    hs = np.arange(cfg.h_lo_km, cfg.h_hi_km + 1e-9, cfg.dh_km)
    adm = hs >= cfg.hmin_cirrus_km
    n_adm = int(adm.sum())
    cube = np.full((n_adm,) + glat.shape, np.nan) if do_map else None
    amp_mid = None
    curve = np.full(len(hs), np.nan)
    valid10 = nloc10 = 0
    ka = 0
    for k, hkm in enumerate(hs):
        A, B = project_pair(hkm, glat, glon, samplers, sig, cfg,
                            winds=sd.winds, offset_s=sd.offset_s)
        _, curve[k], nml, vf = scene_correlation(A, B, winmask,
                                                 cfg.min_local_px)
        if abs(hkm - 10.0) < 1e-6:
            valid10, nloc10 = vf, nml
        if do_map and adm[k]:
            cube[ka], a = local_corr(A, B, win)
            if ka == n_adm // 2:
                amp_mid = a
                rec["feat_amp"] = float(np.nanmean(a[winmask]))
            ka += 1

    coverage_ok = (valid10 >= cfg.min_valid_frac
                   and nloc10 >= cfg.min_local_px)
    rec.update(coverage_ok=coverage_ok, valid_frac=valid10)

    h_pk, r_pk, n_low, n_pk = cirrus_peak(hs, curve, cfg)
    h_f = r_f = sig_h = np.nan
    if np.isfinite(h_pk) and h_pk <= cfg.h_hi_km - cfg.edge_km:
        h_f, r_f, sig_h = refine_peak(hs, curve, h_pk)
    rec.update(h_coarse=h_pk, h_local=h_f, r_local=r_f, sigma_h=sig_h,
               n_low_peaks=n_low, n_peaks=n_pk)

    if sd.winds is not None and np.isfinite(h_f):
        u, v = sd.winds[0](h_f), sd.winds[1](h_f)
        spd = float(np.hypot(u, v))
        rec.update(u_pk=u, v_pk=v,
                   wind_disp_km=spd * abs(sd.offset_s) / 1e3,
                   wind_resid_km=(cfg.wind_sigma_base
                                  + cfg.wind_sigma_frac * spd)
                                 * abs(sd.offset_s) / 1e3)

    rec["qc"] = qc_verdict(h_pk, r_f if np.isfinite(h_f) else r_pk,
                           sd.offset_s, coverage_ok,
                           rec["wind_applied"], rec["wind_resid_km"], cfg)

    extras = dict(hs=hs, curve=curve, glat=glat, glon=glon,
                  Hq=None, rmax=None)

    prof_out = None
    if do_map:
        hs_adm = hs[adm]
        kmax = np.nanargmax(np.nan_to_num(cube, nan=-9.0), axis=0)
        rmax = np.take_along_axis(cube, kmax[None], 0)[0]
        km_ = np.clip(kmax, 1, n_adm - 2)
        r0, r1, r2 = (np.take_along_axis(cube, (km_ + d)[None], 0)[0]
                      for d in (-1, 0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            dh = 0.5 * (r0 - r2) / (r0 - 2 * r1 + r2)
        Hq = np.where((rmax > cfg.map_r_min) & (amp_mid > cfg.map_amp_min_k)
                      & (kmax > 0) & (kmax < n_adm - 1),
                      hs_adm[km_] + np.clip(dh, -1, 1) * cfg.dh_km, np.nan)
        prof_out = pr.copy()
        ii = np.searchsorted(glat[:, 0], pr.lat.values).clip(2, glat.shape[0]-3)
        jj = np.searchsorted(glon[0, :], pr.lon.values).clip(2, glat.shape[1]-3)
        prof_out["h_map"] = [_patch_median(Hq, i, j) for i, j in zip(ii, jj)]
        prof_out["r_map"] = [_patch_median(rmax, i, j) for i, j in zip(ii, jj)]
        d = (prof_out.h_map - prof_out.top_km).dropna()
        rec.update(map_n=int(len(d)),
                   map_coverage=float(np.isfinite(Hq).mean()),
                   map_bias=float(d.mean()) if len(d) else np.nan,
                   map_scatter=float(d.std()) if len(d) else np.nan)
        extras.update(Hq=Hq, rmax=rmax)

    return rec, prof_out, extras