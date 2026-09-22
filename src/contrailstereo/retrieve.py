"""The retrieval: one Case in, one Result out.

    retrieve(case, cfg)
      ├─ make_grid(case.bbox)                      geometry
      ├─ load_frames(case)      -> {(sat, k): Frame}   data + prep.btd_view
      ├─ load_winds(case)       -> WindProfile | None  data.era5
      ├─ make_preps(grid)       -> per-view conditioning
      ├─ match_<mode>(...)      -> correlation cube over heights
      ├─ height_map(...)        -> refined, masked height map
      └─ scene_qc(...)          -> verdict + diagnostics

Nothing here knows about CALIOP. Validation samples the returned map at
truth points with ``sample_at``; the same function serves flight tracks.

Frames and projection are mode-agnostic: a frame is one satellite at one
time, and ``project`` advects the grid by wind x (frame time - reference
time) before parallax projection. Snapshot mode uses the East and West
frames at the case time; track mode will use the same machinery with frames
at several times (``match_track`` is the only missing piece).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .config import DEFAULT, SAT_LON, Paths, StereoConfig, load_paths
from .data import era5, goes
from .geometry import (apparent_surface_latlon, km_filters, locate, make_grid,
                       vza_deg)
from .prep import btd_view, make_preps, observability
from .types import (QC_COVERAGE, QC_NO_DATA, QC_OFFSET, QC_OK, Case, Frame,
                    Grid, Result, WindProfile)

M_PER_DEG = 111.0e3          # advection shift, as pre-restructure
QC_REF_H_KM = 10.0           # height at which valid overlap is judged
_FETCH = "fetch"


# ======================================================================
# Inputs
# ======================================================================
def frame_offsets_min(cfg: StereoConfig) -> tuple:
    """Frame times relative to the case time [min], by mode."""
    return (0.0,) if cfg.mode == "snapshot" else tuple(cfg.track.dts_min)


def ref_key(cfg: StereoConfig):
    """Key of the reference frame: East at the case time."""
    return (cfg.sat_east, frame_offsets_min(cfg).index(0.0))


def frames_signature(cfg: StereoConfig) -> tuple:
    """Every config field load_frames reads. Variants with equal
    signatures can share one set of loaded frames."""
    return (cfg.sat_east, cfg.sat_west, cfg.channels, frame_offsets_min(cfg),
            cfg.stripe_nsig, cfg.stripe_dilate, cfg.stripe_mask)


def load_frames(case: Case, cfg: StereoConfig = DEFAULT,
                paths: Paths | None = None) -> dict:
    """{(sat, k): Frame} for East and West at each frame offset.

    The West view is destriped (rows flagged on its full-sector BTD, masked
    in both channels when cfg.stripe_mask); diagnostics go in Frame.stripe.
    """
    paths = paths or load_paths()
    a, b = cfg.channels
    frames = {}
    for k, dt in enumerate(frame_offsets_min(cfg)):
        when = case.time + pd.Timedelta(minutes=dt)
        for sat in (cfg.sat_east, cfg.sat_west):
            f, dom = goes.fetch_channels(sat, (a, b), when, paths.goes_cache)
            da, db, stripe = f[a], f[b], {}
            if sat == cfg.sat_west:
                rows, scores = goes.stripe_rows(da, db, cfg.stripe_nsig,
                                                cfg.stripe_dilate)
                stripe = dict(max_score=float(np.nanmax(scores)),
                              n_rows=int(rows.size))
                if cfg.stripe_mask and rows.size:
                    da, db = goes.mask_rows(da, rows), goes.mask_rows(db, rows)
            frames[(sat, k)] = Frame(sat=sat, sat_lon=SAT_LON[sat],
                                     time=goes.scan_time(f[a]),
                                     view=btd_view(da, db), domain=dom,
                                     stripe=stripe)
    return frames


def load_winds(case: Case, cfg: StereoConfig = DEFAULT,
               paths: Paths | None = None):
    """ERA5 profile at the case anchor, or None if disabled/unavailable."""
    if not cfg.use_era5:
        return None
    paths = paths or load_paths()
    lat, lon = case.anchor
    return era5.fetch_winds(case.time, lat, lon, cfg.wind_levels,
                            paths.era5_cache)


# ======================================================================
# Matching
# ======================================================================
def project(frame: Frame, ref_time, h_km, grid: Grid, prep,
            wind: WindProfile | None = None) -> np.ndarray:
    """The frame's conditioned field on the grid, for a cloud at h_km.

    The grid is advected by wind(h) x (frame.time - ref_time) BEFORE the
    parallax projection -- it is the cloud that moves, not its ground
    projection, and the order matters at oblique angles.
    """
    dt = (frame.time - pd.Timestamp(ref_time)).total_seconds()
    glat, glon = grid.lat, grid.lon
    if wind is not None and dt != 0.0:
        u, v = wind.at(h_km)
        glat = grid.lat + v * dt / M_PER_DEG
        glon = grid.lon + u * dt / (M_PER_DEG * np.cos(np.deg2rad(grid.lat)))
    alat, alon = apparent_surface_latlon(glat, glon, h_km * 1000.0,
                                         frame.sat_lon)
    return prep(frame.view(alat, alon))


def local_corr(A, B, win_px):
    """Moving-window Pearson r at every pixel, NaN-aware via weights.

    Returns (r, amp): amp is A's local standard deviation. Windows with
    less than 70% valid support are NaN.
    """
    wgt = (np.isfinite(A) & np.isfinite(B)).astype(float)
    A0, B0 = np.nan_to_num(A), np.nan_to_num(B)
    w = np.maximum(uniform_filter(wgt, win_px), 1e-9)
    mean = lambda x: uniform_filter(x, win_px) / w
    mA, mB = mean(A0), mean(B0)
    cAB = mean(A0 * B0) - mA * mB
    vA, vB = mean(A0**2) - mA**2, mean(B0**2) - mB**2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cAB / np.sqrt(vA * vB)
    r[uniform_filter(wgt, win_px) < 0.7] = np.nan
    return r, np.sqrt(np.maximum(vA, 0))


def height_scan(cfg: StereoConfig) -> np.ndarray:
    return np.arange(cfg.h_lo_km, cfg.h_hi_km + 1e-9, cfg.dh_km)


def match_snapshot(frames, grid, cfg, preps, wind):
    """Correlation cube over heights from the East/West pair.

    Returns dict:
        hs         : scanned heights [km]
        R          : (n_h, *grid.shape) local correlation
        amp        : East local amplitude at the middle scan height [K]
        valid_frac : fraction of the grid where both views are valid at
                     the scan height nearest QC_REF_H_KM
    """
    E, W = cfg.sat_east, cfg.sat_west
    _, k0 = ref_key(cfg)
    ref = frames[(E, k0)]
    _, win = km_filters(grid, cfg)
    hs = height_scan(cfg)
    k_ref = int(np.argmin(np.abs(hs - QC_REF_H_KM)))
    R = np.full((hs.size,) + grid.shape, np.nan)
    amp = valid = None
    for k, h in enumerate(hs):
        A = project(ref, ref.time, h, grid, preps[E], wind)
        B = project(frames[(W, k0)], ref.time, h, grid, preps[W], wind)
        R[k], a = local_corr(A, B, win)
        if k == hs.size // 2:
            amp = a
        if k == k_ref:
            valid = float((np.isfinite(A) & np.isfinite(B)).mean())
    return dict(hs=hs, R=R, amp=amp, valid_frac=valid)


def match_track(frames, grid, cfg, preps, wind):
    """Multi-frame matching with a fitted wind (planned). Same inputs and
    outputs as match_snapshot, plus fitted (u, v) in the returned dict."""
    raise NotImplementedError("tracking mode is not implemented yet")


MATCHERS = {"snapshot": match_snapshot, "track": match_track}


def height_map(hs, R, amp, cfg: StereoConfig = DEFAULT):
    """Per-pixel height from the correlation cube.

    Argmax over heights, three-point parabolic sub-step refinement (clipped
    to +-1 step), then masked where r <= cfg.r_min, amplitude <=
    cfg.amp_min_k, or the peak sits on either end of the scan.
    Returns (H [km], rmax).
    """
    n = hs.size
    kmax = np.nanargmax(np.nan_to_num(R, nan=-9.0), axis=0)
    rmax = np.take_along_axis(R, kmax[None], 0)[0]
    km_ = np.clip(kmax, 1, n - 2)
    r0, r1, r2 = (np.take_along_axis(R, (km_ + d)[None], 0)[0]
                  for d in (-1, 0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        dh = 0.5 * (r0 - r2) / (r0 - 2 * r1 + r2)
        keep = ((rmax > cfg.r_min) & (amp > cfg.amp_min_k)
                & (kmax > 0) & (kmax < n - 1))
    H = np.where(keep, hs[km_] + np.clip(dh, -1, 1) * cfg.dh_km, np.nan)
    return H, rmax


# ======================================================================
# QC
# ======================================================================
def scene_qc(offset_s, valid_frac, H, wind, cfg: StereoConfig = DEFAULT):
    """Truth-free scene verdict (no_data is decided before matching).

    coverage : valid East/West overlap on the grid below cfg.min_valid_frac
    offset   : with a wind correction, the residual displacement from wind
               uncertainty (base + frac x speed at the median map height)
               x |offset| exceeds cfg.wind_resid_max_km; without one,
               |offset| >= cfg.max_offset_nowind_s
    Returns (verdict, diag).
    """
    h_ref = float(np.nanmedian(H)) if np.isfinite(H).any() else float(
        np.mean([cfg.h_lo_km, cfg.h_hi_km]))
    diag = dict(h_median=h_ref if np.isfinite(H).any() else np.nan,
                wind_applied=wind is not None, u_ref=np.nan, v_ref=np.nan,
                wind_disp_km=0.0, wind_resid_km=0.0)
    if wind is not None:
        u, v = wind.at(h_ref)
        spd = float(np.hypot(u, v))
        diag.update(u_ref=u, v_ref=v,
                    wind_disp_km=spd * abs(offset_s) / 1e3,
                    wind_resid_km=(cfg.wind_sigma_base
                                   + cfg.wind_sigma_frac * spd)
                    * abs(offset_s) / 1e3)
    if valid_frac is None or valid_frac < cfg.min_valid_frac:
        return QC_COVERAGE, diag
    if wind is not None:
        if diag["wind_resid_km"] > cfg.wind_resid_max_km:
            return QC_OFFSET, diag
    elif abs(offset_s) >= cfg.max_offset_nowind_s:
        return QC_OFFSET, diag
    return QC_OK, diag


def case_observability(frames, grid, H, cfg: StereoConfig = DEFAULT):
    """s_eff map for the case: structure-tensor observability of the
    reference view's BTD, parallax-corrected to the median retrieved height
    (11 km if there is none). Returns (s_eff map, diag)."""
    E, W = cfg.sat_east, cfg.sat_west
    h_ref = float(np.nanmedian(H)) if np.isfinite(H).any() else 11.0
    fE = frames[(E, 0)]
    alat, alon = apparent_surface_latlon(grid.lat, grid.lon, h_ref * 1e3, fE.sat_lon)
    o = observability(fE.view(alat, alon), grid, (E, W), cfg)
    on = np.isfinite(H) & np.isfinite(o["s_eff"])
    return o["s_eff"], dict(k_km_per_km=o["k_km_per_km"], axis_deg=o["axis_deg"],
                            s_eff_median=float(np.median(o["s_eff"][on])) if on.any() else np.nan,
                            s_eff_h_ref=h_ref)


# ======================================================================
# Orchestration
# ======================================================================
def retrieve(case: Case, cfg: StereoConfig = DEFAULT, paths: Paths | None = None,
             frames: dict | None = None, wind=_FETCH) -> Result:
    """Run the retrieval for one case.

    frames : pre-loaded frames (from load_frames with an equal
             frames_signature) -- lets variants share one load.
    wind   : "fetch" (ERA5 when the East/West offset needs it), None (no
             correction), or a WindProfile to use as given.

    Returns a Result. Maps are returned for every verdict except no_data;
    deciding which verdicts count is the caller's job.
    """
    if cfg.mode not in MATCHERS:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    if cfg.mode == "track":
        match_track(None, None, cfg, None, None)       # fail before loading

    grid = make_grid(case.bbox, cfg)
    frames = frames if frames is not None else load_frames(case, cfg, paths)
    E, W = cfg.sat_east, cfg.sat_west
    _, k0 = ref_key(cfg)
    fE, fW = frames[(E, k0)], frames[(W, k0)]
    offset = (fW.time - fE.time).total_seconds()
    lat0, lon0 = case.anchor
    diag = dict(offset_s=offset, domain_east=fE.domain, domain_west=fW.domain,
                vza_east=float(vza_deg(lat0, lon0, SAT_LON[E])),
                vza_west=float(vza_deg(lat0, lon0, SAT_LON[W])),
                stripe_max=fW.stripe.get("max_score", np.nan),
                stripe_rows=fW.stripe.get("n_rows", 0),
                grid_ny=grid.shape[0], grid_nx=grid.shape[1])
    diag["vza_max"] = max(diag["vza_east"], diag["vza_west"])

    if abs(offset) > cfg.no_data_s:
        return Result(case.id, cfg.mode, QC_NO_DATA, grid, diag=diag,
                      config_hash=cfg.config_hash())

    if isinstance(wind, str) and wind == _FETCH:
        wind = (load_winds(case, cfg, paths)
                if abs(offset) >= cfg.wind_min_offset_s else None)

    preps, pinfo = make_preps(grid, (E, W), cfg)
    m = MATCHERS[cfg.mode](frames, grid, cfg, preps, wind)
    H, rmax = height_map(m["hs"], m["R"], m["amp"], cfg)
    s_eff, odiag = case_observability(frames, grid, H, cfg)
    if cfg.obs_min_s_eff > 0:
        H = np.where(s_eff >= cfg.obs_min_s_eff, H, np.nan)
    qc, qdiag = scene_qc(offset, m["valid_frac"], H, wind, cfg)
    diag.update(qdiag)
    diag.update(pinfo)
    diag.update(odiag)
    diag.update(valid_frac=m["valid_frac"],
                map_coverage=float(np.isfinite(H).mean()))
    return Result(case.id, cfg.mode, qc, grid, height=H, r=rmax,
                  amp=m["amp"], s_eff=s_eff, diag=diag,
                  config_hash=cfg.config_hash())


# ======================================================================
# Sampling the result
# ======================================================================
def sample_at(result: Result, lat, lon, patch=5, index="nearest"):
    """Patch-median height and correlation at arbitrary points.

    patch : odd patch size in pixels.
    index : "nearest" -- patch centred on the nearest pixel; points off the
                         grid get NaN; patches truncate at the grid edge.
            "legacy"  -- the pre-restructure rule (first grid line at or
                         above the point, clipped to stay patch/2 from the
                         edge). Only for reproducing v4 results exactly.

    Returns DataFrame(h, r, s_eff), indexed like `lat` if it is a Series
    (s_eff is NaN if the result has none).
    """
    idx = lat.index if isinstance(lat, pd.Series) else None
    lat, lon = np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(
        np.asarray(lon, float))
    n = lat.size
    if result.height is None:
        return pd.DataFrame(dict(h=np.full(n, np.nan), r=np.full(n, np.nan),
                                 s_eff=np.full(n, np.nan)), index=idx)
    g, half = result.grid, patch // 2
    ny, nx = g.shape
    if index == "legacy":
        ii = np.searchsorted(g.lat1d, lat).clip(half, ny - 1 - half)
        jj = np.searchsorted(g.lon1d, lon).clip(half, nx - 1 - half)
        ok = np.ones(n, bool)
    elif index == "nearest":
        fi, fj = locate(g, lat, lon)
        ok = np.isfinite(fi) & np.isfinite(fj)
        ii = np.rint(np.nan_to_num(fi)).astype(int)
        jj = np.rint(np.nan_to_num(fj)).astype(int)
    else:
        raise ValueError(f"unknown index rule {index!r}")

    def med(Z, i, j):
        p = Z[max(i - half, 0):i + half + 1, max(j - half, 0):j + half + 1]
        return float(np.nanmedian(p)) if np.isfinite(p).any() else np.nan

    h = [med(result.height, i, j) if o else np.nan for i, j, o in zip(ii, jj, ok)]
    r = [med(result.r, i, j) if o else np.nan for i, j, o in zip(ii, jj, ok)]
    if result.s_eff is not None:        # nearest pixel, as the observability analysis
        se = [float(result.s_eff[i, j]) if o else np.nan for i, j, o in zip(ii, jj, ok)]
    else:
        se = [np.nan] * n
    return pd.DataFrame(dict(h=h, r=r, s_eff=se), index=idx)