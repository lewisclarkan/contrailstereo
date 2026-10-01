"""Validation: run variants over cases, attach truth, measure, compare.

The only module that joins retrievals to truth. Everything is organised
around a Run -- one variant over a set of cases -- stored as

    {tag}_config.json     the full config (tag = name_hash)
    {tag}_cases.jsonl     one line per case: verdict + diagnostics
    {tag}_profiles.csv    one row per truth point: top_km, h, r (+ h_raw and the advection)
    {tag}_maps/           optional per-case height/r/amp netCDF

Metrics
-------
All metrics are built from per-case sums over truth points passing an
r gate, so a single run's summary, the A/B comparison and every bootstrap
use one definition. "corr" metrics remove a leave-one-case-out constant:
for case s, the mean of the OTHER cases' mean errors (case-level, so big
cases don't dominate). That is the k = n_cases limit of scene-fold CV, and
deterministic -- which matters for paired comparisons.

Which cases count is an explicit argument (qc=("ok",) by default), never
implicit.

Comparisons are paired: the same cases (and, for "common" metrics, the
same truth points), with confidence intervals from resampling CASES jointly
for both variants.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import StereoConfig, diff, load_paths
from .geometry import advect_latlon
from .retrieve import frames_signature, load_frames, retrieve, sample_at
from .types import QC_DATA_INVALID, QC_ERROR, QC_OK, Case, Result

log = logging.getLogger(__name__)

SPLIT_SALT = "contrailstereo-split-v1"
PROFILE_COLS = ["case_id", "pid", "contrail", "lat", "lon", "top_km", "h", "r", "s_eff"]
ADV_COLS = ["h_raw", "lat_adv", "lon_adv", "dt_s", "disp_km"]
CONTRAIL_GAP_KM = 5.0     # consecutive truth points further apart start a new contrail


# ======================================================================
# Variants and runs
# ======================================================================
@dataclass(frozen=True)
class Variant:
    """A named configuration. cfg is None only for imported legacy runs."""
    name: str
    cfg: Optional[StereoConfig]

    @property
    def hash(self) -> str:
        return self.cfg.config_hash() if self.cfg is not None else "legacy"

    @property
    def tag(self) -> str:
        return f"{self.name}_{self.hash}"


@dataclass
class Run:
    """One variant over a set of cases.

    cases    : one row per case -- case_id, scene, qc, error, time, anchor,
               n_truth, and every Result.diag field.
    profiles : one row per truth point (PROFILE_COLS), or None if the run
               was not validated.
    """
    variant: Variant
    cases: pd.DataFrame
    profiles: Optional[pd.DataFrame] = None

    @property
    def name(self):
        return self.variant.name


def _files(out_dir, v: Variant):
    d = Path(out_dir)
    return (d / f"{v.tag}_config.json", d / f"{v.tag}_cases.jsonl",
            d / f"{v.tag}_profiles.csv", d / f"{v.tag}_maps")


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def load_run(out_dir, variant: Variant) -> Run:
    """Read a stored run. The last line per case wins; truth rows for cases
    without a committed case line are dropped."""
    _, cp, pp, _ = _files(out_dir, variant)
    if cp.exists() and cp.stat().st_size:
        cases = pd.read_json(cp, lines=True, dtype=False)
        cases = cases.drop_duplicates("case_id", keep="last")
    else:
        cases = pd.DataFrame(columns=["case_id", "qc"])
    profiles = None
    if pp.exists():
        profiles = pd.read_csv(pp)
        for col in ("s_eff", "contrail", *ADV_COLS):
            if col not in profiles:
                profiles[col] = np.nan
        profiles = (profiles[profiles.case_id.isin(cases.case_id)]
                    .drop_duplicates(["case_id", "pid"], keep="last"))
        profiles = profiles.reset_index(drop=True)
    return Run(variant, cases.reset_index(drop=True), profiles)


# ======================================================================
# Truth
# ======================================================================
def contrail_ids(lat, lon, pid, gap_km=CONTRAIL_GAP_KM):
    """Contrail index per truth point: contiguous runs along the lidar track
    (in pid order), split where consecutive points are > gap_km apart."""
    lat, lon, pid = (np.asarray(a) for a in (lat, lon, pid))
    o = np.argsort(pid)
    seg = np.r_[0, np.cumsum(_gap_km(lat[o], lon[o]) > gap_km)]
    out = np.empty(lat.size, int)
    out[o] = seg
    return out


def truth_times(result: Result, case:Case, index="nearest") -> pd.DataFrame:
    """Lidar observation time of each case's truth points (PROFILE_COLS): height,
    r, s_eff, and the contrail each point belongs to """

    t = case.truth
    when = (pd.to_datetime(t["time"]).values if "time" in t
            else np.full(len(t), np.datetime64(case.time)))
    return when.astype("datetime64[ns]")


def advected_truth(case: Case, t_map, wind):
    """Truth positions carried from each profile's lidar time to the map time t_map.
    Returns lat, lon, dt_s (map-lidar), disp_km."""

    t = case.truth
    dt = (np.datetime64(pd.Timestamp(t_map), "ns") - truth_times(case)) / np.timedelta64(1, "s")
    h = t.top_km.values.astype(float)
    u, v = np.interp(h, wind.h, wind.u), np.interp(h, wind.h, wind.v)
    lat, lon = advect_latlon(t.lat.values, t.lon.values, u, v, dt)
    return lat, lon, dt, np.hypot(u, v) * np.abs(dt) / 1e3


def attach_truth(result:Result, case: Case, index="nearest",
                 advect=True) -> pd.DataFrame:
    """Sample the result at the case's truth points (PROFILE_COLS + ADV_COLS):
    height, r, s_eff, and the contrail each point belongs to."""

    t = case.truth
    nan = np.full(len(t), np.nan)
    lat_a = lon_a = dt = disp = nan
    raw = sample_at(result, t.lat, t.lon, index=index)
    s = raw

    if advect and result.wind is not None and result.ref_time is not None:
        lat_a, lon_a, dt, disp = advected_truth(case, result.ref_time, result.wind)
        s = sample_at(result, lat_a, lon_a, index=index)
    return pd.DataFrame(dict(case_id=case.id, pid=t.index.values,
                             contrail=contrail_ids(t.lat.values, t.lon.values,
                                                  t.index.values),
                             lat=t.lat.values, lon=t.lon.values,
                             top_km=t.top_km.values, h=s.h.values,
                             r=s.r.values, s_eff=s.s_eff.values,
                             h_raw=raw.h.values, lat_adv=lat_a, lon_adv=lon_a,
                             dt_s=dt, disp_km=disp))


def _save_map(path: Path, result: Result):
    import xarray as xr
    g = result.grid
    ds = xr.Dataset(
        {k: (("i", "j"), getattr(result, k).astype("float32"))
         for k in ("height", "r", "amp") if getattr(result, k) is not None},
        coords=dict(lat=(("i", "j"), g.lat), lon=(("i", "j"), g.lon)))
    ds.attrs.update(case_id=result.case_id, qc=result.qc,
                    config_hash=result.config_hash, grid_kind=g.kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.part")
    ds.to_netcdf(tmp)
    os.replace(tmp, path)


# ======================================================================
# Running
# ======================================================================
def run(cases, variants, out_dir, paths=None, validate=True, save_maps=False,
        resume=True, retry_errors=True, index="nearest",
        frame_loader: Callable = load_frames, verbose=True) -> dict:
    """Run every variant on every case; checkpoint after each (case, variant).

    Files are named by variant name AND config hash, so editing a variant's
    config starts fresh files rather than mixing results. Pipeline failures
    are recorded as qc="error" with the message, and retried on the next
    call unless retry_errors=False. Variants whose configs load the same
    inputs (equal frames_signature) share one load per case.

    frame_loader : (case, cfg, paths) -> frames; swap in for testing or
                   for a non-GOES source.
    Returns {variant name: Run}.
    """
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    if any(v.cfg is None for v in variants):
        raise ValueError("legacy variants cannot be run")
    if validate and any(c.truth is None for c in cases):
        raise ValueError("validate=True needs truth on every case")
    paths = paths or load_paths()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = {}
    for v in variants:
        cfp, cp, pp, mp = _files(out_dir, v)
        if not resume:
            for p in (cp, pp):
                p.unlink(missing_ok=True)
        v.cfg.to_file(cfp)
        prev = load_run(out_dir, v).cases
        if retry_errors and len(prev):
            prev = prev[prev.qc != QC_ERROR]
        done[v.name] = set(prev.case_id)

    todo = [c for c in cases if any(c.id not in done[v.name] for v in variants)]
    if verbose:
        print(f"{len(cases)} cases x {len(variants)} variants; "
              f"{len(todo)} cases need work")

    for i, case in enumerate(todo):
        frames = {}
        for v in variants:
            if case.id in done[v.name]:
                continue
            cfp, cp, pp, mp = _files(out_dir, v)
            base = dict(case_id=case.id, scene=case.meta.get("scene"),
                        time=case.time, anchor_lat=case.anchor[0],
                        anchor_lon=case.anchor[1],
                        n_truth=0 if case.truth is None else len(case.truth),
                        config_hash=v.hash)
            try:
                sig = frames_signature(v.cfg)
                if sig not in frames:
                    try:
                        frames[sig] = frame_loader(case, v.cfg, paths)
                    except Exception as e:             # noqa: BLE001
                        frames[sig] = e                # shared failure too
                if isinstance(frames[sig], Exception):
                    raise frames[sig]
                res = retrieve(case, v.cfg, paths, frames=frames[sig])
                row = dict(base, qc=res.qc, error=None, **res.diag)
                if validate:
                    prof = attach_truth(res, case, index, v.cfg.advect_truth)
                    prof.to_csv(pp, mode="a", index=False,
                                header=not pp.exists(), columns=PROFILE_COLS + ADV_COLS)
                if save_maps and res.height is not None:
                    safe = hashlib.sha1(case.id.encode()).hexdigest()[:12]
                    _save_map(mp / f"{safe}.nc", res)
            except Exception as e:                     # noqa: BLE001
                row = dict(base, qc=QC_ERROR, error=f"{type(e).__name__}: {e}")
                if verbose:
                    traceback.print_exc(limit=2)
            with open(cp, "a") as f:                   # the commit
                f.write(json.dumps({k: _jsonable(x) for k, x in row.items()})
                        + "\n")
            if verbose:
                sc = "" if base["scene"] is None else f"scene {base['scene']:4d} "
                msg = row["error"] or (
                    f"coverage {row.get('map_coverage', 0):.2f}  "
                    f"h_med {row.get('h_median', float('nan')):5.2f}")
                print(f"  [{i + 1}/{len(todo)}] {v.name:>12s} {sc}"
                      f"qc={row['qc']:8s} {msg}")
    return {v.name: load_run(out_dir, v) for v in variants}


# ======================================================================
# Legacy import
# ======================================================================
LEGACY_QC = {"no_data": "no_data", "coverage": "coverage", "offset": "offset"}


def load_legacy_run(results_csv, profiles_csv, cases, name="v4") -> Run:
    """Import pre-restructure CLI outputs as a Run.

    Scene numbers are mapped to case ids through the cases' meta["scene"]
    (same collocation file and filters as the old run). Truth rows are
    matched to pids on (scene, lat, lon). Old scene-peak verdicts
    (low_edge, no_peak, low_r, ...) judged a product that no longer exists,
    so they map to "ok"; the original is kept in qc_legacy.
    """
    by_scene = {c.meta["scene"]: c for c in cases if "scene" in c.meta}
    res = pd.read_csv(results_csv)
    rows = []
    for _, r in res.iterrows():
        c = by_scene.get(int(r.scene))
        if c is None:
            continue
        d = {k: r[k] for k in ("offset_s", "vza_east", "vza_west",
                               "map_coverage", "valid_frac") if k in r}
        rows.append(dict(case_id=c.id, scene=int(r.scene), time=c.time,
                         anchor_lat=c.anchor[0], anchor_lon=c.anchor[1],
                         n_truth=len(c.truth), config_hash="legacy",
                         qc=LEGACY_QC.get(r.qc, QC_OK), qc_legacy=r.qc,
                         error=None, **d))
    cases_df = pd.DataFrame(rows)
    if {"vza_east", "vza_west"} <= set(cases_df.columns):
        cases_df["vza_max"] = cases_df[["vza_east", "vza_west"]].max(axis=1)

    old = pd.read_csv(profiles_csv)
    truth = pd.concat([c.truth.assign(case_id=c.id, scene=c.meta["scene"])
                       .rename_axis("pid").reset_index() for c in cases
                       if c.id in set(cases_df.case_id)])
    key = lambda d: (d.scene.astype(int).astype(str) + "|"
                     + d.lat.round(5).astype(str) + "|"
                     + d.lon.round(5).astype(str))
    truth["_k"], old["_k"] = key(truth), key(old)
    m = truth.merge(old[["_k", "h_map", "r_map"]].drop_duplicates("_k"),
                    on="_k", how="left")
    n_miss = int(m.h_map.isna().sum() - old.h_map.isna().sum())
    if n_miss > 0.01 * len(old):
        log.warning("legacy import: %d truth points without a legacy match",
                    n_miss)
    m = m.rename(columns=dict(h_map="h", r_map="r")).assign(s_eff=np.nan)
    m["contrail"] = 0
    for cid, g in m.groupby("case_id"):
        m.loc[g.index, "contrail"] = contrail_ids(g.lat.values, g.lon.values, g.pid.values)
    prof = m[PROFILE_COLS].assign(**{c: np.nan for c in ADV_COLS})
    return Run(Variant(name, None), cases_df, prof.reset_index(drop=True))


# ======================================================================
# Split
# ======================================================================
def assign_split(ids, holdout_frac=0.3, salt=SPLIT_SALT) -> pd.Series:
    """Deterministic dev/holdout label per case id (hash-based, so it does
    not depend on case order)."""
    out = {}
    for k in ids:
        u = int(hashlib.sha256(f"{salt}:{k}".encode()).hexdigest()[:12], 16)
        out[k] = "holdout" if u / 16**12 < holdout_frac else "dev"
    return pd.Series(out, name="split")


def select(cases, split="dev", n=None, holdout_frac=0.3, max_vza=None,
           sats=(16, 17), order="random"):
    """Cases in a split ('dev' | 'holdout' | 'all').

    order : "random" (default) -- a fixed pseudo-random order (hash of the
            case id, with a different salt from the split), so the first n
            cases are a REPRESENTATIVE subset and the same on every run.
            "table" -- the input order. For CALIOP cases that is profile
            count, largest first, so the first n are the largest cases.
    max_vza : keep only cases whose anchor is seen by every satellite in
              `sats` at a viewing zenith angle <= max_vza [deg].

    Scene numbers are unaffected (they live in case.meta["scene"]).
    """
    if max_vza is not None:
        from .config import SAT_LON
        from .geometry import vza_deg
        cases = [c for c in cases
                 if max(vza_deg(*c.anchor, SAT_LON[s]) for s in sats) <= max_vza]
    if split != "all":
        lab = assign_split([c.id for c in cases], holdout_frac)
        cases = [c for c in cases if lab[c.id] == split]
    if order == "random":
        key = lambda c: hashlib.sha256(f"{SPLIT_SALT}:order:{c.id}".encode()).hexdigest()
        cases = sorted(cases, key=key)
    elif order != "table":
        raise ValueError(f"unknown order {order!r}")
    return list(cases[:n]) if n else list(cases)


# ======================================================================
# Per-case sums
# ======================================================================
def _usable(run: Run, qc) -> pd.Series:
    q = run.cases.set_index("case_id").qc
    return q.isin(qc) if qc is not None else ~q.isin([QC_ERROR])


def _passes(p, gate, s_min=0.0):
    """Truth points passing the r gate and (if s_min > 0) the s_eff gate.
    Points with no s_eff (no-data scenes, off-grid points) fail the s_eff
    gate, just as points with no height fail the r gate."""
    ok = np.isfinite(p.h.values) & (p.r.values > gate)
    if s_min and s_min > 0:
        if "s_eff" not in p:
            raise ValueError("s_min > 0 needs an s_eff column in the profiles")
        with np.errstate(invalid="ignore"):
            ok &= np.nan_to_num(p.s_eff.values, nan=-np.inf) >= s_min
    return ok


def _require_s_eff(run: Run, s_min):
    """Run-level check: gating on s_eff needs a run that stored it."""
    if s_min and s_min > 0 and (run.profiles is None or "s_eff" not in run.profiles
                                or run.profiles.s_eff.isna().all()):
        raise ValueError(f"run {run.name!r} has no stored s_eff (made before s_eff "
                         "was stored?); rerun it to gate on s_eff")


def _case_sums(run: Run, ids, gate, qc, s_min=0.0):
    """Arrays aligned with ids:
        n_all : truth points in the case
        n_use : truth points in the case if it is usable, else 0
        n, S1, S2 : count, sum and sum of squares of error (h - top) over
                    points passing the gate in usable cases."""
    if run.profiles is None:
        raise ValueError(f"run {run.name!r} has no truth profiles")
    _require_s_eff(run, s_min)
    use = _usable(run, qc)
    p = run.profiles
    e = (p.h - p.top_km).values
    ok = _passes(p, gate, s_min) & p.case_id.map(use).fillna(
        False).values.astype(bool)
    d = pd.DataFrame(dict(case_id=p.case_id, one=1.0, n=ok.astype(float),
                          S1=np.where(ok, e, 0.0), S2=np.where(ok, e * e, 0.0)))
    g = d.groupby("case_id")[["one", "n", "S1", "S2"]].sum().reindex(ids,
                                                                    fill_value=0)
    usable = pd.Series(ids).map(use).fillna(False).values.astype(bool)
    return dict(n_all=g.one.values, n_use=np.where(usable, g.one.values, 0.0),
                n=g.n.values, S1=g.S1.values,
                S2=g.S2.values)


# ======================================================================
# Weighted metrics  (W: (B, S) case weights; each returns (B,))
# ======================================================================
def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _loo_const(W, S1, n):
    """Mean of the OTHER cases' mean errors. Under bootstrap weights all
    copies of case s are left out of its own constant. NaN where no other
    case has data."""
    has = (n > 0).astype(float)
    m0 = np.nan_to_num(_div(S1, n))
    tot, cnt = W @ (m0 * has), W @ has
    num = tot[:, None] - W * (m0 * has)[None, :]
    den = cnt[:, None] - W * has[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _metrics(W, n, S1, S2, eval_mask=None):
    """bias, rmse_raw, rmse_corr, within, between, n from per-case sums.
    eval_mask restricts where errors are evaluated; the correction constant
    always comes from all cases."""
    c = _loo_const(W, S1, n)
    We = W if eval_mask is None else W * eval_mask[None, :]
    N = We @ n
    used = (We > 0) & (n > 0)[None, :]
    bad = (used & ~np.isfinite(c)).any(1)
    c0 = np.where(used, np.nan_to_num(c), 0.0)
    corr2 = (We * (S2 - 2 * c0 * S1 + n * c0**2)).sum(1)
    within2 = We @ (S2 - np.nan_to_num(_div(S1**2, n)))
    m = np.nan_to_num(_div(S1, n))
    between2 = (We * (n * (m - c0) ** 2)).sum(1)
    nanif = lambda x: np.where(bad, np.nan, x)
    return dict(bias=_div(We @ S1, N),
                rmse_raw=np.sqrt(_div(We @ S2, N)),
                rmse_corr=nanif(np.sqrt(_div(corr2, N))),
                within=np.sqrt(np.clip(_div(within2, N), 0, None)),
                between=nanif(np.sqrt(np.clip(_div(between2, N), 0, None))),
                n=N)


def _boot_weights(S, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)


def _ci(x, ci=95):
    x = x[np.isfinite(x)]
    if not x.size:
        return (np.nan, np.nan)
    q = (100 - ci) / 2
    return tuple(np.percentile(x, [q, 100 - q]))


# ======================================================================
# Single run
# ======================================================================
def summary(run: Run, gate=0.6, qc=(QC_OK,), n_boot=0, seed=0, s_min=0.0) -> dict:
    """Accuracy of one run against truth.

    n_cases/n_usable : cases completed / with qc in `qc`
    yield            : n_usable / n_cases -- cases lost to data or geometry
    coverage         : gated points / truth points IN USABLE CASES -- pixels
                       lost within cases the retrieval could attempt
    bias, rmse_raw, rmse_corr, within (matching precision inside a case),
    between (case-to-case offset spread), r2 (per-point, Meijer-comparable)
    With n_boot > 0, adds <metric>_ci from a case bootstrap.
    """
    cs = run.cases[run.cases.qc != QC_ERROR]
    ids = list(cs.case_id)
    s = _case_sums(run, ids, gate, qc, s_min)
    W1 = np.ones((1, len(ids)))
    m = {k: float(v[0]) for k, v in _metrics(W1, s["n"], s["S1"], s["S2"]).items()}
    p = run.profiles
    use = _usable(run, qc)
    q = p[p.case_id.map(use).fillna(False).astype(bool) & _passes(p, gate, s_min)]
    y = q.top_km.values
    ss_tot = float(((y - y.mean()) ** 2).sum()) if len(q) > 1 else 0.0
    r2 = (1 - float(((q.h - q.top_km) ** 2).sum()) / ss_tot
          if ss_tot > 0 else np.nan)
    out = dict(gate=gate, s_min=s_min, n_cases=len(ids), n_usable=int(use.reindex(ids).sum()),
               n_cases_scored=int((s["n"] > 0).sum()),
               **{"yield": float(use.reindex(ids).mean()) if ids else np.nan},
               coverage=float(s["n"].sum() / s["n_use"].sum())
               if s["n_use"].sum() else np.nan,
               r2=float(r2), **m)
    if n_boot:
        Wb = _boot_weights(len(ids), n_boot, seed)
        mb = _metrics(Wb, s["n"], s["S1"], s["S2"])
        for k in ("bias", "rmse_corr", "within", "between"):
            out[f"{k}_ci"] = _ci(mb[k])
    return out


def gate_ladder(run: Run, gates=(0.5, 0.6, 0.7), qc=(QC_OK,), s_min=0.0) -> pd.DataFrame:
    """Quality/coverage trade-off across r gates; reports scored cases per
    gate so a few-case ladder is not mistaken for a population result."""
    return pd.DataFrame([summary(run, g, qc, s_min=s_min) for g in gates]).set_index("gate")


def case_table(run: Run, gate=0.6, qc=(QC_OK,), s_min=0.0) -> pd.DataFrame:
    """Per-case accuracy, sorted by share of the run's squared corrected
    error (largest first) -- where the error actually comes from.

    n, bias (mean error), sd (within-case scatter), offset (bias minus the
    leave-one-case-out constant: this case's contribution to 'between'),
    sse_share (share of total squared corrected error), plus qc, scene and
    vza_max where available.
    """
    cs = run.cases[run.cases.qc != QC_ERROR]
    ids = list(cs.case_id)
    s = _case_sums(run, ids, gate, qc, s_min)
    W1 = np.ones((1, len(ids)))
    c = _loo_const(W1, s["S1"], s["n"])[0]
    n, S1, S2 = s["n"], s["S1"], s["S2"]
    c0 = np.nan_to_num(c)
    sse = S2 - 2 * c0 * S1 + n * c0**2
    t = pd.DataFrame(dict(case_id=ids, n=n.astype(int),
                          bias=_div(S1, n),
                          sd=np.sqrt(np.clip(_div(S2 - np.nan_to_num(
                              _div(S1**2, n)), n), 0, None)),
                          offset=_div(S1, n) - c,
                          sse_share=sse / sse[n > 0].sum() if (n > 0).any()
                          else np.nan))
    extra = [k for k in ("scene", "qc", "vza_max", "map_coverage")
             if k in cs.columns]
    t = t.merge(cs[["case_id"] + extra], on="case_id", how="left")
    return (t[t.n > 0].sort_values("sse_share", ascending=False)
            .reset_index(drop=True))


# ======================================================================
# Per-contrail metrics
# ======================================================================
def _gap_km(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def contrail_table(run: Run, gate=0.6, qc=(QC_OK,), gap_km=CONTRAIL_GAP_KM, min_points=3,
                   agg="median", s_min=0.0) -> pd.DataFrame:
    """One row per contrail: the unit Meijer et al. (2024) report RMSE in.

    A contrail is a contiguous run of truth points along the lidar track
    within one case, split wherever consecutive points (in pid order) are
    more than gap_km apart. Its retrieved height is the `agg` of its points
    passing the r gate; it is scored if at least min_points pass.

    Columns: case_id, contrail, n_truth, n_gated, top_km (mean truth),
    h (aggregated stereo), err = h - top_km, scored, plus qc / scene /
    vza_max where available. Contrails in cases whose qc is not in `qc`
    are listed with scored=False (they count against contrail coverage).
    """
    if run.profiles is None:
        raise ValueError(f"run {run.name!r} has no truth profiles")
    _require_s_eff(run, s_min)
    cs = run.cases[run.cases.qc != QC_ERROR].set_index("case_id")
    use = _usable(run, qc)
    rows = []
    stored = ("contrail" in run.profiles and run.profiles.contrail.notna().all()
              and gap_km == CONTRAIL_GAP_KM)
    for cid, g in run.profiles[run.profiles.case_id.isin(cs.index)].groupby("case_id"):
        g = g.sort_values("pid")
        seg = (g.contrail.astype(int).values if stored else
               np.r_[0, np.cumsum(_gap_km(g.lat.values, g.lon.values) > gap_km)])
        ok_case = bool(use.get(cid, False))
        for k, s_ in g.groupby(seg):
            ok = ok_case & _passes(s_, gate, s_min)
            hv = s_.h.values[ok]
            h = (float(np.median(hv)) if agg == "median" else float(np.mean(hv))
                 ) if ok.sum() >= min_points else np.nan
            se = s_.s_eff.values if "s_eff" in s_ else np.full(len(s_), np.nan)
            rows.append(dict(case_id=cid, contrail=f"{cid}#{k}",
                             n_truth=len(s_), n_gated=int(ok.sum()),
                             top_km=float(s_.top_km.mean()), h=h,
                             s_eff=float(np.nanmedian(se)) if np.isfinite(se).any() else np.nan))
    t = pd.DataFrame(rows)
    t["err"] = t.h - t.top_km
    t["scored"] = t.h.notna()
    extra = [k for k in ("scene", "qc", "vza_max") if k in cs.columns]
    return t.merge(cs[extra].reset_index(), on="case_id", how="left")


def contrail_summary(run: Run, gate=0.6, qc=(QC_OK,), gap_km=CONTRAIL_GAP_KM, min_points=3,
                     agg="median", n_boot=2000, seed=0, s_min=0.0) -> dict:
    """Per-contrail accuracy, each contrail weighted equally.

    n_contrails        : contrails in usable cases
    contrail_coverage  : share of those that are scored
    bias, rmse_raw     : over scored contrails (the Meijer-comparable pair)
    rmse_corr          : after removing a leave-one-CASE-out constant (the
                         mean error of contrails in other cases), so
                         contrails from the same scene never correct each
                         other
    *_ci               : case-bootstrap 95% intervals (contrails in one
                         case share imagery and weather, so cases are the
                         independent unit)
    """
    t = contrail_table(run, gate, qc, gap_km, min_points, agg, s_min)
    use = _usable(run, qc)
    t = t[t.case_id.map(use).fillna(False).astype(bool)]
    sc = t[t.scored]
    out = dict(gate=gate, s_min=s_min, n_contrails=len(t), n_scored=len(sc),
               contrail_coverage=len(sc) / max(len(t), 1),
               n_cases=int(sc.case_id.nunique()))
    if len(sc) < 2:
        return out

    def stats(d):
        e = d.err.values
        tot, n = e.sum(), len(e)
        per = d.groupby("case_id").err.agg(["sum", "count"])
        c = d.case_id.map((tot - per["sum"]) / (n - per["count"])).values
        return dict(bias=float(e.mean()), rmse_raw=float(np.sqrt((e**2).mean())),
                    rmse_corr=float(np.sqrt(np.nanmean((e - c) ** 2))))
    out.update(stats(sc))
    if n_boot:
        rng = np.random.default_rng(seed)
        cases = sc.case_id.unique()
        groups = {k: g for k, g in sc.groupby("case_id")}
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(cases, cases.size)
            d = pd.concat([groups[k].assign(case_id=f"{k}~{i}")
                           for i, k in enumerate(pick)])
            boot.append(stats(d))
        b = pd.DataFrame(boot)
        for k in ("bias", "rmse_raw", "rmse_corr"):
            out[f"{k}_ci"] = tuple(np.nanpercentile(b[k], [2.5, 97.5]))
    return out


# ======================================================================
# Paired comparison
# ======================================================================
@dataclass
class Paired:
    """Two runs aligned on common cases, reduced to per-case sums."""
    a: Variant
    b: Variant
    gate: float
    qc: tuple
    table: pd.DataFrame                    # one row per common case
    arr: dict = field(repr=False)          # per-case arrays, aligned
    n_error: dict = field(default_factory=dict)


def pair(run_a: Run, run_b: Run, gate=0.6, qc=(QC_OK,), ids=None, s_min=0.0) -> Paired:
    """Align two runs on the cases both completed without error.

    own    : each variant's gated points in ITS usable cases -- what you
             would report for that variant.
    common : points gated in both, in cases usable in both -- pure
             accuracy on identical targets.
    """
    for r_ in (run_a, run_b):
        if r_.profiles is not None and "s_eff" not in r_.profiles:
            r_.profiles = r_.profiles.assign(s_eff=np.nan)
    ca, cb = run_a.cases.set_index("case_id"), run_b.cases.set_index("case_id")
    common = ca.index.intersection(cb.index)
    if ids is not None:
        common = common.intersection(pd.Index(list(ids)))
    n_error = {run_a.name: int((ca.loc[common].qc == QC_ERROR).sum()),
               run_b.name: int((cb.loc[common].qc == QC_ERROR).sum())}
    common = [k for k in common
              if ca.at[k, "qc"] != QC_ERROR and cb.at[k, "qc"] != QC_ERROR]
    if not common:
        raise ValueError("no cases completed by both runs")

    A = {}
    for tag, r in (("A", run_a), ("B", run_b)):
        s = _case_sums(r, common, gate, qc, s_min)
        A["n_all"] = s["n_all"]
        A[f"nuse{tag}"] = s["n_use"]
        A[f"n{tag}"], A[f"S1{tag}"], A[f"S2{tag}"] = s["n"], s["S1"], s["S2"]

    ua, ub = _usable(run_a, qc), _usable(run_b, qc)
    j = run_a.profiles.merge(run_b.profiles[["case_id", "pid", "h", "r", "s_eff"]],
                             on=["case_id", "pid"], suffixes=("_a", "_b"))
    j = j[j.case_id.isin(common)]
    ea, eb = (j.h_a - j.top_km).values, (j.h_b - j.top_km).values
    _require_s_eff(run_a, s_min); _require_s_eff(run_b, s_min)
    with np.errstate(invalid="ignore"):
        sa_ok = (np.nan_to_num(j.s_eff_a.values, nan=-np.inf) >= s_min) if s_min else np.ones(len(j), bool)
        sb_ok = (np.nan_to_num(j.s_eff_b.values, nan=-np.inf) >= s_min) if s_min else np.ones(len(j), bool)
    both = (np.isfinite(ea) & np.isfinite(eb) & (j.r_a.values > gate)
            & (j.r_b.values > gate) & sa_ok & sb_ok
            & j.case_id.map(ua).fillna(False).values.astype(bool)
            & j.case_id.map(ub).fillna(False).values.astype(bool))
    d = pd.DataFrame(dict(case_id=j.case_id, n=both.astype(float),
                          S1a=np.where(both, ea, 0), S2a=np.where(both, ea**2, 0),
                          S1b=np.where(both, eb, 0), S2b=np.where(both, eb**2, 0)))
    g = d.groupby("case_id").sum().reindex(common, fill_value=0)
    A.update(nC=g.n.values, S1CA=g.S1a.values, S2CA=g.S2a.values,
             S1CB=g.S1b.values, S2CB=g.S2b.values)
    A["useA"] = ua.reindex(common).values.astype(float)
    A["useB"] = ub.reindex(common).values.astype(float)

    t = pd.DataFrame(dict(case_id=common,
                          scene=ca.loc[common, "scene"].values
                          if "scene" in ca else None,
                          qc_a=ca.loc[common, "qc"].values,
                          qc_b=cb.loc[common, "qc"].values,
                          n_truth=A["n_all"].astype(int),
                          n_common=A["nC"].astype(int)))
    for col in ("vza_max", "offset_s"):
        if col in ca:
            t[col] = ca.loc[common, col].values.astype(float)
    W1 = np.ones((1, len(common)))
    for tag, S1 in (("a", A["S1CA"]), ("b", A["S1CB"])):
        c = _loo_const(W1, S1, A["nC"])[0]
        t[f"map_mean_{tag}"] = _div(S1, A["nC"])
        t[f"map_off_{tag}"] = t[f"map_mean_{tag}"] - c
    t["d_abs_off"] = t.map_off_b.abs() - t.map_off_a.abs()
    t["cov_a"] = _div(A["nA"], A["nuseA"])
    t["cov_b"] = _div(A["nB"], A["nuseB"])
    return Paired(run_a.variant, run_b.variant, gate, tuple(qc) if qc else (),
                  t, A, n_error)


def _metric_set(P: Paired, W, eval_mask=None) -> dict:
    """{metric: (A values, B values)}, each (B,)."""
    X = P.arr
    Wm = W if eval_mask is None else W * eval_mask[None, :]
    out = {"yield": (_div(Wm @ X["useA"], Wm.sum(1)),
                     _div(Wm @ X["useB"], Wm.sum(1))),
           "coverage (own)": (_div(Wm @ X["nA"], Wm @ X["nuseA"]),
                              _div(Wm @ X["nB"], Wm @ X["nuseB"]))}
    oa = _metrics(W, X["nA"], X["S1A"], X["S2A"], eval_mask)
    ob = _metrics(W, X["nB"], X["S1B"], X["S2B"], eval_mask)
    for k in ("bias", "rmse_corr"):
        out[f"{k} (own)"] = (oa[k], ob[k])
    ca = _metrics(W, X["nC"], X["S1CA"], X["S2CA"], eval_mask)
    cb = _metrics(W, X["nC"], X["S1CB"], X["S2CB"], eval_mask)
    for k in ("bias", "rmse_raw", "rmse_corr", "within", "between"):
        out[f"{k} (common)"] = (ca[k], cb[k])
    return out


def _direction(metric):
    if metric.startswith(("yield", "coverage")):
        return "higher"
    return "abs" if metric.startswith("bias") else "lower"


def _call(lo, hi, direction):
    if not (np.isfinite(lo) and np.isfinite(hi)) or direction == "abs":
        return ""
    if lo > 0:
        return "better" if direction == "higher" else "worse"
    if hi < 0:
        return "worse" if direction == "higher" else "better"
    return "~"


def compare(P: Paired, n_boot=2000, seed=0, ci=95, min_cases=10) -> pd.DataFrame:
    """Headline table: A, B, delta = B - A, paired case-bootstrap CI, call.

    call is better/worse only when the CI excludes zero, '~' otherwise, and
    "n<min_cases" when there are too few cases for a bootstrap CI to mean
    anything (the CI is still shown).
    Bias rows get a companion |bias| row: a signed bias change means
    nothing on its own.
    """
    S = len(P.table)
    W0, Wb = np.ones((1, S)), _boot_weights(S, n_boot, seed)
    pt, bt = _metric_set(P, W0), _metric_set(P, Wb)
    rows = []
    for k, (va, vb) in pt.items():
        ba, bb = bt[k]
        lo, hi = _ci(bb - ba, ci)
        rows.append(dict(metric=k, A=va[0], B=vb[0], delta=vb[0] - va[0],
                         lo=lo, hi=hi, call=_call(lo, hi, _direction(k))))
        if k.startswith("bias"):
            lo, hi = _ci(np.abs(bb) - np.abs(ba), ci)
            rows.append(dict(metric="|" + k.replace(" ", "| ", 1), A=abs(va[0]),
                             B=abs(vb[0]), delta=abs(vb[0]) - abs(va[0]),
                             lo=lo, hi=hi, call=_call(lo, hi, "lower")))
    out = pd.DataFrame(rows).set_index("metric")
    if S < min_cases:
        out["call"] = f"n<{min_cases}"
    out.attrs.update(a=P.a.name, b=P.b.name, hash_a=P.a.hash, hash_b=P.b.hash,
                     gate=P.gate, qc=P.qc, n_cases=S, n_boot=n_boot, ci=ci,
                     n_error=P.n_error,
                     n_common_points=int(P.arr["nC"].sum()),
                     config_diff=(diff(P.a.cfg, P.b.cfg)
                                  if P.a.cfg and P.b.cfg else None))
    return out


def by_stratum(P: Paired, col="vza_max", bins=(0, 55, 60, 65, 90),
               metrics=("rmse_corr (common)", "coverage (own)", "yield"),
               n_boot=2000, seed=0, ci=95, min_cases=5) -> pd.DataFrame:
    """Delta per stratum of a per-case column. The correction constant is
    fitted on ALL cases, so each stratum is judged against the correction
    you would actually apply. Strata below min_cases get no call."""
    S = len(P.table)
    W0, Wb = np.ones((1, S)), _boot_weights(S, n_boot, seed)
    lab = pd.cut(P.table[col], bins)
    rows = []
    for iv in lab.cat.categories:
        mask = (lab == iv).values.astype(float)
        if not mask.sum():
            continue
        pt, bt = _metric_set(P, W0, mask), _metric_set(P, Wb, mask)
        for k in metrics:
            lo, hi = _ci(bt[k][1] - bt[k][0], ci)
            rows.append(dict(stratum=str(iv), metric=k, n_cases=int(mask.sum()),
                             A=pt[k][0][0], B=pt[k][1][0],
                             delta=pt[k][1][0] - pt[k][0][0], lo=lo, hi=hi,
                             call=(_call(lo, hi, _direction(k))
                                   if mask.sum() >= min_cases
                                   else f"n<{min_cases}")))
    return pd.DataFrame(rows)


def verdict_flips(P: Paired) -> pd.DataFrame:
    """Crosstab of QC verdicts: rows = A, columns = B."""
    return pd.crosstab(P.table.qc_a, P.table.qc_b,
                       rownames=[P.a.name], colnames=[P.b.name])


def tail(P: Paired, km=1.0) -> pd.DataFrame:
    """Cases whose map offset exceeds km in either run: fixed/broken/both."""
    t = P.table
    ba, bb = t.map_off_a.abs() > km, t.map_off_b.abs() > km
    out = t[ba | bb].copy()
    out["status"] = np.select([ba[out.index] & ~bb[out.index],
                               ~ba[out.index] & bb[out.index]],
                              ["fixed", "broken"], "both")
    cols = [c for c in ("case_id", "scene", "vza_max", "n_common", "qc_a",
                        "qc_b", "map_off_a", "map_off_b", "status") if c in out]
    return out[cols]


def movers(P: Paired, n=8, by="d_abs_off"):
    """(largest improvements, largest degradations) per case."""
    t = P.table.dropna(subset=[by])
    cols = [c for c in ("case_id", "scene", "vza_max", "n_common", "qc_a",
                        "qc_b", "map_off_a", "map_off_b", "d_abs_off") if c in t]
    return (t[t[by] < 0].nsmallest(n, by)[cols],
            t[t[by] > 0].nlargest(n, by)[cols])


def log_summary(S: pd.DataFrame, path, note="", split="dev") -> Path:
    """Append a compare() table to a running log, stamped with both
    variants, hashes, gate, qc and split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = S.attrs
    out = S.reset_index().assign(
        a=a["a"], b=a["b"], hash_a=a["hash_a"], hash_b=a["hash_b"],
        gate=a["gate"], qc=",".join(a["qc"]), split=split,
        n_cases=a["n_cases"], note=note, logged=pd.Timestamp.now().floor("s"))
    out.to_csv(path, mode="a", index=False, header=not path.exists())
    return path