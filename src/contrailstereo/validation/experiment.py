"""A/B experiment harness: run config variants on the same scenes and
report what the change did, with paired scene-bootstrap uncertainty.

Unit of change
--------------
A variant is a (name, StereoConfig) pair. Anything you want to test must be
expressible as a config field, so the config hash identifies every run and
results are never silently mixed. ``config_diff`` shows what differs.

Pairing
-------
Every comparison is made on the SAME scenes (and, for map-mode accuracy,
the SAME profiles), and every confidence interval comes from resampling
SCENES jointly for both variants. Between-scene spread (~0.4 km) is larger
than most effects worth testing, so unpaired comparisons hide them.

Bias correction
---------------
"corr" metrics remove a leave-one-scene-out constant: for scene s, the mean
of the OTHER scenes' mean errors (scene-level, so big scenes don't dominate).
This is the k = n_scenes limit of the scene-fold CV in ``metrics`` and is
deterministic, which matters for paired comparisons.

Dev / held-out
--------------
``assign_split`` hashes each scene's GOES filename, so the split is fixed
and independent of the scene table's profile-count ordering. Iterate on
"dev"; touch "holdout" once per decision.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import StereoConfig
from ..core.retrieve import SCENE_SCHEMA, run_scene

DATA_INVALID = ("coverage", "no_data")
SPLIT_SALT = "contrailstereo-split-v1"


# ======================================================================
# Variants
# ======================================================================
@dataclass(frozen=True)
class Variant:
    name: str
    cfg: StereoConfig

    @property
    def tag(self):
        return f"{self.name}_{self.cfg.config_hash()}"


def config_diff(a: StereoConfig, b: StereoConfig) -> pd.DataFrame:
    """Fields whose values differ between two configs."""
    da, db = asdict(a), asdict(b)
    rows = [dict(field=k, a=da[k], b=db[k]) for k in da if da[k] != db[k]]
    return pd.DataFrame(rows, columns=["field", "a", "b"])


# ======================================================================
# Scene sources
# ======================================================================
class NetCDFSource:
    """Scenes from the CALIOP collocation file (network/cache for imagery).

    ``scene`` numbers match the CLI's (same table, same ordering).
    """

    def __init__(self, nc_path, **table_kw):
        from ..data.caliop import build_scene_table
        self.scenes, self.prof_df = build_scene_table(nc_path, **table_kw)
        self.scenes = self.scenes.rename_axis("scene").reset_index()
        self._by_key = self.scenes.set_index("goes_file")

    @property
    def table(self):
        return self.scenes.rename(columns={"goes_file": "key"})

    def load(self, key, cfg):
        from ..core.retrieve import load_scene
        row = self._by_key.loc[key].copy()
        row["goes_file"] = key
        return load_scene(row, self.prof_df, cfg)


class FixtureSource:
    """The committed test fixtures (offline, 5 scenes). For checking the
    harness end to end, not for drawing conclusions."""

    def __init__(self, fixdir):
        self.fixdir = fixdir
        rows = []
        for d in sorted(glob.glob(os.path.join(fixdir, "scene_*"))):
            m = json.load(open(os.path.join(d, "meta.json")))
            rows.append(dict(scene=m["scene"], key=m["goes_file"],
                             time=pd.Timestamp(m["time"]), lat=m["lat"],
                             lon=m["lon"], n=m["n_prof"], top_km=m["top_km"],
                             top_sd=m["top_sd"], _dir=d))
        self._t = pd.DataFrame(rows)

    @property
    def table(self):
        return self._t.drop(columns="_dir")

    def load(self, key, cfg):
        import xarray as xr
        from ..core.retrieve import SceneData
        r = self._t.set_index("key").loc[key]
        d = r._dir
        fields = {}
        for sat in (cfg.sat_east, cfg.sat_west):
            fields[sat] = {}
            for ch in cfg.match_channels:
                p = os.path.join(d, f"G{sat}_C{ch:02d}.nc")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"no fixture for G{sat} C{ch:02d}")
                fields[sat][ch] = xr.open_dataset(p)
        a = cfg.match_channels[0]
        t_e = np.datetime64(fields[cfg.sat_east][a].t.values)
        t_w = np.datetime64(fields[cfg.sat_west][a].t.values)
        winds = None
        wp = os.path.join(d, "era5.npz")
        if os.path.exists(wp):
            w = np.load(wp)
            winds = (lambda hk, w=w: float(np.interp(hk, w["h"], w["u"])),
                     lambda hk, w=w: float(np.interp(hk, w["h"], w["v"])))
        offset = float((t_w - t_e) / np.timedelta64(1, "s"))
        if not (cfg.wind_min_offset_s <= abs(offset) <= cfg.no_data_s):
            winds = None
        prof = pd.read_csv(os.path.join(d, "profiles.csv"))
        prof = prof.sort_values("lat").assign(goes_file=key)
        return SceneData(when=r.time.floor("s").to_pydatetime(),
                         lat0=float(r.lat), lon0=float(r.lon), fields=fields,
                         domains={cfg.sat_east: "C", cfg.sat_west: "C"},
                         offset_s=offset, profiles=prof, winds=winds,
                         meta=dict(goes_file=key))


# ======================================================================
# Split
# ======================================================================
def assign_split(keys, holdout_frac=0.3, salt=SPLIT_SALT):
    """Deterministic dev/holdout label per key (hash of the GOES filename)."""
    out = {}
    for k in keys:
        u = int(hashlib.sha256(f"{salt}:{k}".encode()).hexdigest()[:12], 16)
        out[k] = "holdout" if u / 16**12 < holdout_frac else "dev"
    return pd.Series(out, name="split")


def select_keys(source, split="dev", n=None, holdout_frac=0.3):
    """Scene keys for a split ('dev', 'holdout' or 'all'), in table order."""
    t = source.table
    if split != "all":
        lab = assign_split(t.key, holdout_frac)
        t = t[t.key.map(lab) == split]
    keys = list(t.key)
    return keys[:n] if n else keys


# ======================================================================
# Running
# ======================================================================
SCENE_COLS = (["variant", "key", "scene", "time", "lat", "lon", "n_prof",
               "caliop_top_km", "caliop_top_sd", "vza_max", "err_km",
               "error"] + [c for c in SCENE_SCHEMA if c != "scene"])
PROF_COLS = ["key", "pid", "lat", "lon", "top_km", "h_map", "r_map"]


@dataclass
class RunResult:
    variant: Variant
    scenes: pd.DataFrame
    profiles: pd.DataFrame


def _paths(out_dir, v):
    return (os.path.join(out_dir, f"{v.tag}_scenes.csv"),
            os.path.join(out_dir, f"{v.tag}_profiles.csv"))


def _append(path, df, cols):
    df = df.reindex(columns=cols)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def load_run(out_dir, variant):
    """Read a variant's results (last row wins per scene; profile rows for
    scenes without a committed scene row are dropped)."""
    sp, pp = _paths(out_dir, variant)
    s = (pd.read_csv(sp) if os.path.exists(sp)
         else pd.DataFrame(columns=SCENE_COLS))
    s = s.drop_duplicates("key", keep="last").reset_index(drop=True)
    p = (pd.read_csv(pp) if os.path.exists(pp)
         else pd.DataFrame(columns=PROF_COLS))
    p = p[p.key.isin(s.key)].drop_duplicates(["key", "pid"], keep="last")
    return RunResult(variant, s, p.reset_index(drop=True))


def _load_signature(cfg):
    """Config fields that change what load_scene returns. Variants sharing
    a signature share one loaded SceneData per scene."""
    return (cfg.match_channels, cfg.sat_east, cfg.sat_west,
            cfg.wind_min_offset_s, cfg.no_data_s, cfg.wind_levels,
            cfg.goes_cache, cfg.era5_cache)


def run_variants(source, variants, keys, out_dir, do_map=True,
                 resume=True, retry_errors=True, verbose=True):
    """Run every variant on every key; checkpoint after each scene.

    Output files are named by variant name AND config hash, so editing a
    variant's config starts a fresh file instead of mixing results.
    Failures are recorded as qc="error" (not silently dropped) and retried
    on the next call unless retry_errors=False.

    Returns {variant name: RunResult}.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError("variant names must be unique")
    meta = source.table.set_index("key")

    done = {}
    for v in variants:
        sp, pp = _paths(out_dir, v)
        if not resume:
            for p in (sp, pp):
                if os.path.exists(p):
                    os.remove(p)
        prev = load_run(out_dir, v).scenes
        ok = prev if not retry_errors else prev[prev.qc != "error"]
        done[v.name] = set(ok.key)

    todo = [k for k in keys if any(k not in done[v.name] for v in variants)]
    if verbose:
        print(f"{len(keys)} scenes x {len(variants)} variants; "
              f"{len(todo)} scenes need work")

    for i, key in enumerate(todo):
        m = meta.loc[key]
        loaded = {}
        for v in variants:
            if key in done[v.name]:
                continue
            base = dict(variant=v.name, key=key, scene=int(m.scene),
                        time=m.time, lat=m.lat, lon=m.lon, n_prof=int(m.n),
                        caliop_top_km=float(m.top_km),
                        caliop_top_sd=float(m.top_sd))
            sp, pp = _paths(out_dir, v)
            try:
                sig = _load_signature(v.cfg)
                if sig not in loaded:
                    loaded[sig] = source.load(key, v.cfg)
                rec, pr, _ = run_scene(loaded[sig], v.cfg, do_map=do_map)
            except Exception as e:                      # noqa: BLE001
                _append(sp, pd.DataFrame([dict(
                    base, qc="error", error=f"{type(e).__name__}: {e}",
                    config_hash=v.cfg.config_hash())]), SCENE_COLS)
                if verbose:
                    print(f"  [{i+1}/{len(todo)}] {v.name:>12s} scene "
                          f"{base['scene']}: ERROR {type(e).__name__}: {e}")
                    traceback.print_exc(limit=2)
                continue
            rec = {k_: v_ for k_, v_ in rec.items() if k_ != "scene"}
            rec.update(base, vza_max=max(rec["vza_east"], rec["vza_west"]),
                       err_km=(rec["h_local"] - base["caliop_top_km"]
                               if rec["qc"] == "ok" else np.nan))
            if pr is not None:                          # profiles first,
                p = pr.assign(key=key, pid=pr.index)    # scene row commits
                _append(pp, p, PROF_COLS)
            _append(sp, pd.DataFrame([rec]), SCENE_COLS)
            if verbose:
                h = rec["h_local"]
                print(f"  [{i+1}/{len(todo)}] {v.name:>12s} scene "
                      f"{base['scene']:4d}  qc={rec['qc']:8s} "
                      f"h={h:6.2f}" if np.isfinite(h) else
                      f"  [{i+1}/{len(todo)}] {v.name:>12s} scene "
                      f"{base['scene']:4d}  qc={rec['qc']:8s} h=   nan")

    return {v.name: load_run(out_dir, v) for v in variants}


# ======================================================================
# Pairing
# ======================================================================
@dataclass
class Paired:
    """Two runs aligned on common scenes, reduced to per-scene sufficient
    statistics so every metric (and its bootstrap) is a weighted sum."""
    a: str
    b: str
    gate: float
    table: pd.DataFrame           # one row per common scene
    arr: dict = field(repr=False)  # per-scene arrays, aligned with table
    n_error: dict = field(default_factory=dict)
    hash_a: str = ""
    hash_b: str = ""


def _prof_stats(e, mask):
    e = np.where(mask, e, 0.0)
    return mask.sum(), e.sum(), (e ** 2).sum()


def pair(ra: RunResult, rb: RunResult, gate=0.6, keys=None) -> Paired:
    """Align two runs on the scenes both completed without a pipeline error.

    Map-mode populations
    --------------------
    own    : each variant's profiles passing ITS OWN r_map gate -- the
             number you would report for that variant.
    common : profiles passing the gate in BOTH -- pure accuracy on the
             same targets. A change that only moves the gate boundary
             shows up in 'own' and coverage, not here.
    """
    sa, sb = ra.scenes.set_index("key"), rb.scenes.set_index("key")
    common = sa.index.intersection(sb.index)
    if keys is not None:
        common = common.intersection(pd.Index(keys))
    n_error = {ra.variant.name: int((sa.loc[common].qc == "error").sum()),
               rb.variant.name: int((sb.loc[common].qc == "error").sum())}
    common = [k for k in common
              if sa.at[k, "qc"] != "error" and sb.at[k, "qc"] != "error"]
    if not common:
        raise ValueError("no scenes completed by both variants")

    pa = ra.profiles.set_index(["key", "pid"])
    pb = rb.profiles.set_index(["key", "pid"])
    keys_a = set(pa.index.get_level_values(0))
    keys_b = set(pb.index.get_level_values(0))
    S = len(common)
    A = {n: np.zeros(S) for n in (
        "n_all", "nA", "S1A", "S2A", "nB", "S1B", "S2B",
        "nC", "S1CA", "S2CA", "S1CB", "S2CB")}
    rows = []
    for i, k in enumerate(common):
        ea = pa.loc[k] if k in keys_a else None
        eb = pb.loc[k] if k in keys_b else None
        if ea is not None and eb is not None:
            j = ea.join(eb[["h_map", "r_map"]], rsuffix="_b", how="inner")
            err_a = (j.h_map - j.top_km).values
            err_b = (j.h_map_b - j.top_km).values
            ga = np.isfinite(err_a) & (j.r_map.values > gate)
            gb = np.isfinite(err_b) & (j.r_map_b.values > gate)
            gc = ga & gb
            A["n_all"][i] = len(j)
            A["nA"][i], A["S1A"][i], A["S2A"][i] = _prof_stats(err_a, ga)
            A["nB"][i], A["S1B"][i], A["S2B"][i] = _prof_stats(err_b, gb)
            A["nC"][i], A["S1CA"][i], A["S2CA"][i] = _prof_stats(err_a, gc)
            _, A["S1CB"][i], A["S2CB"][i] = _prof_stats(err_b, gc)
        else:
            A["n_all"][i] = sa.at[k, "n_prof"]
        rows.append(dict(key=k, scene=int(sa.at[k, "scene"]),
                         vza_max=float(sa.at[k, "vza_max"]),
                         n_prof=int(A["n_all"][i]),
                         qc_a=sa.at[k, "qc"], qc_b=sb.at[k, "qc"],
                         err_a=float(sa.at[k, "err_km"]),
                         err_b=float(sb.at[k, "err_km"]),
                         h_a=float(sa.at[k, "h_local"]),
                         h_b=float(sb.at[k, "h_local"])))
    t = pd.DataFrame(rows)

    A["okA"] = (t.qc_a == "ok").values.astype(float)
    A["okB"] = (t.qc_b == "ok").values.astype(float)
    A["validA"] = (~t.qc_a.isin(DATA_INVALID)).values.astype(float)
    A["validB"] = (~t.qc_b.isin(DATA_INVALID)).values.astype(float)
    A["eA"] = np.nan_to_num(t.err_a.values)
    A["eB"] = np.nan_to_num(t.err_b.values)
    A["okC"] = A["okA"] * A["okB"]

    # per-scene map offsets on common profiles, relative to each variant's
    # own LOO constant -- the "between-scene" term, scene by scene
    w1 = np.ones((1, S))
    for tag, S1 in (("a", A["S1CA"]), ("b", A["S1CB"])):
        c = _loo_const(w1, S1, A["nC"])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(A["nC"] > 0, S1 / A["nC"], np.nan)
        t[f"map_mean_{tag}"] = m
        t[f"map_off_{tag}"] = m - c
    t["d_abs_off"] = t.map_off_b.abs() - t.map_off_a.abs()
    t["d_abs_err"] = t.err_b.abs() - t.err_a.abs()
    t["n_common"] = A["nC"].astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["cov_a"] = A["nA"] / A["n_all"]
        t["cov_b"] = A["nB"] / A["n_all"]
    return Paired(ra.variant.name, rb.variant.name, gate, t, A, n_error,
                  ra.variant.cfg.config_hash(), rb.variant.cfg.config_hash())


# ======================================================================
# Weighted metrics  (W: (B, S) scene weights; every function returns (B,))
# ======================================================================
def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def _loo_const(W, S1, n):
    """Leave-one-scene-out constant: mean of OTHER scenes' mean errors.

    Under bootstrap weights, ALL copies of scene s are left out of its own
    constant (a scene drawn twice must not correct itself). NaN where no
    other scene has data."""
    has = (n > 0).astype(float)
    m0 = np.nan_to_num(_div(S1, n))
    tot = W @ (m0 * has)
    cnt = W @ has
    num = tot[:, None] - W * (m0 * has)[None, :]
    den = cnt[:, None] - W * has[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _prof_metrics(W, n, S1, S2, eval_mask=None):
    """bias, raw RMSE, LOO-corrected RMSE, within, between from sums.

    eval_mask restricts where errors are EVALUATED while the correction
    constant still comes from all scenes (used for stratification)."""
    c = _loo_const(W, S1, n)
    We = W if eval_mask is None else W * eval_mask[None, :]
    N = We @ n
    used = (We > 0) & (n > 0)[None, :]
    bad = (used & ~np.isfinite(c)).any(1)          # a used scene lacks a constant
    c0 = np.where(used, np.nan_to_num(c), 0.0)
    corr2 = (We * (S2 - 2 * c0 * S1 + n * c0 ** 2)).sum(1)
    within2 = We @ (S2 - np.nan_to_num(_div(S1 ** 2, n)))
    m = np.nan_to_num(_div(S1, n))
    between2 = (We * (n * (m - c0) ** 2)).sum(1)
    nanif = lambda x: np.where(bad, np.nan, x)
    return dict(bias=_div(We @ S1, N),
                rmse_raw=np.sqrt(_div(We @ S2, N)),
                rmse_corr=nanif(np.sqrt(_div(corr2, N))),
                within=np.sqrt(_div(within2, N).clip(min=0)),
                between=nanif(np.sqrt(_div(between2, N).clip(min=0))),
                n_prof=N)


def _scene_metrics(W, ok, e, eval_mask=None):
    return _prof_metrics(W, ok, ok * e, ok * e ** 2, eval_mask)


def _metric_set(P, W, eval_mask=None):
    """{metric: (valA, valB)}, each (B,)."""
    X = P.arr
    Wm = W if eval_mask is None else W * eval_mask[None, :]
    out = {}
    out["scene: yield of valid"] = (_div(Wm @ X["okA"], Wm @ X["validA"]),
                                    _div(Wm @ X["okB"], Wm @ X["validB"]))
    sa = _scene_metrics(W, X["okC"], X["eA"], eval_mask)
    sb = _scene_metrics(W, X["okC"], X["eB"], eval_mask)
    for k in ("bias", "rmse_raw", "rmse_corr"):
        out[f"scene: {k} (common ok)"] = (sa[k], sb[k])
    out["map: coverage (own gate)"] = (_div(Wm @ X["nA"], Wm @ X["n_all"]),
                                       _div(Wm @ X["nB"], Wm @ X["n_all"]))
    oa = _prof_metrics(W, X["nA"], X["S1A"], X["S2A"], eval_mask)
    ob = _prof_metrics(W, X["nB"], X["S1B"], X["S2B"], eval_mask)
    for k in ("bias", "rmse_corr"):
        out[f"map: {k} (own gate)"] = (oa[k], ob[k])
    ca = _prof_metrics(W, X["nC"], X["S1CA"], X["S2CA"], eval_mask)
    cb = _prof_metrics(W, X["nC"], X["S1CB"], X["S2CB"], eval_mask)
    for k in ("bias", "rmse_raw", "rmse_corr", "within", "between"):
        out[f"map: {k} (common)"] = (ca[k], cb[k])
    return out


BETTER = {"yield": "higher", "coverage": "higher", "bias": "abs_lower"}


def _direction(metric):
    for k, d in BETTER.items():
        if k in metric:
            return d
    return "lower"


def _verdict(d, lo, hi, direction):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    if direction == "abs_lower":
        return ""                     # judged via |bias| separately
    if lo > 0:
        return "better" if direction == "higher" else "worse"
    if hi < 0:
        return "worse" if direction == "higher" else "better"
    return "~"


def summarise(P: Paired, n_boot=2000, seed=0, ci=95) -> pd.DataFrame:
    """Headline table: A, B, delta (B - A), paired scene-bootstrap CI.

    'call' is 'better'/'worse' only when the CI excludes zero, '~' otherwise.
    For bias rows, see |bias| instead: the sign of a bias change says
    nothing on its own.
    """
    S = len(P.table)
    rng = np.random.default_rng(seed)
    W0 = np.ones((1, S))
    Wb = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
    point, boot = _metric_set(P, W0), _metric_set(P, Wb)
    q = (100 - ci) / 2
    rows = []
    for k, (va, vb) in point.items():
        ba, bb = boot[k]
        d = bb - ba
        if "bias" in k:
            d_abs = np.abs(bb) - np.abs(ba)
        lo, hi = (np.nanpercentile(d, [q, 100 - q])
                  if np.isfinite(d).any() else (np.nan, np.nan))
        direction = _direction(k)
        rows.append(dict(metric=k, A=va[0], B=vb[0], delta=vb[0] - va[0],
                         lo=lo, hi=hi,
                         call=_verdict(vb[0] - va[0], lo, hi, direction)))
        if "bias" in k:
            lo2, hi2 = (np.nanpercentile(d_abs, [q, 100 - q])
                        if np.isfinite(d_abs).any() else (np.nan, np.nan))
            rows.append(dict(metric=k.replace("bias", "|bias|"),
                             A=abs(va[0]), B=abs(vb[0]),
                             delta=abs(vb[0]) - abs(va[0]), lo=lo2, hi=hi2,
                             call=_verdict(0, lo2, hi2, "lower")))
    out = pd.DataFrame(rows).set_index("metric")
    out.attrs.update(a=P.a, b=P.b, hash_a=P.hash_a, hash_b=P.hash_b,
                     gate=P.gate, n_scenes=S,
                     n_boot=n_boot, n_error=P.n_error,
                     n_common_ok=int(P.arr["okC"].sum()),
                     n_common_prof=int(P.arr["nC"].sum()))
    return out


def by_stratum(P: Paired, col="vza_max", bins=(0, 55, 60, 65, 90),
               metrics=("map: rmse_corr (common)",
                        "scene: rmse_corr (common ok)",
                        "scene: yield of valid"),
               n_boot=2000, seed=0, ci=95, min_scenes=5) -> pd.DataFrame:
    """Delta per stratum of a per-scene column (default max VZA).

    The bias-correction constant is always fitted on ALL scenes: a
    stratum's error is judged against the population correction you would
    actually apply, not one fitted inside the stratum. Strata with fewer
    than min_scenes scenes get no call."""
    S = len(P.table)
    rng = np.random.default_rng(seed)
    W0 = np.ones((1, S))
    Wb = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
    lab = pd.cut(P.table[col], bins)
    q = (100 - ci) / 2
    rows = []
    for iv in lab.cat.categories:
        mask = (lab == iv).values.astype(float)
        if mask.sum() == 0:
            continue
        pt, bt = _metric_set(P, W0, mask), _metric_set(P, Wb, mask)
        for k in metrics:
            d = bt[k][1] - bt[k][0]
            lo, hi = (np.nanpercentile(d, [q, 100 - q])
                      if np.isfinite(d).any() else (np.nan, np.nan))
            rows.append(dict(stratum=str(iv), metric=k,
                             n_scenes=int(mask.sum()),
                             A=pt[k][0][0], B=pt[k][1][0],
                             delta=pt[k][1][0] - pt[k][0][0], lo=lo, hi=hi,
                             call=(_verdict(0, lo, hi, _direction(k))
                                   if mask.sum() >= min_scenes
                                   else f"n<{min_scenes}")))
    return pd.DataFrame(rows)


def verdict_flips(P: Paired) -> pd.DataFrame:
    """Crosstab of QC verdicts: rows = A, columns = B."""
    return pd.crosstab(P.table.qc_a, P.table.qc_b,
                       rownames=[P.a], colnames=[P.b])


def tail(P: Paired, thresh_km=1.0) -> pd.DataFrame:
    """Scenes whose map offset (vs own population constant) exceeds
    thresh_km in either variant, labelled fixed / broken / both."""
    t = P.table
    bad_a = t.map_off_a.abs() > thresh_km
    bad_b = t.map_off_b.abs() > thresh_km
    out = t[bad_a | bad_b].copy()
    out["status"] = np.select(
        [bad_a[out.index] & ~bad_b[out.index],
         ~bad_a[out.index] & bad_b[out.index]],
        ["fixed", "broken"], "both bad")
    return out[["scene", "vza_max", "n_common", "qc_a", "qc_b",
                "map_off_a", "map_off_b", "status"]].sort_values("scene")


def movers(P: Paired, n=8, by="d_abs_off"):
    """Biggest per-scene improvements and degradations."""
    t = P.table.dropna(subset=[by])
    cols = ["scene", "vza_max", "n_common", "qc_a", "qc_b",
            "map_off_a", "map_off_b", "d_abs_off", "err_a", "err_b"]
    return (t[t[by] < 0].nsmallest(n, by)[cols],
            t[t[by] > 0].nlargest(n, by)[cols])


def log_summary(S: pd.DataFrame, path, note="", split="dev"):
    """Append the headline table to a running experiment log (one row per
    metric), stamped with both variants' names, config hashes and split."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = S.reset_index().assign(
        a=S.attrs["a"], b=S.attrs["b"], hash_a=S.attrs.get("hash_a"),
        hash_b=S.attrs.get("hash_b"), gate=S.attrs["gate"], split=split,
        n_scenes=S.attrs["n_scenes"], note=note,
        logged=pd.Timestamp.now().floor("s"))
    out.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    return path