"""Generate metrics from DataFrames or CSV paths

scene df = one row per scene
profile df = one row per mapped profile

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _load(df_or_path):
    return pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path


def _rmse(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan


def loo_corrected_rmse(e):
    """Bias-corrected RMSE where each point's correction excludes itself
    (the honest small-n version of subtracting the mean)."""
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if e.size < 3:
        return np.nan
    return float(np.sqrt(np.mean(
        [(e[i] - np.delete(e, i).mean()) ** 2 for i in range(e.size)])))


def bootstrap_ci(e, n_boot=3000, seed=0):
    """95% CI on the bias-corrected RMSE by scene-level resampling."""
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    rng = np.random.default_rng(seed)
    stats = [np.sqrt((((s := rng.choice(e, e.size)) - s.mean()) ** 2).mean())
             for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ----------------------------------------------------------- scene mode
def scene_summary(scene_df):
    """Yield (with the data-valid decomposition), bias, LOO-corr, CI."""
    df = _load(scene_df)
    ok = df[df.qc == "ok"]
    data_invalid = df.qc.isin(["coverage", "no_data"]).sum()
    e = ok.err_km.values
    lo, hi = bootstrap_ci(e)
    return dict(
        n_scenes=len(df), n_ok=len(ok), n_data_invalid=int(data_invalid),
        yield_of_valid=len(ok) / max(len(df) - data_invalid, 1),
        bias=float(np.nanmean(e)),
        rmse_raw=_rmse(e),
        rmse_loo_corr=loo_corrected_rmse(e),
        ci95=(lo, hi),
        verdicts=df.qc.value_counts().to_dict())


def compare(scene_df_a, scene_df_b, tol_h=0.10):
    """Version-transition diff: verdict flips + height drift on
    co-retrieved scenes. Returns (flips DataFrame, drift dict)."""
    a, b = _load(scene_df_a), _load(scene_df_b)
    m = b.merge(a, on="scene", suffixes=("", "_a"))
    flips = m[m.qc != m.qc_a][["scene", "qc_a", "qc",
                               "err_km_a", "err_km"]]
    both = m[np.isfinite(m.h_local) & np.isfinite(m.h_local_a)]
    dh = (both.h_local - both.h_local_a).abs()
    drift = dict(n_co=len(both), median_dh=float(dh.median()),
                 max_dh=float(dh.max()),
                 n_beyond_tol=int((dh > tol_h).sum()))
    return flips, drift


# ------------------------------------------------------------- map mode
def map_tradeoff(profile_df, gates=(0.5, 0.6, 0.7)):
    """Quality/coverage ladder. Always reports n_scenes per gate --
    a few-scene ladder must not be read as the population result."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    rows = []
    for thr in gates:
        q = pp[pp.r_map > thr]
        e = (q.h_map - q.top_km).values
        if e.size < 5:
            continue
        rows.append(dict(gate=thr, n_profiles=int(e.size),
                         n_scenes=int(q.scene.nunique()),
                         coverage=e.size / len(pp),
                         rmse_raw=_rmse(e),
                         rmse_corr_insample=_rmse(e - e.mean()),
                         bias=float(e.mean())))
    return pd.DataFrame(rows)


def meijer_metric(profile_df, gate=0.6):
    """Per-profile raw RMSE + R^2, his Figure-4 computation, plus the
    truth-SD needed for cross-population R^2 normalisation."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    q = pp[pp.r_map > gate]
    y, yh = q.top_km.values, q.h_map.values
    e = yh - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(gate=gate, n=len(q),
                rmse_raw=_rmse(e),
                r2_raw=float(1 - np.sum(e ** 2) / ss_tot),
                truth_sd=float(y.std()),
                bias=float(e.mean()))


def decomposition(profile_df, gate=0.6):
    """total^2 = within-scene^2 + between-scene^2 on centred errors.
    within = matching precision; between = per-scene offset spread
    (dominated by the depth-varying effective-height term)."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    q = pp[pp.r_map > gate].assign(e=lambda d: d.h_map - d.top_km)
    ec = q.e - q.e.mean()
    total = float(np.sqrt((ec ** 2).mean()))
    within = float(np.sqrt(
        (q.groupby("scene").e.transform(lambda x: x - x.mean()) ** 2).mean()))
    between = float(np.sqrt(max(total ** 2 - within ** 2, 0.0)))
    return dict(gate=gate, n=len(q), n_scenes=int(q.scene.nunique()),
                total=total, within=within, between=between)


def bias_correct_cv(profile_df, gate=0.6, k=5, seed=0):
    """The quotable corrected number: per-profile RMSE after a constant
    bias correction fitted on OTHER scenes (k-fold over scenes)."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    q = pp[pp.r_map > gate].assign(e=lambda d: d.h_map - d.top_km)
    scenes = q.scene.unique()
    folds = np.random.default_rng(seed).integers(0, k, scenes.size)
    fold_of = dict(zip(scenes, folds))
    resid = []
    for f in range(k):
        tr = q[q.scene.map(fold_of) != f]
        te = q[q.scene.map(fold_of) == f]
        # scene-level mean of scene-biases, not profile-pooled -- big
        # scenes must not dominate the constant
        b = tr.groupby("scene").e.mean().mean()
        resid.append(te.e - b)
    resid = pd.concat(resid)
    return dict(gate=gate, n=len(resid), k=k,
                rmse_cv_corrected=_rmse(resid),
                constant=float(q.groupby("scene").e.mean().mean()))


def bias_correct_cv_ci(profile_df, gate=0.6, n_boot=500, seed=0):
    pp = _load(profile_df).dropna(subset=["h_map"])
    pp = pp[pp.r_map > gate]
    sc = pp.scene.unique()
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(sc, sc.size)
        boot = pd.concat([pp[pp.scene == s] for s in pick])
        stats.append(bias_correct_cv(boot, gate=gate)["rmse_cv_corrected"])
    return (float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))