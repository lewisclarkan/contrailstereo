import numpy as np
import pandas as pd
from contrailstereo.validation import metrics as M


def _synthetic_profiles(n_scenes=40, n_per=60, within=0.4, between=0.5,
                        seed=0):
    """Profile frame with KNOWN error structure: per-scene offsets ~
    N(-0.5, between), within-scene noise ~ N(0, within)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        off = -0.5 + between * rng.normal()
        top = 11.0 + rng.normal(0, 1.0)
        for _ in range(n_per):
            rows.append(dict(scene=s, top_km=top,
                             h_map=top + off + within * rng.normal(),
                             r_map=0.8, lat=40.0, lon=-105.0))
    return pd.DataFrame(rows)


def test_decomposition_recovers_components():
    pp = _synthetic_profiles(within=0.4, between=0.5)
    d = M.decomposition(pp, gate=0.6)
    assert abs(d["within"] - 0.4) < 0.03
    assert abs(d["between"] - 0.5) < 0.06


def test_cv_bias_correction_recovers_constant():
    pp = _synthetic_profiles(between=0.15)
    out = M.bias_correct_cv(pp)
    assert abs(out["constant"] - (-0.5)) < 0.06
    # residual should approach sqrt(within^2 + between^2)
    assert abs(out["rmse_cv_corrected"] - np.hypot(0.4, 0.15)) < 0.05


def test_meijer_metric_r2_consistency():
    pp = _synthetic_profiles()
    m = M.meijer_metric(pp)
    # R^2 must equal 1 - rmse^2/var(truth) by construction
    implied = 1 - m["rmse_raw"] ** 2 / m["truth_sd"] ** 2
    assert abs(m["r2_raw"] - implied) < 1e-9


def test_loo_matches_large_n_limit():
    e = np.random.default_rng(1).normal(-0.5, 0.4, 4000)
    assert abs(M.loo_corrected_rmse(e) - 0.4) < 0.02


def test_map_tradeoff_reports_scene_counts():
    pp = _synthetic_profiles(n_scenes=6)
    t = M.map_tradeoff(pp)
    assert "n_scenes" in t.columns and (t.n_scenes == 6).all()


def test_scene_summary_yield_decomposition():
    df = pd.DataFrame(dict(
        qc=["ok"] * 6 + ["coverage"] * 2 + ["low_r", "no_data"],
        err_km=list(np.random.default_rng(2).normal(-0.5, 0.3, 6))
               + [np.nan] * 4,
        h_local=[10.0] * 6 + [np.nan] * 4))
    s = M.scene_summary(df)
    assert s["n_data_invalid"] == 3          # coverage x2 + no_data
    assert abs(s["yield_of_valid"] - 6 / 7) < 1e-9