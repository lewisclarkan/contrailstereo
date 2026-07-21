"""Run-level figures from the output CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load(x):
    return pd.read_csv(x) if isinstance(x, str) else x


def run_summary(scene_df, save=None):
    """Scatter vs CALIOP + error histogram + verdict bars."""
    df = _load(scene_df)
    good = df[df.qc == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    lims = [6, 15]
    if len(good):
        axes[0].errorbar(good.caliop_top_km, good.h_local,
                         yerr=good.sigma_h, fmt="o", ms=4, lw=0.8, alpha=0.85)
    axes[0].plot(lims, lims, "k--", lw=0.8)
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("CALIOP top [km]"); axes[0].set_ylabel("stereo [km]")
    if len(good):
        axes[1].hist(good.err_km, bins=np.arange(-3, 3.01, 0.2), color="C0")
        axes[1].axvline(0, color="k", lw=0.8)
        axes[1].axvline(good.err_km.median(), color="r", ls="--",
                        label=f"median {good.err_km.median():+.2f} km")
        axes[1].legend()
    axes[1].set_xlabel("stereo - CALIOP [km] (qc=ok)")
    df.qc.value_counts().plot.bar(ax=axes[2], color="0.6")
    axes[2].set_ylabel("scenes"); axes[2].set_title("QC verdicts")
    fig.suptitle(f"{len(good)}/{len(df)} retrievals")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return


def map_scatter(profile_df, gate=0.6, bias_correct=True, save=None):
    """Per-profile stereo vs CALIOP (the Meijer-Fig-4-style panel)."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    q = pp[pp.r_map > gate]
    y, yh = q.top_km.values, q.h_map.values
    pred = yh - (yh - y).mean() if bias_correct else yh
    e = pred - y
    rmse = np.sqrt(np.mean(e ** 2))
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [8, 15]
    ax.scatter(y, pred, s=6, alpha=0.25, edgecolors="none")
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("CALIOP top [km]")
    ax.set_ylabel("stereo map height"
                  + (", bias-corrected" if bias_correct else "") + " [km]")
    ax.text(0.04, 0.96, f"RMSE: {rmse:.2f} km\n$R^2$: {r2:.2f}\nn = {len(q)}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", alpha=0.85, ec="0.7"))
    ax.set_title(f"r$_{{map}}$ > {gate}  ({len(q)/len(pp):.0%} of profiles)",
                 fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return


def tradeoff_plot(profile_df, gates=np.arange(0.3, 0.86, 0.05), save=None):
    """The quality/coverage curve as a figure."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    xs, rs, cov = [], [], []
    for thr in gates:
        q = pp[pp.r_map > thr]
        e = (q.h_map - q.top_km).values
        if e.size < 30:
            continue
        xs.append(thr)
        rs.append(np.sqrt(((e - e.mean()) ** 2).mean()))
        cov.append(e.size / len(pp))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, rs, "o-", color="C0")
    ax.set_xlabel("r$_{map}$ gate"); ax.set_ylabel("bias-corr RMSE [km]",
                                                   color="C0")
    ax2 = ax.twinx()
    ax2.plot(xs, cov, "s--", color="C1")
    ax2.set_ylabel("profile coverage", color="C1")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150); plt.close(fig)
    return