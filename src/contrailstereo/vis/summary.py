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


from matplotlib.colors import LogNorm

def map_scatter(profile_df, gate=0.6, bias_correct=True, ax=None,
                base_fs=13, rmse_label=None, lims=(8, 15), save=None):
    """Per-profile stereo vs CALIOP density scatter (one panel).

    rmse_label : float, optional -- externally computed RMSE (e.g. the
    cross-validated value from metrics.bias_correct_cv) to display
    instead of the in-sample value computed here. The thesis caption
    promises CV'd numbers, so pass it for the corrected panel.
    """
    pp = _load(profile_df).dropna(subset=["h_map"])
    q = pp[pp.r_map > gate]
    y, yh = q.top_km.values, q.h_map.values
    pred = yh - (yh - y).mean() if bias_correct else yh
    e = pred - y
    rmse = rmse_label if rmse_label is not None else np.sqrt(np.mean(e**2))
    r2 = 1 - np.sum(e**2) / np.sum((y - y.mean())**2)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
    hb = ax.hexbin(y, pred, gridsize=55, cmap="Blues", norm=LogNorm(),
                   mincnt=1, extent=(*lims, *lims), linewidths=0.1)
    ax.plot(lims, lims, "r--", lw=1.2)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    ax.set_xlabel("CALIOP cloud-top altitude [km]", fontsize=base_fs)
    ax.set_ylabel("Stereo height"
                  + (" (bias-corrected)" if bias_correct else "")
                  + " [km]", fontsize=base_fs)
    ax.tick_params(labelsize=base_fs - 2)
    ax.text(0.04, 0.97,
            f"RMSE = {rmse:.3f} km\n$R^2$ = {r2:.2f}\n$n$ = {len(q):,}",
            transform=ax.transAxes, va="top", fontsize=base_fs - 1,
            bbox=dict(fc="white", alpha=0.9, ec="0.6", pad=4))
    if standalone:
        plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.03,
                     label="profiles per bin")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=300); plt.close(fig)
        return fig
    return hb


def map_scatter_pair(profile_df, gate=0.6, base_fs=13,
                     rmse_corr=None, save=None):
    """(a) raw and (b) bias-corrected panels with shared density scale.
    rmse_corr: pass metrics.bias_correct_cv(...)['rmse_cv_corrected'] so
    panel (b) quotes the cross-validated value."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    for ax, bc, lab, rl in [(axes[0], False, "(a)", None),
                            (axes[1], True, "(b)", rmse_corr)]:
        hb = map_scatter(profile_df, gate=gate, bias_correct=bc, ax=ax,
                         base_fs=base_fs, rmse_label=rl)
        ax.text(0.95, 0.045, lab, transform=ax.transAxes,
                        fontsize=base_fs + 1, fontweight="bold",
                        ha="right", va="bottom")
    axes[1].set_ylabel("Stereo height (bias-corrected) [km]",
                       fontsize=base_fs)
    cb = fig.colorbar(hb, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Profiles per bin", fontsize=base_fs)
    cb.ax.tick_params(labelsize=base_fs - 2)
    if save:
        fig.savefig(save, dpi=300); plt.close(fig)
    return fig


from contrailstereo.validation.metrics import bias_correct_cv

def tradeoff_plot(profile_df, gates=np.arange(0.3, 0.86, 0.05),
                  report_gate=0.6, cv=True, base_fs=13, save=None):
    """Quality--coverage tradeoff. cv=True computes the bias correction
    by scene-blocked cross-validation at every gate (the quotable
    version, per §2.2.3); cv=False falls back to in-sample."""
    pp = _load(profile_df).dropna(subset=["h_map"])
    xs, rs, cov = [], [], []
    for thr in gates:
        q = pp[pp.r_map > thr]
        if len(q) < 30 or q.scene.nunique() < 10:
            continue
        if cv:
            rmse = bias_correct_cv(q.assign(r_map=1.0), gate=0.0
                                   )["rmse_cv_corrected"]
        else:
            e = (q.h_map - q.top_km).values
            rmse = np.sqrt(((e - e.mean()) ** 2).mean())
        xs.append(thr); rs.append(rmse); cov.append(len(q) / len(pp))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    c1, c2 = "#1f77b4", "#d95f02"
    print(xs, rs, cov)
    ax.plot(xs, rs, "o-", color=c1, lw=1.8, ms=6, zorder=3)
    ax.set_xlabel("Correlation gate", fontsize=base_fs)
    ax.set_ylabel("Bias-corrected RMSE [km]", fontsize=base_fs, color=c1)
    ax.tick_params(axis="y", labelcolor=c1, labelsize=base_fs - 2)
    ax.tick_params(axis="x", labelsize=base_fs - 2)

    ax2 = ax.twinx()
    ax2.plot(xs, cov, "s--", color=c2, lw=1.8, ms=6, zorder=3)
    ax2.set_ylabel("Profile coverage", fontsize=base_fs, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2, labelsize=base_fs - 2)
    ax2.set_ylim(0, 1.02)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.0%}"))

    ax.axvline(report_gate, color="0.35", ls=":", lw=1.4)
    #ax.annotate("reporting gate", xy=(report_gate, ax.get_ylim()[1]),
    #            xytext=(4, -4), textcoords="offset points",
    #            fontsize=base_fs - 2, color="0.35", va="top")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300); plt.close(fig)
    return fig