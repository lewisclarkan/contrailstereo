"""Figures. Every function returns a matplotlib Figure (and saves it if
`save` is given); nothing here computes results -- it plots Results, Runs
and comparisons produced elsewhere.

Case      map_panel(result, truth)
Run       truth_scatter(run), gate_tradeoff(run), run_overview(run)
Compare   forest(table), case_scatter(paired), strata(table), flips(crosstab)

Maps use pcolormesh on the grid's own lat/lon, so they work on any grid
kind (not only regular lat/lon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import validation as V
from .types import QC_OK, Result

_CALL = {"better": "#2a9d4b", "worse": "#c0392b"}
_GREY = "0.45"


def _finish(fig, save):
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=200)
    return fig


def _mesh(ax, grid, Z, **kw):
    return ax.pcolormesh(grid.lon, grid.lat, Z, shading="auto",
                         rasterized=True, **kw)


def _geo_axes(ax, grid):
    ax.set_aspect(1.0 / np.cos(np.deg2rad(grid.bbox.center[0])))
    ax.set_xlim(grid.bbox.lon0, grid.bbox.lon1)
    ax.set_ylim(grid.bbox.lat0, grid.bbox.lat1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("lon [deg]")


# ======================================================================
# Case
# ======================================================================
FT_KM = 0.3048


def height_norm(h_lims, h_step=0.2, fl_bins=False, cmap="turbo"):
    """Discrete colour scale for heights: one colour per h_step km (or per
    flight level, 1000 ft, when fl_bins). Returns (cmap, norm, edges, ticks,
    tick labels)."""
    from matplotlib.colors import BoundaryNorm
    lo, hi = h_lims
    if fl_bins:
        fl0, fl1 = int(np.floor(lo / FT_KM)), int(np.ceil(hi / FT_KM))
        edges = (np.arange(fl0, fl1 + 1) - 0.5) * FT_KM        # bins centred on FLs
        centres = np.arange(fl0, fl1) * FT_KM
        step = max(1, int(round(len(centres) / 8)))
        ticks = centres[::step]
        labels = [f"FL{int(round(c / FT_KM * 10))}" for c in ticks]   # km -> hundreds of ft
    else:
        edges = np.arange(lo, hi + h_step / 2, h_step)
        step = max(1, int(round(len(edges) / 9)))
        ticks = edges[::step]
        labels = [f"{t:.1f}" for t in ticks]
    cm = plt.get_cmap(cmap, len(edges) + 1)          # + below/above-range bins
    return cm, BoundaryNorm(edges, cm.N, extend="both"), edges, ticks, labels


def map_panel(result: Result, truth: pd.DataFrame | None = None, btd=None,
              h_lims=(8.0, 14.0), title=None, save=None, h_step=0.2,
              fl_bins=False, h_cmap="turbo"):
    """Height map, peak correlation (and optional BTD) for one case, with
    the along-track comparison below when truth is given.

    truth   : attach_truth output (lat, lon, top_km, h, r) or any frame with
              lat, lon, top_km.
    btd     : optional 2-D field on result.grid (e.g. the parallax-corrected
              reference-view BTD) for context.
    h_step  : height colour bands [km] (default 200 m); None = continuous.
    fl_bins : one band per flight level (1000 ft), labelled FLxxx.
    """
    if result.height is None:
        raise ValueError(f"case {result.case_id}: no maps (qc={result.qc})")
    g = result.grid
    if h_step is None and not fl_bins:
        hkw, hcb = dict(cmap="viridis", vmin=h_lims[0], vmax=h_lims[1]), {}
        hlabel = "height [km]"
    else:
        cm, norm, edges, ticks, labels = height_norm(h_lims, h_step or FT_KM, fl_bins, h_cmap)
        hkw = dict(cmap=cm, norm=norm)
        hcb = dict(ticks=ticks, labels=labels)
        hlabel = ("height (1 band = 1 flight level)" if fl_bins
                  else f"height [km] ({(h_step or FT_KM) * 1e3:.0f} m bands)")
    panels = ([("BTD [K]", btd, dict(cmap="RdBu_r"))] if btd is not None
              else []) + [
        (hlabel, result.height, hkw),
        ("peak r", result.r, dict(cmap="magma", vmin=0, vmax=1))] + (
        [("s_eff [km/km]", result.s_eff, dict(cmap="cividis", vmin=0, vmax=2.5))]
        if result.s_eff is not None else [])
    nrow = 2 if truth is not None else 1
    fig = plt.figure(figsize=(4.4 * len(panels), 4.6 + 2.6 * (nrow - 1)))
    gs = fig.add_gridspec(nrow, len(panels),
                          height_ratios=[3, 1.3][:nrow])
    for k, (label, Z, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        if label.startswith("BTD"):
            v = np.nanpercentile(np.abs(Z), 99)
            kw = dict(kw, vmin=-v, vmax=v)
        m = _mesh(ax, g, Z, **kw)
        cb = fig.colorbar(m, ax=ax, shrink=0.8, label=label)
        if Z is result.height and hcb:
            cb.set_ticks(hcb["ticks"]); cb.set_ticklabels(hcb["labels"])
        if truth is not None:
            ax.plot(truth.lon, truth.lat, "w-", lw=2.2)
            ax.plot(truth.lon, truth.lat, "k-", lw=0.8)
        _geo_axes(ax, g)
        if k == 0:
            ax.set_ylabel("lat [deg]")
    if truth is not None:
        ax = fig.add_subplot(gs[1, :])
        t = truth.sort_values("lat")
        ax.plot(t.lat, t.top_km, "k.", ms=3, label="truth top")
        if "h" in t:
            ax.plot(t.lat, t.h, ".", color="C0", ms=4, label="stereo")
        ax.set_xlabel("lat [deg]")
        ax.set_ylabel("height [km]")
        ax.set_ylim(*h_lims)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    d = result.diag
    fig.suptitle(title or (
        f"{result.case_id}   qc={result.qc}   "
        f"VZA E/W {d.get('vza_east', np.nan):.0f}/{d.get('vza_west', np.nan):.0f}   "
        f"offset {d.get('offset_s', np.nan):+.1f} s   "
        f"coverage {d.get('map_coverage', np.nan):.2f}"), fontsize=9)
    fig.tight_layout()
    return _finish(fig, save)


# ======================================================================
# Run
# ======================================================================
def truth_scatter(run: V.Run, gate=0.6, qc=(QC_OK,), lims=(7.0, 16.0),
                  bias_correct=True, ax=None, save=None, s_min=0.0):
    """Retrieved vs truth height, 2-D histogram with 1:1 line and stats.
    bias_correct subtracts each case's leave-one-case-out constant (what
    rmse_corr measures); the stats box always shows both."""
    s = V.summary(run, gate, qc, s_min=s_min)
    p = run.profiles
    use = V._usable(run, qc)
    q = p[p.case_id.map(use).fillna(False).astype(bool)
          & V._passes(p, gate, s_min)].copy()
    q["e"] = q.h - q.top_km
    if bias_correct and len(q):
        means = q.groupby("case_id").e.mean()
        tot, cnt = means.sum(), len(means)
        q["h"] = q.h - q.case_id.map(lambda k: (tot - means[k]) / (cnt - 1)
                                     if cnt > 1 else 0.0)
    fig = ax.figure if ax is not None else plt.figure(figsize=(5.2, 4.8))
    ax = ax or fig.add_subplot(111)
    if len(q):
        h = ax.hist2d(q.top_km, q.h, bins=60, range=[lims, lims],
                      cmap="Blues", cmin=1)
        fig.colorbar(h[3], ax=ax, shrink=0.8, label="points")
    ax.plot(lims, lims, "k-", lw=0.8)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel("truth top [km]")
    ax.set_ylabel("stereo height" + (" (bias-corrected)" if bias_correct
                                     else "") + " [km]")
    ax.text(0.03, 0.97,
            f"n = {int(s['n'])} ({s['n_cases_scored']} cases)\n"
            f"yield {s['yield']:.2f}, coverage {s['coverage']:.2f}\n"
            f"bias {s['bias']:+.2f} km\n"
            f"RMSE raw {s['rmse_raw']:.2f}, corr {s['rmse_corr']:.2f} km\n"
            f"within {s['within']:.2f}, between {s['between']:.2f} km",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="w", ec="0.7", alpha=0.9))
    ax.set_title(f"{run.name}   r > {gate}" + (f", s_eff ≥ {s_min}" if s_min else ""),
                 fontsize=10)
    return _finish(fig, save)


def gate_tradeoff(run: V.Run, gates=np.arange(0.3, 0.86, 0.05), qc=(QC_OK,),
                  save=None, s_min=0.0):
    """Corrected RMSE (with within/between) and coverage against the r gate."""
    lad = V.gate_ladder(run, tuple(np.round(gates, 3)), qc, s_min=s_min)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(lad.index, lad.rmse_corr, "o-", color="k", label="rmse_corr")
    ax.plot(lad.index, lad.within, "--", color="C0", label="within")
    ax.plot(lad.index, lad.between, "--", color="C3", label="between")
    ax.set_xlabel("r gate")
    ax.set_ylabel("km")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(lad.index, lad.coverage, "s:", color="C2", label="coverage")
    ax2.set_ylabel("coverage", color="C2")
    ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title(run.name + (f"   s_eff ≥ {s_min}" if s_min else ""), fontsize=10)
    fig.tight_layout()
    return _finish(fig, save)


def run_overview(run: V.Run, save=None):
    """QC verdict counts, and map coverage against max VZA per case."""
    c = run.cases
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    vc = c.qc.value_counts()
    axs[0].bar(vc.index, vc.values,
               color=["C2" if k == QC_OK else "0.6" for k in vc.index])
    for i, v in enumerate(vc.values):
        axs[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    axs[0].set_ylabel("cases")
    axs[0].set_title("QC verdicts", fontsize=10)
    if {"vza_max", "map_coverage"} <= set(c.columns):
        for ok, col in ((True, "C2"), (False, "0.6")):
            d = c[(c.qc == QC_OK) == ok]
            axs[1].scatter(d.vza_max, d.map_coverage, s=12, color=col,
                           label=QC_OK if ok else "other")
        axs[1].set_xlabel("max VZA [deg]")
        axs[1].set_ylabel("map coverage")
        axs[1].legend(fontsize=8)
        axs[1].grid(alpha=0.3)
    else:
        axs[1].set_visible(False)
    fig.suptitle(f"{run.name}: {len(c)} cases", fontsize=10)
    fig.tight_layout()
    return _finish(fig, save)


# ======================================================================
# Comparison
# ======================================================================
def forest(table: pd.DataFrame, rows=None, title=None, save=None):
    """compare() deltas with CIs; fractions and km on separate panels.
    Coloured only where the CI excludes zero in a good/bad direction."""
    t = table
    rows = rows or [m for m in t.index if not m.startswith("bias")]
    t = t.loc[[m for m in rows if m in t.index]]
    frac = t.index.str.startswith(("yield", "coverage"))
    groups = [(g, u) for g, u in ((t[frac], "fraction"), (t[~frac], "km"))
              if len(g)]
    fig, axs = plt.subplots(
        len(groups), 1, figsize=(7.5, 0.36 * len(t) + 1.6), squeeze=False,
        gridspec_kw=dict(height_ratios=[len(g) + 0.6 for g, _ in groups]))
    for ax, (g, unit) in zip(axs[:, 0], groups):
        g = g[::-1]
        for y, (_, r) in enumerate(g.iterrows()):
            c = _CALL.get(r.call, _GREY)
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=2.2)
            ax.plot(r.delta, y, "o", color=c, ms=6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=9)
        ax.set_ylim(-0.6, len(g) - 0.4)
        ax.set_xlabel(f"{t.attrs.get('b', 'B')} − {t.attrs.get('a', 'A')} "
                      f"[{unit}]", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axs[0, 0].set_title(title or f"n = {t.attrs.get('n_cases')} cases, "
                        f"{t.attrs.get('ci', 95)}% paired case bootstrap",
                        fontsize=10)
    fig.tight_layout()
    return _finish(fig, save)


def case_scatter(P: V.Paired, color="vza_max", ax=None, save=None):
    """|case map offset| A vs B (common points). Below the diagonal =
    improved; marker size ~ common points; large movers labelled."""
    t = P.table.dropna(subset=["map_off_a", "map_off_b"])
    fig = ax.figure if ax is not None else plt.figure(figsize=(5.4, 5.0))
    ax = ax or fig.add_subplot(111)
    x, y = t.map_off_a.abs(), t.map_off_b.abs()
    lim = max(0.5, float(np.nanmax([x.max(), y.max()])) * 1.05) if len(t) else 1
    c = t[color] if color in t else None
    sc = ax.scatter(x, y, c=c, s=12 + 0.08 * t.n_common, cmap="viridis",
                    edgecolor="k", lw=0.3)
    if c is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8, label=color)
    ax.plot([0, lim], [0, lim], "k-", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"|case offset|  {P.a.name}  [km]")
    ax.set_ylabel(f"|case offset|  {P.b.name}  [km]")
    lab = "scene" if "scene" in t and t.scene.notna().all() else "case_id"
    for _, r in t[(x - y).abs() > 0.25 * lim].iterrows():
        ax.annotate(str(r[lab]), (abs(r.map_off_a), abs(r.map_off_b)),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.grid(alpha=0.3)
    ax.set_title("per-case offset, common points", fontsize=9)
    return _finish(fig, save)


def strata(table: pd.DataFrame, metric="rmse_corr (common)", ax=None,
           save=None):
    """by_stratum() deltas with CIs for one metric, annotated with n."""
    s = table[table.metric == metric].reset_index(drop=True)
    fig = ax.figure if ax is not None else plt.figure(figsize=(6.0, 3.6))
    ax = ax or fig.add_subplot(111)
    for i, r in s.iterrows():
        c = _CALL.get(r.call, _GREY)
        ax.plot([i, i], [r.lo, r.hi], color=c, lw=2.2)
        ax.plot(i, r.delta, "o", color=c, ms=6)
        ax.annotate(f"n={r.n_cases}", (i, r.hi), ha="center", xytext=(0, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.stratum)
    ax.set_ylabel("Δ (B − A)")
    ax.set_title(metric, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, save)


def flips(ct: pd.DataFrame, save=None):
    """verdict_flips() crosstab as an annotated heatmap (changes in red)."""
    fig, ax = plt.subplots(figsize=(0.7 * ct.shape[1] + 2.5,
                                    0.55 * ct.shape[0] + 1.8))
    v = ct.values.astype(float)
    ax.imshow(np.log1p(v), cmap="Blues")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j]:
                changed = ct.index[i] != ct.columns[j]
                ax.text(j, i, int(v[i, j]), ha="center", va="center",
                        fontsize=9, color="#c0392b" if changed else "k",
                        fontweight="bold" if changed else "normal")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_xlabel(ct.columns.name)
    ax.set_ylabel(ct.index.name)
    ax.set_title("QC verdicts (red = changed)", fontsize=10)
    fig.tight_layout()
    return _finish(fig, save)