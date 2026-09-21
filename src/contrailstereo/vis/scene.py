"""Per-scene figures"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import DEFAULT


def curve_plot(extras, rec, truth_km=None, save=None):
    """Correlation-vs-height: the method's native diagnostic."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(extras["hs"], extras["curve"], color="C0", label="window r")
    if truth_km is not None:
        ax.axvline(truth_km, color="r", ls="--", label="CALIOP top")
    if np.isfinite(rec["h_local"]):
        c = "C0" if rec["qc"] == "ok" else "C3"
        ax.axvline(rec["h_local"], color=c, ls=":")
        if np.isfinite(rec["sigma_h"]):
            ax.axvspan(rec["h_local"] - rec["sigma_h"],
                       rec["h_local"] + rec["sigma_h"], color=c, alpha=0.15)
    ax.axvline(DEFAULT.hmin_cirrus_km, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("assumed height [km]"); ax.set_ylabel("E-W correlation")
    ax.set_title(f"qc={rec['qc']}  offset={rec['offset_s']:+.0f}s",
                 fontsize=9)
    ax.legend(fontsize=7); fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return


def map_panel(extras, pr, rec, scene_idx, btd=None, save=None,
              diagnostic_title=True, base_fs=13, h_lims=None,
              panel_labels=True):
    """Publication-grade: BTD + height + quality maps over the along-track
    comparison. h_lims=None -> robust adaptive stretch on the height map."""
    glat, glon = extras["glat"], extras["glon"]
    Hq, rmax = extras["Hq"], extras["rmax"]
    if Hq is None:
        return None
    ext = [glon.min(), glon.max(), glat.min(), glat.max()]
    prs = pr.sort_values("lat")
    south, north = prs.iloc[0], prs.iloc[-1]

    if h_lims is None:                       # adaptive: show the structure
        lo = np.nanpercentile(Hq, 5)
        hi = np.nanpercentile(Hq, 95)
        pad = 0.15 * (hi - lo + 1e-9)
        h_lims = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.18)
    map_axes = []

    panels = [(btd, "RdBu_r", None, "BTD [K]"),
              (Hq, "viridis", h_lims, "Stereo height [km]"),
              (rmax, "magma", (0, 0.9), "Peak correlation")]
    for k, (Z, cm, vl, cblabel) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        map_axes.append(ax)
        if Z is not None:
            if vl is None:
                vl = (np.nanpercentile(Z, 2), np.nanpercentile(Z, 98))
            im = ax.imshow(Z, origin="lower", cmap=cm,
                           vmin=vl[0], vmax=vl[1], extent=ext)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(cblabel, fontsize=base_fs)
            cb.ax.tick_params(labelsize=base_fs - 2)
        ax.plot(pr.lon, pr.lat, "-", color="magenta", lw=1.2)
        ax.plot(south.lon, south.lat, "v", color="lime", ms=10, mec="k",
                mew=0.9, zorder=5)
        ax.plot(north.lon, north.lat, "^", color="orange", ms=10, mec="k",
                mew=0.9, zorder=5)
        if panel_labels:
            ax.text(0.03, 0.97, f"({'abc'[k]})", transform=ax.transAxes,
                    fontsize=base_fs + 1, fontweight="bold", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :])
    ax.plot(pr.lat, pr.top_km, ".", color="magenta", ms=6,
            label="CALIOP cloud top")
    if "h_map" in pr:
        ax.plot(pr.lat, pr.h_map, ".", color="C0", ms=6,
                label="Stereo height")
    for pt, mk, c in ((south, "v", "lime"), (north, "^", "orange")):
        ax.axvline(pt.lat, color=c, lw=0.9, alpha=0.5)
        ax.plot(pt.lat, 8.15, mk, color=c, ms=10, mec="k", mew=0.9,
                clip_on=False, zorder=5)
    ax.set_ylim(8, 15)
    ax.set_xlabel("Latitude along track [\u00b0N]   "
                  "(\u25bc southern end, \u25b2 northern end)",
                  fontsize=base_fs)
    ax.set_ylabel("Height [km]", fontsize=base_fs)
    ax.tick_params(labelsize=base_fs - 2)
    ax.legend(fontsize=base_fs, loc="lower center", framealpha=0.9)
    if panel_labels:
        ax.text(0.012, 0.95, "(d)", transform=ax.transAxes,
                fontsize=base_fs + 1, fontweight="bold", va="top")
    if diagnostic_title:
        ax.set_title(f"scene {scene_idx}: map_n={rec['map_n']}  "
                     f"map_bias={rec['map_bias']:+.2f}  qc={rec['qc']}",
                     fontsize=base_fs - 3)

    fig.tight_layout()
    x0 = min(a.get_position().x0 for a in map_axes)
    x1 = max(a.get_position().x1 for a in map_axes)
    b = ax.get_position()
    ax.set_position([x0, b.y0, x1 - x0, b.height])

    if save:
        fig.savefig(save, dpi=300, bbox_inches=None)
        plt.close(fig)
    return fig