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
    return fig


def map_panel(extras, pr, rec, scene_idx, save=None):
    """Height map + quality map + along-track comparison."""
    glat, glon = extras["glat"], extras["glon"]
    Hq, rmax = extras["Hq"], extras["rmax"]
    if Hq is None:
        return None
    ext = [glon.min(), glon.max(), glat.min(), glat.max()]
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    for k, (Z, cm, vl, ttl) in enumerate(
            [(Hq, "viridis", (8, 14), "stereo height [km]"),
             (rmax, "magma", (0, 0.9), "peak local r")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(Z, origin="lower", cmap=cm, vmin=vl[0], vmax=vl[1],
                       extent=ext)
        plt.colorbar(im, ax=ax)
        ax.plot(pr.lon, pr.lat, "-", color="magenta", lw=1)
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    ax.plot(pr.lat, pr.top_km, ".", color="magenta", ms=5, label="CALIOP")
    if "h_map" in pr:
        ax.plot(pr.lat, pr.h_map, ".", color="C0", ms=5, label="stereo map")
    ax.set_ylim(8, 15); ax.set_xlabel("latitude along track")
    ax.set_ylabel("height [km]"); ax.legend(fontsize=8)
    ax.set_title(f"scene {scene_idx}: map_n={rec['map_n']}  "
                 f"map_bias={rec['map_bias']:+.2f}  qc={rec['qc']}",
                 fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130); plt.close(fig)
    return fig