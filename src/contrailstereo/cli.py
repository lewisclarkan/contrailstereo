"""Batch runner

Usage:
    contrailstereo run collocatinons.nc --n 150 --out results.csv --maps
    
"""

from __future__ import annotations

import argparse
import os
import traceback

import numpy as np
import pandas as pd

from .config import DEFAULT, StereoConfig
from .data.caliop import build_scene_table
from .core.retrieve import load_scene, run_scene, SCENE_SCHEMA


def _print_row(i, n_total, row, rec):
    h = rec["h_local"]
    mb = rec["map_bias"]
    print(f"[{i + 1:3d}/{n_total}] {row.time:%Y-%m-%d %H:%M}  "
          f"h={'  nan' if not np.isfinite(h) else f'{h:5.2f}'}  "
          f"qc={rec['qc']:8s}  map_n={rec['map_n']:3d}  "
          f"map_bias={'  nan' if not np.isfinite(mb) else f'{mb:+.2f}'}")


def _summary(out, pp, cfg):
    print(f"\n{len(out)} scenes  (config {cfg.config_hash()})")
    print(out.qc.value_counts().to_string())
    good = out[out.qc == "ok"]
    if len(good):
        e = good.err_km
        print(f"\nSCENE mode: yield {len(good)}/{len(out)}  "
              f"bias {e.mean():+.3f}  "
              f"bias-corr RMSE {np.sqrt(((e - e.mean()) ** 2).mean()):.3f}")
    if pp is not None and len(pp):
        p = pp.dropna(subset=["h_map"])
        print(f"\nMAP mode ({len(p)} mapped profiles):")
        for thr in cfg.map_gates:
            q = p[p.r_map > thr]
            if len(q) < 5:
                continue
            e = (q.h_map - q.top_km).values
            print(f"  r_map>{thr:.1f}: n={len(e)} ({len(e) / len(p):.0%})  "
                  f"raw {np.sqrt((e ** 2).mean()):.3f}  "
                  f"bias-corr {np.sqrt(((e - e.mean()) ** 2).mean()):.3f}  "
                  f"bias {e.mean():+.3f}")
    wa = out.wind_applied.sum() if "wind_applied" in out else 0
    print(f"\nwind correction applied: {wa}/{len(out)}")   # silent-failure tell


def cmd_run(args):
    cfg = DEFAULT
    scenes, prof_df = build_scene_table(args.nc)
    scenes = scenes.head(args.n)
    n_total = len(scenes)

    # ---- resume ---------------------------------------------------------
    rows, pp_frames, done = [], [], set()
    if os.path.exists(args.out) and not args.no_resume:
        prev = pd.read_csv(args.out)
        prior = set(prev.config_hash.dropna().unique())
        if prior - {cfg.config_hash()}:
            raise SystemExit(
                f"ERROR: {args.out} was produced with config "
                f"{prior} != current {cfg.config_hash()}.\n"
                f"Use --no-resume (fresh file) or restore the config.")
        rows = prev.to_dict("records")
        done = set(prev.scene.astype(int))
        if os.path.exists(args.profiles_out):
            old = pd.read_csv(args.profiles_out)
            pp_frames = [old[old.scene.isin(done)]]     # only completed scenes
        print(f"resuming: {len(done)} scenes already in {args.out}")

    if args.maps:
        os.makedirs(args.maps_dir, exist_ok=True)

    for i, row in scenes.iterrows():
        if i in done:
            continue
        try:
            sd = load_scene(row, prof_df, cfg)
            rec, pr, grids = run_scene(sd, cfg, do_map=not args.no_map)
        except Exception as e:                          # noqa: BLE001
            print(f"[{i + 1:3d}/{n_total}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            continue

        rec.update(scene=i, time=row.time, lat=row.lat, lon=row.lon,
                   n_prof=int(row.n), caliop_top_km=float(row.top_km),
                   caliop_top_sd=float(row.top_sd),
                   goes_file=row.goes_file,
                   err_km=(rec["h_local"] - row.top_km)
                          if rec["qc"] == "ok" else np.nan)
        rows.append(rec)
        _print_row(i, n_total, row, rec)

        # ---- checkpoint (both files, full rewrite) ----------------------
        pd.DataFrame(rows).to_csv(args.out, index=False)
        if pr is not None:
            p = pr[["lat", "lon", "top_km", "h_map", "r_map"]].copy()
            p["scene"] = i
            pp_frames.append(p)
            pd.concat(pp_frames, ignore_index=True).to_csv(
                args.profiles_out, index=False)

        if args.maps and grids is not None:
            from .viz.scene import map_panel            # lazy: matplotlib
            map_panel(grids, pr, rec, i,
                      save=os.path.join(args.maps_dir, f"scene_{i:04d}.png"))

    if not rows:
        print("no scenes completed")
        return
    out = pd.DataFrame(rows)
    pp = (pd.concat(pp_frames, ignore_index=True)
          if pp_frames else None)
    _summary(out, pp, cfg)


def main():
    ap = argparse.ArgumentParser(prog="contrailstereo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="batch retrieval over the scene table")
    r.add_argument("nc", help="collocations netCDF")
    r.add_argument("--n", type=int, default=150)
    r.add_argument("--out", default="outputs/results.csv")
    r.add_argument("--profiles-out", default="outputs/profiles.csv")
    r.add_argument("--maps", action="store_true",
                   help="save per-scene map figures (slow)")
    r.add_argument("--maps-dir", default="outputs/maps")
    r.add_argument("--no-map", action="store_true",
                   help="skip the map cube entirely (scene scalars only)")
    r.add_argument("--no-resume", action="store_true")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()