"""contrailstereo: geostationary stereo heights for contrails and thin cirrus.

    contrailstereo run      [--config F] [--set k=v ...] [--split dev] [--n N]
    contrailstereo run      --time T --bbox LAT0 LAT1 LON0 LON1   (no truth)
    contrailstereo compare  A B [--set-a k=v ...] [--set-b k=v ...] [--log]
    contrailstereo config   [--config F] [--set k=v ...]

A and B are "default" or a config file (JSON/TOML); --set-a/--set-b apply
overrides on top. Runs are cached by config hash under <outputs>/runs, so
`compare` reuses anything already computed and `run` resumes after an
interruption.

Examples
    contrailstereo run --split dev
    contrailstereo compare default default --set-b prep=psf_iso --log \\
        --note "PSF iso vs hp"
    contrailstereo run --time "2019-12-01 21:12" --bbox 32.8 34.8 -116 -114
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import validation as V
from .config import DEFAULT, StereoConfig, diff, load_paths, parse_overrides
from .retrieve import load_frames
from .types import QC_OK, Case

FRAME_LOADER = load_frames        # module-level so tests can substitute it


# ======================================================================
# helpers
# ======================================================================
def _config(spec, overrides) -> StereoConfig:
    base = DEFAULT if spec in (None, "default") else StereoConfig.from_file(spec)
    return base.set(**parse_overrides(overrides or [], base)) if overrides else base


def _name(spec, overrides):
    """Readable variant name: 'base', the config file's stem, and/or the
    overrides (e.g. 'win_km-21'). The config hash is appended to file names
    anyway, so this only needs to be recognisable."""
    stem = None if spec in (None, "default") else Path(spec).stem
    if not overrides:
        return stem or "base"
    ov = "_".join(o.replace("=", "-").replace(",", "-").replace("/", "-")
                  for o in overrides)[:48]
    return f"{stem}+{ov}" if stem else ov


def _cases(args, paths, sats=(DEFAULT.sat_east, DEFAULT.sat_west)):
    from .data.caliop import load_cases
    nc = args.collocations or paths.collocations
    if nc is None:
        sys.exit("no collocations file: pass --collocations or set it in "
                 "[paths] / CONTRAILSTEREO_COLLOCATIONS")
    return V.select(load_cases(nc), args.split, n=args.n, max_vza=args.max_vza,
                    sats=tuple(sats))


def _qc(s):
    return None if s == "all" else tuple(x.strip() for x in s.split(","))


def _print(df, fmt="{:.3f}"):
    with pd.option_context("display.float_format", fmt.format,
                           "display.width", 160, "display.max_columns", 30):
        print(df.to_string())


def _add_common(p):
    p.add_argument("--collocations", help="collocation netCDF "
                   "(default: from paths)")
    p.add_argument("--split", default="dev", choices=("dev", "holdout", "all"))
    p.add_argument("--n", type=int, default=None, help="first N cases only")
    p.add_argument("--max-vza", type=float, default=None,
                   help="keep cases seen by both satellites at VZA <= this [deg]")
    p.add_argument("--gate", type=float, default=0.6, help="r gate for metrics")
    p.add_argument("--s-min", type=float, default=0.0,
                   help="s_eff gate for metrics [km/km] (e.g. 1.0)")
    p.add_argument("--qc", default=QC_OK,
                   help="verdicts that count, comma-separated, or 'all'")
    p.add_argument("--out", default=None, help="run directory "
                   "(default: <outputs>/runs)")
    p.add_argument("--index", default="nearest", choices=("nearest", "legacy"),
                   help="truth sampling rule (legacy reproduces v4)")
    p.add_argument("--quiet", action="store_true")


# ======================================================================
# commands
# ======================================================================
def cmd_config(args):
    cfg = _config(args.config, args.set)
    print(f"config hash: {cfg.config_hash()}")
    d = diff(DEFAULT, cfg)
    print("differs from default:" if d else "identical to default")
    for k, a, b in d:
        print(f"  {k}: {a} -> {b}")
    if args.show:
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
    if args.write:
        print(f"written to {cfg.to_file(args.write)}")
    p = load_paths()
    print("\npaths:")
    for k in ("goes_cache", "era5_cache", "outputs", "collocations"):
        print(f"  {k:13s} {getattr(p, k)}")
    return 0


def cmd_run(args):
    paths = load_paths()
    cfg = _config(args.config, args.set)
    v = V.Variant(args.name or _name(args.config, args.set), cfg)
    out = Path(args.out) if args.out else paths.outputs / "runs"

    if args.time or args.bbox:                       # ad-hoc, truth-free
        if not (args.time and args.bbox):
            sys.exit("--time and --bbox go together")
        t = pd.Timestamp(args.time)
        case = Case(f"adhoc_{t:%Y%m%dT%H%M%S}", t, tuple(args.bbox))
        runs = V.run([case], [v], out, paths=paths, validate=False,
                     save_maps=True, frame_loader=FRAME_LOADER,
                     verbose=not args.quiet)
        c = runs[v.name].cases.iloc[-1]
        print(f"\n{case.id}: qc={c.qc}  coverage={c.get('map_coverage')}  "
              f"h_median={c.get('h_median')}")
        print(f"map: {out / (v.tag + '_maps')}")
        return 0 if c.qc != "error" else 1

    cases = _cases(args, paths, (cfg.sat_east, cfg.sat_west))
    runs = V.run(cases, [v], out, paths=paths, validate=not args.no_validate,
                 save_maps=args.maps, resume=not args.no_resume,
                 index=args.index, frame_loader=FRAME_LOADER,
                 verbose=not args.quiet)
    r = runs[v.name]
    r = V.Run(r.variant, r.cases[r.cases.case_id.isin([c.id for c in cases])],
              r.profiles)
    print(f"\n{v.tag}: {len(r.cases)} cases ({args.split})")
    print(r.cases.qc.value_counts().to_string())
    if r.profiles is not None and len(r.profiles):
        qc = _qc(args.qc)
        s = V.summary(r, args.gate, qc, n_boot=1000, s_min=args.s_min)
        print(f"\nr > {args.gate}, s_eff >= {args.s_min}, qc {args.qc}: n={int(s['n'])} "
              f"({s['n_cases_scored']} cases)  coverage {s['coverage']:.3f}  "
              f"bias {s['bias']:+.3f}  rmse_raw {s['rmse_raw']:.3f}  "
              f"rmse_corr {s['rmse_corr']:.3f} "
              f"[{s['rmse_corr_ci'][0]:.3f}, {s['rmse_corr_ci'][1]:.3f}]  "
              f"within {s['within']:.3f}  between {s['between']:.3f}")
        print()
        _print(V.gate_ladder(r, (0.5, 0.6, 0.7), qc, s_min=args.s_min)[
            ["n", "n_cases_scored", "coverage", "bias", "rmse_corr",
             "within", "between"]])
        c = V.contrail_summary(r, args.gate, qc, n_boot=1000, s_min=args.s_min)
        if "rmse_corr" in c:
            ci = lambda k: f"[{c[k + '_ci'][0]:.3f}, {c[k + '_ci'][1]:.3f}]"
            print(f"\nper contrail: {c['n_scored']}/{c['n_contrails']} scored "
                  f"(coverage {c['contrail_coverage']:.2f})  bias {c['bias']:+.3f} {ci('bias')}  "
                  f"rmse_raw {c['rmse_raw']:.3f} {ci('rmse_raw')}  "
                  f"rmse_corr {c['rmse_corr']:.3f} {ci('rmse_corr')}")
    return 0


def cmd_compare(args):
    paths = load_paths()
    ca, cb = _config(args.a, args.set_a), _config(args.b, args.set_b)
    if ca.config_hash() == cb.config_hash():
        sys.exit("A and B are the same configuration")
    va = V.Variant(args.name_a or _name(args.a, args.set_a), ca)
    vb = V.Variant(args.name_b or _name(args.b, args.set_b), cb)
    if va.name == vb.name:
        vb = V.Variant(vb.name + "_b", cb)
    out = Path(args.out) if args.out else paths.outputs / "runs"

    print(f"A = {va.tag}\nB = {vb.tag}")
    for k, a, b in diff(ca, cb):
        print(f"  {k}: {a} -> {b}")
    cases = _cases(args, paths, {ca.sat_east, ca.sat_west, cb.sat_east,
                                 cb.sat_west})
    runs = V.run(cases, [va, vb], out, paths=paths, index=args.index,
                 frame_loader=FRAME_LOADER, verbose=not args.quiet)
    ids = [c.id for c in cases]
    P = V.pair(runs[va.name], runs[vb.name], args.gate, _qc(args.qc), ids=ids,
               s_min=args.s_min)
    S = V.compare(P, n_boot=args.n_boot)
    a = S.attrs
    print(f"\n{a['n_cases']} common cases, {a['n_common_points']} common "
          f"points (r > {args.gate}, qc {args.qc}); errors {a['n_error']}")
    _print(S[["A", "B", "delta", "lo", "hi", "call"]])
    print("\nverdicts (rows A, columns B):")
    print(V.verdict_flips(P).to_string())
    if "vza_max" in P.table:
        print("\nby max VZA:")
        _print(V.by_stratum(P, n_boot=args.n_boot)[
            ["stratum", "metric", "n_cases", "delta", "lo", "hi", "call"]])
    t = V.tail(P)
    if len(t):
        print(f"\ncases offset > 1 km in either run:")
        _print(t.drop(columns="case_id", errors="ignore"))
    if args.log:
        p = V.log_summary(S, paths.outputs / "experiments" / "log.csv",
                          note=args.note or "", split=args.split)
        print(f"\nlogged to {p}")
    return 0


# ======================================================================
# entry point
# ======================================================================
def build_parser():
    ap = argparse.ArgumentParser(prog="contrailstereo",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("config", help="show a resolved config, hash and paths")
    c.add_argument("--config")
    c.add_argument("--set", nargs="*", default=[], metavar="K=V")
    c.add_argument("--show", action="store_true", help="print every field")
    c.add_argument("--write", metavar="FILE", help="save the config as JSON")
    c.set_defaults(func=cmd_config)

    r = sub.add_parser("run", help="run one configuration")
    r.add_argument("--config")
    r.add_argument("--set", nargs="*", default=[], metavar="K=V")
    r.add_argument("--name")
    r.add_argument("--no-validate", action="store_true")
    r.add_argument("--maps", action="store_true", help="save per-case maps")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--time", help="ad-hoc case time (UTC), with --bbox")
    r.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LAT0", "LAT1", "LON0", "LON1"))
    _add_common(r)
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("compare", help="A/B comparison of two configurations")
    m.add_argument("a", help='"default" or a config file')
    m.add_argument("b", help='"default" or a config file')
    m.add_argument("--set-a", nargs="*", default=[], metavar="K=V")
    m.add_argument("--set-b", nargs="*", default=[], metavar="K=V")
    m.add_argument("--name-a")
    m.add_argument("--name-b")
    m.add_argument("--n-boot", type=int, default=2000)
    m.add_argument("--log", action="store_true",
                   help="append the table to <outputs>/experiments/log.csv")
    m.add_argument("--note", help="one line for the log")
    _add_common(m)
    m.set_defaults(func=cmd_compare)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())