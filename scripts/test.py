"""Check the restructured retrieval reproduces the v4 headline numbers.

Runs the default config over the same scenes as v4 (the first N_SCENES of
the collocation scene table), with the old profile-sampling rule and every
completed scene counted, as v4 did. Then prints the new numbers next to
the v4 ones from analysis.ipynb.

Usage:
    python scripts/check_v4.py [collocations.nc] [--n 248]

Resumable: interrupt and rerun freely. Outputs go to <outputs>/v4check/.

Which rows must match
---------------------
n_points, n_cases_scored, bias, rmse_raw and within have identical
definitions in v4 and now; they should agree to rounding (+-0.005 km).

rmse_corr is not compared: v4 quoted in-sample (0.584) and 5-fold CV
(0.585) corrections, the new package uses leave-one-case-out, so a small
difference is expected. Coverage is also defined differently (v4: share of
MAPPED points; now: share of ALL truth points).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from contrailstereo.config import DEFAULT, load_paths
from contrailstereo.data.caliop import load_cases
from contrailstereo import validation as V

# v4 map mode, channels 13/15, from analysis.ipynb (map_tradeoff, decomposition)
V4 = pd.DataFrame(
    {0.5: dict(n_points=11793, n_cases_scored=186, bias=-0.290422,
               rmse_raw=0.679574, within=np.nan),
     0.6: dict(n_points=10932, n_cases_scored=183, bias=-0.300549,
               rmse_raw=0.656401, within=0.430042),
     0.7: dict(n_points=9676, n_cases_scored=166, bias=-0.293966,
               rmse_raw=0.636428, within=np.nan)}).T

TOL_KM = 0.005


def headline(run, gates=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    """New numbers in the v4 layout, with v4 alongside and a match flag."""
    rows = []
    for g in gates:
        s = V.summary(run, gate=g, qc=None)          # v4 counted every scene
        new = dict(n_points=s["n"], n_cases_scored=s["n_cases_scored"],
                   bias=s["bias"], rmse_raw=s["rmse_raw"], within=s["within"])
        for k, v in new.items():
            old = V4.at[g, k] if g in V4.index else np.nan
            if np.isnan(old):
                ok = ""
            elif k.startswith("n_"):
                ok = "yes" if int(round(v)) == int(old) else "NO"
            else:
                ok = "yes" if abs(v - old) <= TOL_KM else "NO"
            rows.append(dict(gate=g, metric=k, v4=old, new=v, diff=v - old,
                             match=ok))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("collocations", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=248,
                    help="scenes from the top of the table (v4 used 248)")
    a = ap.parse_args(argv)

    paths = load_paths()
    nc = a.collocations or paths.collocations
    if nc is None:
        sys.exit("give the collocations file, or set it in [paths]")
    cases = load_cases(nc)[: a.n]
    out = paths.outputs / "v4check"

    runs = V.run(cases, [V.Variant("new", DEFAULT)], out, paths=paths,
                 index="legacy")                    # v4 sampling rule
    run = runs["new"]
    err = run.cases[run.cases.qc == "error"]
    if len(err):
        print(f"\n{len(err)} cases failed; rerun to retry:\n"
              + err[["scene", "error"]].to_string(index=False))

    t = headline(run)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 120):
        print("\n" + t.to_string(index=False))
    s = V.summary(run, gate=0.6, qc=None)
    print(f"\nnew rmse_corr (leave-one-case-out) at gate 0.6: "
          f"{s['rmse_corr']:.3f}   (v4: 0.584 in-sample, 0.585 5-fold CV)")
    verdicts = run.cases.qc.value_counts().to_dict()
    print(f"verdicts: {verdicts}")
    bad = t[t.match == "NO"]
    print("\nALL HEADLINE NUMBERS MATCH" if bad.empty
          else f"\n{len(bad)} MISMATCH(ES) -- see rows marked NO")
    return 0 if bad.empty else 1


if __name__ == "__main__":
    sys.exit(main())