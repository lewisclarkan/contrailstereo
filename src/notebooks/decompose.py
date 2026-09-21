import numpy as np
import pandas as pd


def loo_residuals(q, hcol="h_map", tcol="top_km", scol="scene"):
    """Per-profile errors with a leave-one-scene-out constant removed."""
    e = (q[hcol] - q[tcol]).values
    sc = q[scol].values
    tot, cnt = e.sum(), len(e)
    out = np.empty_like(e, dtype=float)
    for s in np.unique(sc):
        m = sc == s
        out[m] = e[m] - (tot - e[m].sum()) / (cnt - m.sum())
    return out


def within_between(q, hcol="h_map", tcol="top_km", scol="scene",
                   min_n=2, n_boot=1000, seed=0):
    """Split per-profile error into within-scene scatter and between-scene
    offset spread, both in km, such that within^2 + between^2 = total^2.

    within  : profile-to-profile scatter about each scene's own mean error
              -- matching precision inside a scene
    between : spread of scene mean errors about the overall mean
              -- scene-to-scene variation in what level the match locks to
    """
    q = q.copy()
    q["_e"] = loo_residuals(q, hcol, tcol, scol)
    g = q.groupby(scol)["_e"]
    n = g.count()
    keep = n[n >= min_n].index
    q = q[q[scol].isin(keep)]
    g = q.groupby(scol)["_e"]
    n, mu = g.count(), g.mean()

    total = np.sqrt((q["_e"] ** 2).mean())
    # pooled within-scene variance, weighted by profiles per scene
    within = np.sqrt(np.average(g.var(ddof=0), weights=n))
    # profile-weighted spread of scene means about the pooled mean
    grand = np.average(mu, weights=n)
    between = np.sqrt(np.average((mu - grand) ** 2, weights=n))

    # bootstrap over SCENES, since profiles within a scene are not independent
    rng = np.random.default_rng(seed)
    scenes = mu.index.values
    bw, bb = [], []
    for _ in range(n_boot):
        pick = rng.choice(scenes, len(scenes), replace=True)
        sub = pd.concat([q[q[scol] == s] for s in pick])
        gg = sub.groupby(sub[scol])["_e"]
        nn, mm = gg.count(), gg.mean()
        bw.append(np.sqrt(np.average(gg.var(ddof=0), weights=nn)))
        gr = np.average(mm, weights=nn)
        bb.append(np.sqrt(np.average((mm - gr) ** 2, weights=nn)))

    return dict(
        n_profiles=len(q), n_scenes=len(keep),
        total_km=total, within_km=within, between_km=between,
        within_ci=tuple(np.percentile(bw, [2.5, 97.5])),
        between_ci=tuple(np.percentile(bb, [2.5, 97.5])),
        between_share=between ** 2 / (within ** 2 + between ** 2),
        check_km=np.sqrt(within ** 2 + between ** 2),
    )


if __name__ == "__main__":
    import sys
    res = pd.read_csv(sys.argv[2])
    res["vza_max"] = res[["vza_east", "vza_west"]].max(axis=1)
    pp = pd.read_csv(sys.argv[1]).dropna(subset=["h_map"])
    pp = pp.merge(res[["scene", "vza_max"]], on="scene")
    pp = pp[pp.r_map > 0.6]

    for label, sub in [("full domain", pp),
                       ("inner domain", pp[pp.vza_max <= 65])]:
        d = within_between(sub)
        print(f"\n{label}: {d['n_profiles']} profiles, {d['n_scenes']} scenes")
        print(f"  total    {d['total_km']:.3f} km")
        print(f"  within   {d['within_km']:.3f} km  "
              f"[{d['within_ci'][0]:.3f}, {d['within_ci'][1]:.3f}]")
        print(f"  between  {d['between_km']:.3f} km  "
              f"[{d['between_ci'][0]:.3f}, {d['between_ci'][1]:.3f}]")
        print(f"  between accounts for {100*d['between_share']:.0f}% "
              f"of the error variance")
        print(f"  check: sqrt(w^2+b^2) = {d['check_km']:.3f} "
              f"vs total {d['total_km']:.3f}")