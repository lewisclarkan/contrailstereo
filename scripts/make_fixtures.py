"""Capture  fixtures: cropped GOES channels + ERA5 profile +
scene metadata, small enough to commit. Run locally; CI consumes."""
import json
import os
import sys
import numpy as np
from pyproj import Proj

sys.path.insert(0, "src")
from contrailstereo.config import DEFAULT, SAT_LON, R_EQ, R_POL
from contrailstereo.geometry import build_grid
from contrailstereo.data.goes import fetch_pair
from contrailstereo.data.era5 import fetch_era5_winds
from contrailstereo.data.caliop import build_scene_table

NC_PATH = sys.argv[1]
FIXTURES = [0, 8, 12, 41, 46]
OUT = "tests/fixtures"

scenes, prof_df = build_scene_table(NC_PATH)


def crop_to_grid(ds, glat, glon, margin=0.3):
    """Subset a CMI dataset to the grid's extent (+margin deg) in the
    satellite's own scan coordinates."""
    pj = ds["goes_imager_projection"]
    H = float(pj.attrs["perspective_point_height"])
    p = Proj(proj="geos", h=H,
             lon_0=float(pj.attrs["longitude_of_projection_origin"]),
             sweep=pj.attrs.get("sweep_angle_axis", "x"), a=R_EQ, b=R_POL)
    corners_lon = [glon.min() - margin, glon.max() + margin]
    corners_lat = [glat.min() - margin, glat.max() + margin]
    xs, ys = [], []
    for lo in corners_lon:
        for la in corners_lat:
            x, y = p(lo, la)
            xs.append(x / H); ys.append(y / H)      # back to scan-angle units
    return ds.sel(x=slice(min(xs), max(xs)), y=slice(max(ys), min(ys)))


for si in FIXTURES:
    row = scenes.iloc[si]
    pr = prof_df[prof_df.goes_file == row.goes_file]
    glat, glon, _ = build_grid(pr)
    d = os.path.join(OUT, f"scene_{si:04d}")
    os.makedirs(d, exist_ok=True)

    when = row.time.to_pydatetime()
    for sat in (16, 17):
        fields, domain = fetch_pair(when, sat)
        for ch in (14, 15):
            crop = crop_to_grid(fields[ch], glat, glon)
            crop.to_netcdf(os.path.join(d, f"G{sat}_C{ch:02d}.nc"),
                           encoding={"CMI": {"zlib": True, "complevel": 6}})

    winds = fetch_era5_winds(when, float(row.lat), float(row.lon))
    if winds:
        hs = np.arange(6.0, 16.01, 0.25)
        np.savez(os.path.join(d, "era5.npz"), h=hs,
                 u=[winds[0](h) for h in hs], v=[winds[1](h) for h in hs])

    pr[["lat", "lon", "top_km"]].to_csv(os.path.join(d, "profiles.csv"),
                                        index=False)
    meta = dict(scene=si, time=str(row.time), lat=float(row.lat),
                lon=float(row.lon), goes_file=str(row.goes_file),
                top_km=float(row.top_km), top_sd=float(row.top_sd),
                n_prof=int(row.n))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"scene {si}: fixture written "
          f"({sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6:.1f} MB)")