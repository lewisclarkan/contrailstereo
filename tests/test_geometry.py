import numpy as np
import pandas as pd

from contrailstereo.geometry import (apparent_surface_latlon, vza_deg, 
                                     parallax_displacement_km, build_grid, 
                                     km_filters)
from contrailstereo.config import SAT_LON, DEFAULT

def test_zero_height_is_identity():
    glat, glon = apparent_surface_latlon(40.0, -105.0, 0.0, SAT_LON[16])
    assert abs(glat - 40.0) < 1e-6 and abs(glon - (-105.0)) < 1e-6


def test_subsatellite_vza_zero_and_no_parallax():
    assert vza_deg(0.0, SAT_LON[16], SAT_LON[16]) < 0.05
    assert parallax_displacement_km(0.0, SAT_LON[16], 10.0, SAT_LON[16]) < 0.05


def test_displacement_magnitude_midlat():
    # ~10 km cloud over CONUS: displacement of order 10 km;
    # tan(50 deg)*10 ~ 12 -- assert the right regime.
    d = parallax_displacement_km(40.0, -105.0, 10.0, SAT_LON[16])
    assert 8.0 < d < 20.0


def test_displacement_linear_in_h():
    d5 = parallax_displacement_km(40.0, -105.0, 5.0, SAT_LON[16])
    d10 = parallax_displacement_km(40.0, -105.0, 10.0, SAT_LON[16])
    assert abs(d10 / d5 - 2.0) < 0.05        # near-linear over cruise band


def test_east_west_satellites_displace_oppositely_in_lon():
    _, lon_e = apparent_surface_latlon(40.0, -105.0, 10e3, SAT_LON[16])
    _, lon_w = apparent_surface_latlon(40.0, -105.0, 10e3, SAT_LON[17])
    assert (lon_e - (-105.0)) * (lon_w - (-105.0)) < 0   # opposite signs


def test_grid_resolution_near_isotropic():
    pr = pd.DataFrame({"lat": [39.0, 41.0], "lon": [-106.0, -104.0]})
    glat, glon, px = build_grid(pr)
    assert abs(px[0] - DEFAULT.px_km) / DEFAULT.px_km < 0.25
    assert abs(px[1] - DEFAULT.px_km) / DEFAULT.px_km < 0.25
    sig, win = km_filters(px)
    assert win[0] % 2 == 1 and win[1] % 2 == 1     # odd windows