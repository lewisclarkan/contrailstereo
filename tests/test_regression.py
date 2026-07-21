"""Test porting versus previous tabulated results"""

import numpy as np
import pytest
from contrailstereo.core.retrieve import run_scene

TOL_H, TOL_R = 0.10, 0.06

# FILL FROM stereo_results_v4.csv (v4-frozen): scene -> (h_local, r_local)
EXPECTED = {
    0:  (12.435932606943034, 0.5349254818968916),
    8:  (12.681758014755491, 0.7666753256449432),
    12: (10.21845434311886, 0.6997006548092664),
    41: (10.463284207658292, 0.685261725651932),
    46: (11.007415050035336, 0.7050446865956808),
}


@pytest.mark.parametrize("si", list(EXPECTED))
def test_fixture_scene_regression(fixture_scene, si):
    exp_h, exp_r = EXPECTED[si]
    if exp_h is None:
        pytest.skip("expected values not yet filled from v4-frozen CSV")
    rec, prof, grids = run_scene(fixture_scene(si))
    assert abs(rec["h_local"] - exp_h) < TOL_H, rec["h_local"]
    assert abs(rec["r_local"] - exp_r) < TOL_R, rec["r_local"]
    assert rec["qc"] == "ok"