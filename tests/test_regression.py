"""Test porting versus previous tabulated results"""

import numpy as np
import pytest
from contrailstereo.core.retrieve import run_scene

TOL_H, TOL_R = 0.10, 0.06

# scene -> (h_local, r_local)
EXPECTED = {
    0:  (12.428229972295018, 0.7304172333357890),
    8:  (12.816813369198638, 0.9022128791579789),
    12: (10.209732376065160, 0.8697948994707438),
    41: (10.504625517931550, 0.8584396268094228),
    46: (11.182688644208719, 0.8834683616128096),
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