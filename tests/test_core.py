import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from contrailstereo.config import DEFAULT, SAT_LON
from contrailstereo.core.fields import hp_km, make_bandpass
from contrailstereo.core.peaks import cirrus_peak, refine_peak
from contrailstereo.core.match import local_corr, project_pair
from contrailstereo.geometry import apparent_surface_latlon


# ------------------------------------------------------------- fields
def test_hp_constant_field_is_zero():
    out = hp_km(np.full((60, 60), 3.7), (6, 6))
    assert np.nanmax(np.abs(out)) < 1e-9


def test_hp_nan_propagates_without_bleed():
    f = np.random.default_rng(0).normal(size=(60, 60))
    f[20:23, :] = np.nan
    out = hp_km(f, (4, 4))
    assert np.isnan(out[21]).all()            # NaN in -> NaN out
    assert np.isfinite(out[10]).all()         # far rows untouched


def test_hp_transfer_function():
    """Gaussian high-pass: gain = 1 - exp(-2 pi^2 sigma^2 / lambda^2).
    Check attenuation at a long wavelength and transparency at a short
    one against theory, rather than pretending the filter is ideal."""
    x = np.arange(240, dtype=float)
    sig = 6.0

    def gain(lam):
        f = np.tile(np.sin(2 * np.pi * x / lam), (60, 1))
        out = hp_km(f, (sig, sig))
        return out[30, 40:-40].std() / f[30, 40:-40].std()

    expected = lambda lam: 1.0 - np.exp(-2 * np.pi**2 * sig**2 / lam**2)
    for lam in (8.0, 30.0, 100.0):
        assert abs(gain(lam) - expected(lam)) < 0.06, lam


# -------------------------------------------------------------- peaks
def _bump(hs, h0, w=1.2, a=0.6):
    return a * np.exp(-0.5 * ((hs - h0) / w) ** 2)


def test_peak_found_and_refined():
    hs = np.arange(0, 16.01, 0.25)
    h, r, nlow, npk = cirrus_peak(hs, _bump(hs, 11.13))
    hf, rf, sig = refine_peak(hs, _bump(hs, 11.13), h)
    assert abs(hf - 11.13) < 0.02 and np.isfinite(sig)


def test_monotone_to_endpoint_yields_no_peak():
    hs = np.arange(0, 16.01, 0.25)
    h, *_ = cirrus_peak(hs, np.linspace(-0.2, 0.6, len(hs)))
    assert not np.isfinite(h)                 # scene-28 lesson


def test_low_peak_counted_high_peak_selected():
    hs = np.arange(0, 16.01, 0.25)
    s = _bump(hs, 4.0, a=0.8) + _bump(hs, 11.0, a=0.4)
    h, r, nlow, npk = cirrus_peak(hs, s)
    assert abs(h - 11.0) < 0.3 and nlow == 1


# -------------------------------------------------------------- match
def test_local_corr_identity():
    A = np.random.default_rng(1).normal(size=(80, 80))
    r, amp = local_corr(A, A.copy(), 15)
    assert np.nanmin(r) > 0.999


def test_synthetic_parallax_end_to_end():
    """The whole geometric core against constructed truth: two 'views' of
    the same texture, each sampled through its own satellite's parallax
    at a known height -> the scan must peak at that height."""
    H_TRUE = 11.0
    rng = np.random.default_rng(2)
    lat0, lon0 = 40.0, -105.0
    glat, glon = np.meshgrid(np.linspace(lat0 - .5, lat0 + .5, 120),
                             np.linspace(lon0 - 1., lon0 + 1., 120),
                             indexing="ij")

    # smooth random texture defined on true cloud position
    tex = gaussian_filter(rng.normal(size=(400, 400)), 3)
    tla = np.linspace(lat0 - 1.5, lat0 + 1.5, 400)
    tlo = np.linspace(lon0 - 3.0, lon0 + 3.0, 400)

    def cloud_field(qlat, qlon):
        i = np.clip(((qlat - tla[0]) / (tla[1] - tla[0])), 0, 398)
        j = np.clip(((qlon - tlo[0]) / (tlo[1] - tlo[0])), 0, 398)
        return tex[i.astype(int), j.astype(int)]

    # a "sampler" per satellite: what that satellite records at ground
    # position g is the cloud texture at the position whose parallax
    # projection lands on g -- constructed by inverse lookup: sample the
    # cloud at the apparent position of (g, H_TRUE) seen from the OTHER
    # direction. Simplest faithful construction: the imagery at ground
    # point g equals texture at the point p with apparent(p, H_TRUE) = g;
    # for the test we approximate p by apparent(g, -H_TRUE) reciprocity.
    def make_fake_sampler(sat):
        def f(alat, alon):
            # invert: which cloud point projects to (alat, alon)?
            # for small heights the map is near-affine; use one Newton step
            pla, plo = alat.copy(), alon.copy()
            for _ in range(3):
                gla, glo2 = apparent_surface_latlon(pla, plo,
                                                   H_TRUE * 1e3,
                                                   SAT_LON[sat])
                pla += alat - gla
                plo += alon - glo2
            return cloud_field(pla, plo)
        return f

    samplers = {16: make_fake_sampler(16), 17: make_fake_sampler(17)}
    hs = np.arange(6.0, 15.01, 0.25)
    lo = []
    wm = np.ones_like(glat, bool)
    for h in hs:
        A, B = project_pair(h, glat, glon, samplers, (4, 4))
        m = np.isfinite(A) & np.isfinite(B)
        lo.append(np.corrcoef(A[m], B[m])[0, 1])
    h_pk, r_pk, *_ = cirrus_peak(hs, np.array(lo))
    assert abs(h_pk - H_TRUE) <= 0.25          # within one grid step
    assert r_pk > 0.9