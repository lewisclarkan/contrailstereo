"""Peak selection and refinement on correlation vs height curves"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..config import StereoConfig, DEFAULT


def cirrus_peak(hs, score, cfg: StereoConfig = DEFAULT):
    """Select the strongest interior local maximum at h >= hmin_cirrus_km.

    Parameters
    ----------
    hs : ndarray
        Assumed heights [km] of the scan.
    score : ndarray
        Correlation at each height; NaNs are treated as very low, so a
        peak flanked by NaN coverage gaps still counts as interior only
        if find_peaks sees it as one.

    Returns
    -------
    h_peak, r_peak : float
        Height [km] and correlation of the selected peak (NaN, NaN if no
        admissible interior peak exists).
    n_low_peaks : int
        Interior peaks below the cirrus floor (deck/ground-lock tally).
    n_peaks_total : int
        All interior peaks at any height.
    """
    s = np.nan_to_num(np.asarray(score, float), nan=-9.0)
    pk, _ = find_peaks(s, prominence=cfg.prominence)   # interior maxima only

    if pk.size == 0:
        return np.nan, np.nan, 0, 0
    high = pk[hs[pk] >= cfg.hmin_cirrus_km]
    n_low = int(pk.size - high.size)
    if high.size == 0:
        return np.nan, np.nan, n_low, int(pk.size)
    j = high[np.argmax(s[high])]
    return float(hs[j]), float(s[j]), n_low, int(pk.size)


def refine_peak(hs, score, h0, halfwidth=1.0):
    """Parabolic sub-grid refinement of a peak near h0, with uncertainty.

    Fits a quadratic over |h - h0| <= halfwidth; the vertex is the refined
    height (clipped to the fit window). sigma_h = sqrt(residual_noise /
    curvature) -- a matching-noise floor, NOT a calibrated total
    uncertainty (underdispersed ~x2.4 against truth; see the sigma_total
    calibration item).

    Returns
    -------
    h_ref, r_ref, sigma_h : float
        sigma_h is NaN when the fit is untrustworthy (too few points or
        non-concave), in which case (h_ref, r_ref) fall back to the
        discrete argmax within the window.
    """
    s = np.asarray(score, float)
    hs = np.asarray(hs, float)
    m = np.isfinite(s) & (np.abs(hs - h0) <= halfwidth)
    if m.sum() < 5:
        k = int(np.nanargmax(np.where(m, s, -9)))
        return float(hs[k]), float(s[k]), np.nan
    c = np.polyfit(hs[m], s[m], 2)
    if c[0] >= 0:                          # not concave: no trustworthy peak
        k = int(np.nanargmax(np.where(m, s, -9)))
        return float(hs[k]), float(s[k]), np.nan
    href = float(np.clip(-c[1] / (2 * c[0]), hs[m].min(), hs[m].max()))
    rref = float(np.polyval(c, href))
    resid = s[m] - np.polyval(c, hs[m])
    noise = float(np.std(resid)) if resid.size > 3 else np.nan
    curv = -2 * c[0]
    sigma_h = (float(np.sqrt(noise / curv))
               if (np.isfinite(noise) and curv > 0 and noise > 0) else np.nan)
    return href, rref, sigma_h