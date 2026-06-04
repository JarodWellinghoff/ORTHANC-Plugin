"""
Core MTF/PSF computation, refactored from the original web/Celery pipeline.

Changes from the original Func_for_mtf:
  - No DICOM/Celery/cloud side effects: pure numerical routine that takes an
    HU image array and ROI parameters and returns PSF/MTF data.
  - Background ring is concentric with the ROI (the original code used
    swapped (cx, cy) indexing for the big rect, which broke concentricity
    whenever cx0 != cy0).
  - Polar resampling is vectorized (~100x faster than the nested Python
    loops in the original; bit-for-bit identical to the original's
    normal_round nearest-pixel sampling for non-negative offsets).
  - Returns a dict of named outputs instead of a positional tuple.
"""

from math import ceil, floor
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import fft


def normal_round(n):
    """Round-half-away-from-zero (matches MATLAB / original code convention)."""
    if n - floor(n) < 0.5:
        return floor(n)
    return ceil(n)


def _polar_nearest(img2d, center_yx, n_radial, n_angles=360):
    """Vectorized polar resampling using round-half-up nearest-pixel.

    Matches the original's normal_round() nearest sampling for the
    non-negative offset range that the MTF code actually uses.
    """
    cy, cx = center_yx
    angles = 2.0 * np.pi * np.arange(n_angles) / n_angles
    radii = np.arange(n_radial)
    R, A = np.meshgrid(radii, angles, indexing="ij")
    Xi = np.floor(R * np.cos(A) + cy + 0.5).astype(int)
    Yi = np.floor(R * np.sin(A) + cx + 0.5).astype(int)
    Xi = np.clip(Xi, 0, img2d.shape[0] - 1)
    Yi = np.clip(Yi, 0, img2d.shape[1] - 1)
    return img2d[Xi, Yi]


def compute_mtf(
    img_hu,
    cy0,
    cx0,
    psfroi,
    rFOV_cm,
    image_dim_for_pixel_size=None,
    maxfreq=100.0,
    zeropadding=16,
):
    """Compute PSF / MTF for a wire or bead at (cy0, cx0) with ROI size psfroi.

    Parameters
    ----------
    img_hu : (H, W) ndarray
        Image already in HU (rescale slope/intercept applied).
    cy0, cx0 : int
        ROI center, in (row, col) indices.
    psfroi : int
        ROI side length in pixels (small inner rect).
    rFOV_cm : float
        Reconstruction FOV in cm.
    image_dim_for_pixel_size : int, optional
        Image dimension used to compute pixel size = rFOV / dim.
        Defaults to H. (Use the native acquisition dim if the image
        was post-resampled.)
    maxfreq : float
        Maximum frequency (1/cm) to evaluate the MTF over.
    zeropadding : int
        Zero-padding factor in k-space (16 in the original).

    Returns
    -------
    dict with keys
        psf_2d        : (n, n) interpolated PSF (background-subtracted)
        psf_1d_all    : (m, 3) columns are 0deg, 90deg, radial-avg
        xxx           : (m,) PSF radial coordinate in cm, mirrored to be symmetric
        fwhm_psf      : FWHM of the radially-averaged PSF (cm)
        mtf_2d        : (k, k) 2D MTF magnitude (already centered/cropped)
        mtf_1d_all    : (k/2, 3) columns are 0deg, 90deg, radial-avg
        freq          : (k/2,) MTF frequency axis in 1/cm
        mtf_eval      : (3,) [f50, f10, f02] cutoff frequencies in 1/cm
        background_value : float, median HU of the background ring
        psf_peak      : float, peak HU above background
        ind_max       : (row, col) of PSF peak in oversampled grid
        unit          : 1/cm per MTF frequency bin
        unit_dx       : cm per oversampled spatial pixel
    """
    img_hu = np.asarray(img_hu, dtype=np.float32)
    H, W = img_hu.shape
    if image_dim_for_pixel_size is None:
        image_dim_for_pixel_size = H

    psfroi = int(psfroi)
    cy0, cx0 = int(cy0), int(cx0)
    if psfroi < 8:
        raise ValueError("psfroi must be >= 8 pixels.")

    dense = max(int(160 / psfroi), 4)
    pos_pad = floor(psfroi / 2)
    neg_pad = ceil(psfroi / 2)

    # --- Validate bounds (big rect = ROI + 2 px each side for background ring) ---
    half_big = int(psfroi / 2) + 2
    if (
        cy0 - half_big < 0
        or cy0 + half_big > H
        or cx0 - half_big < 0
        or cx0 + half_big > W
    ):
        raise ValueError(
            f"ROI (center=({cy0},{cx0}), size={psfroi}) extends outside "
            f"image bounds ({H}x{W}). Move the ROI inward or shrink it."
        )

    # --- Background estimation: ring around ROI (now correctly concentric) ---
    half_roi = int(psfroi / 2)
    big_rect = img_hu[
        cy0 - half_big : cy0 + half_big,
        cx0 - half_big : cx0 + half_big,
    ].astype(np.float64)
    br, bc = big_rect.shape
    Y, X = np.meshgrid(np.arange(br), np.arange(bc), indexing="ij")
    cy_b, cx_b = br // 2, bc // 2
    inner = (
        (Y >= cy_b - half_roi)
        & (Y < cy_b + half_roi)
        & (X >= cx_b - half_roi)
        & (X < cx_b + half_roi)
    )
    background_value = float(np.median(big_rect[~inner]))

    # --- Crop ROI, subtract background ---
    Img = img_hu[
        cy0 - neg_pad : cy0 + pos_pad,
        cx0 - neg_pad : cx0 + pos_pad,
    ].astype(np.float64)
    Img = Img - background_value

    # --- Bilinear oversample (modern replacement for deprecated interp2d) ---
    x0 = np.arange(psfroi).astype(float)
    spline = RectBivariateSpline(x0, x0, Img, kx=1, ky=1)
    xind = np.linspace(1, psfroi, psfroi * dense - 3)
    nsize = xind.size + 2
    psf = spline(xind, xind)
    psf = np.pad(psf, ((1, 1), (1, 1)), "constant", constant_values=0)

    # --- Find PSF peak in oversampled grid ---
    inner_psf = psf[1:-1, 1:-1]
    imax = np.unravel_index(np.argmax(inner_psf), inner_psf.shape)
    ind_max = (imax[0], imax[1])
    psf_peak = float(psf[ind_max[0], ind_max[1]])
    if psf_peak <= 0:
        raise ValueError(
            "PSF peak is non-positive after background subtraction. "
            "Check that the ROI is centered on a high-contrast feature."
        )
    psfccc = min(
        nsize - ind_max[0] - 1, nsize - ind_max[1] - 1, ind_max[0] + 1, ind_max[1] + 1
    )

    # --- Polar PSF + radial average (vectorized) ---
    psf_polar = _polar_nearest(psf, ind_max, psfccc, n_angles=360)
    psf_1d_radial = psf_polar.mean(axis=1, keepdims=True)
    psf_1d_avg = psf_1d_radial / psf_peak
    psf_polar_norm = psf_polar / psf_peak

    # --- Build symmetric 1D PSF (mirror to negative side) ---
    unit_dx = rFOV_cm / image_dim_for_pixel_size / dense  # cm per oversampled pixel
    xxx = np.arange(psfccc) * unit_dx
    xxx = np.concatenate((-np.flipud(xxx)[:-1], xxx))
    psf_1d_avg_full = np.concatenate((np.flipud(psf_1d_avg[0:]), psf_1d_avg[1:]))
    psf_polar_norm_full = np.concatenate(
        (np.flipud(psf_polar_norm[0:, 180:]), psf_polar_norm[1:, 0:180])
    )
    # Three columns: 0deg, 90deg, radial-avg (matches original convention)
    psf_1D_all = np.concatenate(
        (psf_polar_norm_full[:, 0:91:90], psf_1d_avg_full), axis=1
    )

    # --- FWHM ---
    radial_avg = psf_1D_all[:, -1]
    halfMax = (radial_avg.min() + radial_avg.max()) / 2
    idx_above = np.where(radial_avg >= halfMax)[0]
    fwhm_psf = (
        float(xxx[idx_above[-1]] - xxx[idx_above[0]]) if idx_above.size >= 2 else 0.0
    )

    # --- MTF: zero-pad and 2D FFT ---
    unit = 1.0 / unit_dx / (nsize * zeropadding)  # 1/cm per freq bin

    psfl = np.zeros((nsize * zeropadding, nsize * zeropadding))
    psfl[0:nsize, 0:nsize] = psf

    mtf0 = np.abs(fft.fft2(psfl))
    mtf0 = fft.fftshift(mtf0)
    cc = normal_round(nsize * zeropadding / 2)

    # Cap to stay within mtf0 bounds. The original code silently shrinks the
    # slice when maxfreq exceeds the zero-padded grid's Nyquist, which then
    # causes a downstream index error. Capping mtfccc to cc fixes both.
    mtfccc = min(normal_round(maxfreq / unit), cc)
    mtf = mtf0[cc - mtfccc : cc + mtfccc, cc - mtfccc : cc + mtfccc]
    freq = np.arange(mtfccc) * unit

    # --- Polar MTF + radial average ---
    mtf_polar = _polar_nearest(mtf, (mtfccc - 1, mtfccc - 1), mtfccc, n_angles=360)
    mtf_polar_norm = mtf_polar / mtf[mtfccc, mtfccc]
    mtf_1d_avg = mtf_polar_norm.mean(axis=1)
    mtf_1D_all = np.column_stack(
        (mtf_polar_norm[:, 0], mtf_polar_norm[:, 90], mtf_1d_avg)
    )

    # --- Cutoff frequencies (1-based index times unit, matching original) ---
    def find_first_below(arr, thresh):
        below = np.where(arr <= thresh)[0]
        return float((below[0] + 1) * unit) if below.size else float("nan")

    mtf_eval = np.array(
        [
            find_first_below(mtf_1d_avg, 0.53),
            find_first_below(mtf_1d_avg, 0.105),
            find_first_below(mtf_1d_avg, 0.023),
        ]
    )

    psf_disp = psf[1 : nsize - 1, 1 : nsize - 1]

    return dict(
        psf_2d=psf_disp,
        psf_1d_all=psf_1D_all,
        xxx=xxx,
        fwhm_psf=fwhm_psf,
        mtf_2d=mtf,
        mtf_1d_all=mtf_1D_all,
        freq=freq,
        mtf_eval=mtf_eval,
        background_value=background_value,
        psf_peak=psf_peak,
        ind_max=ind_max,
        unit=float(unit),
        unit_dx=float(unit_dx),
    )
