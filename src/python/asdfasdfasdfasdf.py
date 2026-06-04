import os
from pathlib import Path
import glob

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from scipy.interpolate import RectBivariateSpline
from scipy import fft
import math
import re
import textwrap
import matplotlib.patches as mpatches
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt


def _wrap_path(p, width=36):
    """Wrap a path at separators when possible; hard-break only oversized segments."""
    s = str(p)
    tokens = re.findall(r"[^\\/]+[\\/]?|[\\/]", s)
    out, current = [], ""
    for tok in tokens:
        if len(current) + len(tok) <= width:
            current += tok
        else:
            if current:
                out.append(current)
            while len(tok) > width:
                out.append(tok[:width])
                tok = tok[width:]
            current = tok
    if current:
        out.append(current)
    return "\n".join(out)


def save_mtf_report(mtf, dcm_info, dcm_dir, output_path, title=None):
    """
    Save a one-page PDF + companion .json for an MTF measurement.

    Layout (2 rows × 4 cols):
        col 0      col 1      col 2 (spans rows)   col 3 (sidebar, spans rows)
        2D PSF     2D MTF     Ensemble image       Acquisition / metrics / source
        1D PSF     1D MTF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xxx_mm = mtf["xxx"] * 10.0  # cm -> mm
    freq = mtf["freq"]
    f_peak, eta_peak, f50, f10, f02, f_tail = mtf["mtf_eval"]

    fig = plt.figure(figsize=(15, 8), dpi=120)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.85], hspace=0.32, wspace=0.32)

    # --- Ensemble image with ROI + wire markers (spans both rows) ---
    ax = fig.add_subplot(gs[:, 0])
    ens = mtf["ensemble_image"]
    cx_w, cy_w = mtf["wire_xy"]
    psfroi_px = mtf["psfroi"]
    bg = float(mtf["background_value"])
    pk = float(mtf["psf_peak"])

    # Anchor window between background and wire peak so the wire is always
    # visible (peak sits at ~83 % brightness with these numbers) while
    # preserving enough range to see the surrounding phantom.
    ww = max(1.5 * pk, 400.0)
    wl = bg + 0.5 * pk
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    ax.imshow(ens, cmap="gray", vmin=lo, vmax=hi)

    # ROI rectangle — matches the exact slice taken in compute_mtf
    neg_pad = math.ceil(psfroi_px / 2)
    pos_pad = math.floor(psfroi_px / 2)
    rect = mpatches.Rectangle(
        (cx_w - neg_pad - 0.5, cy_w - neg_pad - 0.5),  # -0.5 → pixel edges
        neg_pad + pos_pad,
        neg_pad + pos_pad,
        fill=False,
        edgecolor="red",
        lw=1.5,
    )
    ax.add_patch(rect)
    # Center dot on detected wire location
    ax.plot(cx_w, cy_w, marker=".", color="red", markersize=2)
    ax.set_title(f"Ensemble image — (W/L = {ww:.0f}/{wl:.0f})")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- 2D PSF ---
    ax = fig.add_subplot(gs[0, 1])
    psf = mtf["psf_2D"]
    H, W = psf.shape
    ext_mm = mtf["unit_dx"] * 10.0 * np.array([-W / 2, W / 2, -H / 2, H / 2])
    ax.imshow(psf, cmap="gray", extent=ext_mm, origin="lower")
    ax.set_title("PSF (2D)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # --- 2D MTF (normalized) ---
    ax = fig.add_subplot(gs[0, 2])
    mtf2d = mtf["mtf_2D"]
    fmax = min(mtf["maxfreq"], freq[-1] if len(freq) else 1.0, f_tail)
    ext_f = np.array([-fmax, fmax, -fmax, fmax])
    ax.imshow(
        mtf2d / mtf2d.max(),
        cmap="gray",
        extent=ext_f,
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax.set_title("MTF (2D, normalized)")
    ax.set_xlabel(r"$f_x$ (1/mm)")
    ax.set_ylabel(r"$f_y$ (1/mm)")

    # --- 1D PSF ---
    ax = fig.add_subplot(gs[1, 1])
    labels = ["0°", "90°", "radial avg"]
    ax.plot(xxx_mm, mtf["psf_1D"], lw=1.5)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"PSF (1D)  —  FWHM = {mtf['fwhm_psf'] * 10:.3f} mm")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("Normalized")
    ax.grid(alpha=0.3)

    # --- 1D MTF ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(freq, mtf["mtf_1D"], lw=1.5)
    for f, lvl, name in zip(
        [f_peak, f50, f10, f02],
        [eta_peak, 0.5, 0.1, 0.02],
        ["f_peak", "f50", "f10", "f02"],
    ):
        if np.isfinite(f):
            ax.axvline(f, color="k", ls=":", lw=0.6, alpha=0.5)
            ax.axhline(lvl, color="k", ls=":", lw=0.6, alpha=0.5)
            ax.annotate(
                (
                    f"{name}={f:.2f}"
                    if name != "f_peak"
                    else f"{name}=({f:.3f}, {lvl:.3f})"
                ),
                xy=(f, lvl),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, eta_peak * 1.05)
    ax.set_title("MTF (1D)")
    ax.set_xlabel("Frequency (1/mm)")
    ax.set_ylabel("MTF")
    ax.grid(alpha=0.3)

    # --- Sidebar: acquisition + metrics + source ---
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    ps = dcm_info.get("PixelSpacing", [0])
    ps_val = float(ps[0]) if len(ps) else 0.0
    wrapped_dir = _wrap_path(dcm_dir, width=36)

    lines = [
        "── Acquisition ──",
        f"Patient ID  : {dcm_info.get('PatientID', '')}",
        f"Study date  : {dcm_info.get('StudyDate', '')}",
        f"Scanner     : {dcm_info.get('Manufacturer', '')}",
        f"Model       : {dcm_info.get('ModelName', '')}",
        f"Station     : {dcm_info.get('StationName', '')}",
        f"Kernel      : {dcm_info.get('ConvolutionKernel', '')}",
        f"rFOV        : {dcm_info.get('ReconstructionDiameter', '')} mm",
        f"Pixel size  : {ps_val:.4f} mm",
        "",
        "── Metrics ──",
        f"FWHM (PSF)  : {mtf['fwhm_psf'] * 10:.3f} mm",
        f"f_peak      : {f_peak:.3f} 1/mm",
        f"eta_peak    : {eta_peak:.3f}",
        f"f50         : {f50:.3f} 1/mm",
        f"f10         : {f10:.3f} 1/mm",
        f"f02         : {f02:.3f} 1/mm",
        f"Background  : {mtf['background_value']:.1f} HU",
        f"PSF peak    : {mtf['psf_peak']:.1f} HU (above bg)",
        "",
        "── Computation ──",
        f"Δf (freq)   : {mtf['unit']:.5f} 1/mm",
        f"Δx (space)  : {mtf['unit_dx'] * 10:.5f} mm",
        f"ind_max     : {tuple(int(v) for v in mtf['ind_max'])}",
        f"Wire (x,y)  : ({cx_w}, {cy_w})",
        "",
        "── Source ──",
        wrapped_dir,
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        family="monospace",
        fontsize=8.5,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # --- .json: DICOM metadata + scalar metrics (unchanged) ---
    def _serial(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__iter__") and not isinstance(o, str):
            return list(o)
        return str(o)

    meta = {
        "dcm_info": {k: _serial(v) for k, v in dcm_info.items()},
        "dcm_dir": str(dcm_dir),
        "metrics": {
            "fwhm_psf_mm": float(mtf["fwhm_psf"]) * 10,
            "f_peak": float(f_peak),
            "eta_peak": float(eta_peak),
            "f50": float(f50),
            "f10": float(f10),
            "f02": float(f02),
            "f_tail": float(f_tail),
            "background_HU": float(mtf["background_value"]),
            "psf_peak_HU": float(mtf["psf_peak"]),
            "unit_df_per_cm": float(mtf["unit"]),
            "unit_dx_cm": float(mtf["unit_dx"]),
            "ind_max": [int(v) for v in mtf["ind_max"]],
            "wire_xy": [int(cx_w), int(cy_w)],
            "psfroi_px": int(psfroi_px),
            "psf_1D": _serial(mtf["psf_1D"]),
            "mtf_1D": _serial(mtf["mtf_1D"]),
        },
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=_serial)

    return {"pdf": str(pdf_path), "json": str(json_path)}


def normal_round(n):
    """Round-half-away-from-zero (matches MATLAB / original code convention)."""
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def is_leaf(path):
    p = Path(path)
    if not p.is_dir():
        return False
    # Check if any item inside is a directory
    return not any(child.is_dir() for child in p.iterdir())


def has_dicom_files(path):
    p = Path(path)
    if not p.is_dir():
        return False
    if not any(child.is_file() for child in p.iterdir()):
        return False
    # Check if any file inside is a DICOM file
    for child in p.iterdir():
        if child.is_file():
            try:
                pydicom.dcmread(str(child), stop_before_pixels=True)
                return True
            except Exception:
                continue
    return False


def get_dicom_info(dcm_path):
    def int_or_array(val):
        if isinstance(val, pydicom.multival.MultiValue):
            return [int(v) for v in val]
        return [int(val)]

    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return {
            "PatientID": ds.get("PatientID", ""),
            "StudyDate": ds.get("StudyDate", ""),
            "Modality": ds.get("Modality", ""),
            "Manufacturer": ds.get("Manufacturer", ""),
            "ModelName": ds.get("ManufacturerModelName", ""),
            "StationName": ds.get("StationName", ""),
            "ConvolutionKernel": ds.get("ConvolutionKernel", ""),
            "PixelSpacing": ds.get("PixelSpacing", []),
            "ReconstructionDiameter": ds.get("ReconstructionDiameter", ""),
            "WindowCenter": int_or_array(ds.get("WindowCenter", "")),
            "WindowWidth": int_or_array(ds.get("WindowWidth", "")),
        }
    except Exception as e:
        print(f"Error reading DICOM file {dcm_path}: {e}")
        return {}


def order_dcm_slices(dcm_files):
    slices = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            instance_number = int(ds.get("InstanceNumber", 0))
            slices.append((instance_number, dcm_file))
        except Exception as e:
            print(f"Error reading DICOM file {dcm_file}: {e}")
    slices.sort(key=lambda x: x[0])
    return [s[1] for s in slices]


def get_dcm_dirs_from_path(parent_dir):
    dcm_dirs = [
        x
        for x in glob.glob(os.path.join(parent_dir, "**"), recursive=True)
        if os.path.isdir(x)
        and "test" not in x.lower()
        and is_leaf(x)
        and has_dicom_files(x)
    ]
    return sorted(dcm_dirs)


def get_dcm_dirs_from_txt(txt_file):
    dcm_dirs = []
    with open(txt_file, "r") as f:
        for line in f:
            dcm_dirs.append(line.strip())
    return sorted(dcm_dirs)


def save_dcm_dirs_to_txt(dcm_dirs, txt_file):
    with open(txt_file, "w") as f:
        for dcm_dir in dcm_dirs:
            f.write(dcm_dir + "\n")


def get_ensemble_image(dcm_files):
    images = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            img = apply_modality_lut(ds.pixel_array, ds)
            images.append(img)
        except Exception as e:
            print(f"Error reading DICOM file {dcm_file}: {e}")
    if not images:
        raise ValueError("No valid DICOM images found.")
    ensemble_image = np.mean(images, axis=0)
    return ensemble_image


def rescale_image(image, image_min=0, image_max=255, _type=np.uint8):
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=_type)
    scaled = (image - img_min) / (img_max - img_min) * (
        image_max - image_min
    ) + image_min
    return scaled.astype(_type)


def window_image(image, window_center, window_width):
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    windowed = np.clip(image, lower_bound, upper_bound)
    return rescale_image(windowed)


def show_image(image, title="Image", window_center=None, window_width=None):
    if window_center is not None and window_width is not None:
        image = window_image(image, window_center, window_width)
    image = rescale_image(image)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_wire(dcm_files, dcm_info, verbose=0):
    ensemble_image = get_ensemble_image(dcm_files)
    img = ensemble_image.astype(np.float64)
    h, w = img.shape

    px = float(min(dcm_info["PixelSpacing"]))
    sigma_wire = max(0.5 / px, 1.0)

    # 1) Identify the bright ring as the largest above-threshold component.
    #    We don't care about its exact shape, only where NOT to look for the wire.
    thresh = img.mean() + 2.0 * img.std()
    ring_mask = (img > thresh).astype(np.uint8)
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(ring_mask, 8)

    if n_lbl > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        ring_lbl = 1 + int(np.argmax(areas))
        ring_cx, ring_cy = centroids[ring_lbl]
        # Restrict search to well inside the ring's bounding box
        bw = stats[ring_lbl, cv2.CC_STAT_WIDTH]
        bh = stats[ring_lbl, cv2.CC_STAT_HEIGHT]
        inner_r = 0.35 * min(bw, bh)
    else:
        ring_cx, ring_cy, inner_r = w / 2, h / 2, min(h, w) / 4

    yy, xx = np.mgrid[:h, :w]
    search_mask = np.hypot(xx - ring_cx, yy - ring_cy) < inner_r

    # 2) DoG bandpass at the wire scale; the ring is excluded so it can't win.
    #    DoG also kills any smooth gradient bleeding in from the ring's PSF skirt.
    small = cv2.GaussianBlur(img, (0, 0), sigma_wire)
    large = cv2.GaussianBlur(img, (0, 0), sigma_wire * 4)
    dog = small - large
    dog[~search_mask] = -np.inf

    y0, x0 = np.unravel_index(np.argmax(dog), dog.shape)

    # 3) Sub-pixel centroid in a small window of the raw image.
    wsz = int(round(3 * sigma_wire))
    y_lo, y_hi = max(y0 - wsz, 0), min(y0 + wsz + 1, h)
    x_lo, x_hi = max(x0 - wsz, 0), min(x0 + wsz + 1, w)
    patch = img[y_lo:y_hi, x_lo:x_hi].copy()
    patch -= patch.min()
    yy_p, xx_p = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    cy = (patch * yy_p).sum() / patch.sum()
    cx = (patch * xx_p).sum() / patch.sum()

    # Visualize
    if verbose > 0:
        vis = cv2.cvtColor(rescale_image(ensemble_image), cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), 2)
        cv2.drawMarker(
            vis,
            (int(round(cx)), int(round(cy))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1,
        )
        show_image(vis, title="Detected Wire")

    return cx, cy


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
    wire_xy,
    dcm_info,
    psfroi,
    zeropadding=16,
    _sweep=False,
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
        psf_2D        : (n, n) interpolated PSF (background-subtracted)
        psf_1D        : (m,) radially-averaged PSF
        xxx           : (m,) PSF radial coordinate in cm, mirrored to be symmetric
        fwhm_psf      : FWHM of the radially-averaged PSF (cm)
        mtf_2D        : (k, k) 2D MTF magnitude (already centered/cropped)
        mtf_1D        : (k/2,) radially-averaged MTF
        freq          : (k/2,) MTF frequency axis in 1/cm
        mtf_eval      : (3,) [f50, f10, f02] cutoff frequencies in 1/cm
        background_value : float, median HU of the background ring
        psf_peak      : float, peak HU above background
        ind_max       : (row, col) of PSF peak in oversampled grid
        unit          : 1/cm per MTF frequency bin
        unit_dx       : cm per oversampled spatial pixel
    """

    maxfreq = 0.5 / (float(dcm_info["PixelSpacing"][0]) * 10)  # 1/mm
    cx0, cy0 = wire_xy
    img_hu = np.asarray(img_hu, dtype=np.float32)

    rFOV_mm = float(
        dcm_info["ReconstructionDiameter"],
    )
    image_dim_for_pixel_size = dcm_info["ReconstructionDiameter"] / (
        dcm_info["PixelSpacing"][0]
    )
    H, W = img_hu.shape
    if image_dim_for_pixel_size is None:
        image_dim_for_pixel_size = H

    psfroi = int(psfroi)
    cy0, cx0 = int(cy0), int(cx0)
    if psfroi < 8:
        raise ValueError("psfroi must be >= 8 pixels.")

    dense = max(int(160 / psfroi), 4)
    pos_pad = math.floor(psfroi / 2)
    neg_pad = math.ceil(psfroi / 2)

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
    psf_1D_radial = psf_polar.mean(axis=1, keepdims=True)
    psf_1D_avg = psf_1D_radial / psf_peak
    psf_polar_norm = psf_polar / psf_peak

    # --- Build symmetric 1D PSF (mirror to negative side) ---
    # cm per oversampled pixel
    unit_dx = float(rFOV_mm) / image_dim_for_pixel_size / dense
    xxx = np.arange(psfccc) * unit_dx
    xxx = np.concatenate((-np.flipud(xxx)[:-1], xxx))
    psf_1D_avg_full = np.concatenate((np.flipud(psf_1D_avg[0:]), psf_1D_avg[1:]))
    psf_1D = psf_1D_avg_full.flatten()

    # --- FWHM ---
    halfMax = (psf_1D.min() + psf_1D.max()) / 2
    idx_above = np.where(psf_1D >= halfMax)[0]
    fwhm_psf = (
        float(xxx[idx_above[-1]] - xxx[idx_above[0]]) if idx_above.size >= 2 else 0.0
    )

    # --- MTF: zero-pad and 2D FFT ---
    unit = 1.0 / unit_dx / (nsize * zeropadding)  # 1/mm per freq bin

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
    mtf_1D = mtf_polar_norm.mean(axis=1)

    # --- Cutoff frequencies (1-based index times unit, matching original) ---
    def find_first_below(arr, thresh):
        below = arr - thresh
        indices = np.where(np.sign(below[:-1]) != np.sign(below[1:]))[0]
        if len(indices) == 0:
            return len(arr) - 1
        return min(indices.min() + 1, len(arr) - 1)

    t = np.linspace(0, 1, len(mtf_1D))
    t_new = np.linspace(0, 1, 100001)
    freq_new = np.interp(t_new, t, freq)
    mtf_1d_avg_new = np.interp(t_new, t, mtf_1D)

    eta_50 = 0.5
    eta_10 = 0.1
    eta_2 = 0.02

    idx_peak = mtf_1d_avg_new.argmax()
    idx_50 = find_first_below(mtf_1d_avg_new, eta_50)
    idx_10 = find_first_below(mtf_1d_avg_new, eta_10)
    idx_2 = find_first_below(mtf_1d_avg_new, eta_2)

    f_peak = freq_new[idx_peak]
    f_50 = freq_new[idx_50]
    f_10 = freq_new[idx_10]
    f_2 = freq_new[idx_2]
    f_tail = f_2 * 2

    if f_peak == 0:
        eta_peak = 1
    else:
        eta_peak = max(mtf_1d_avg_new)

    # Sanity check:
    eps = 1e-3
    if abs(mtf_1d_avg_new[idx_50] - eta_50) > eps:
        f_50 = float("nan")
    if abs(mtf_1d_avg_new[idx_10] - eta_10) > eps:
        f_10 = float("nan")
    if abs(mtf_1d_avg_new[idx_2] - eta_2) > eps:
        f_2 = float("nan")
        f_tail = float("nan")

    mtf_eval = np.array([f_peak, eta_peak, f_50, f_10, f_2, f_tail])

    psf_disp = psf[1 : nsize - 1, 1 : nsize - 1]

    if _sweep:
        return dict(
            fwhm_psf=fwhm_psf,
            mtf_eval=mtf_eval,
        )

    return dict(
        psf_2D=psf_disp,
        psf_1D=psf_1D,
        xxx=xxx,
        fwhm_psf=fwhm_psf,
        mtf_2D=mtf,
        mtf_1D=mtf_1D,
        freq=freq,
        mtf_eval=mtf_eval,
        background_value=background_value,
        psf_peak=psf_peak,
        ind_max=ind_max,
        unit=float(unit),
        unit_dx=float(unit_dx),
        ensemble_image=img_hu,
        wire_xy=(int(cx0), int(cy0)),
        psfroi=int(psfroi),
        maxfreq=float(maxfreq),
    )


def find_optimal_roi(
    img_hu,
    wire_xy,
    dcm_info,
    sizes=None,
    stability_tol=0.02,
    n_consecutive=2,
    output_path=None,
):
    """
    Sweep ROI sizes and pick the smallest one at which the MTF has converged.

    Convergence criterion: f50, f10, and f02 each change by less than
    `stability_tol` (fractional) across the next `n_consecutive` larger
    sizes — i.e., the curve has flattened, not just briefly crossed.

    Parameters
    ----------
    dcm_files, dcm_info : same as compute_mtf.
    sizes : iterable of int, optional
        ROI side lengths (px) to test. Default: 8..64 in irregular steps.
    stability_tol : float
        Relative tolerance for declaring a metric stable.
    n_consecutive : int
        Number of consecutive larger ROIs that must agree with the candidate.
    output_path : str or Path, optional
        If given, save a PDF showing the convergence curves.

    Returns
    -------
    dict with keys:
        sizes, f50, f10, f02, fwhm_mm  — sweep results
        optimal_size                   — recommended ROI side (px)
        reason                         — explanation string
    """
    if sizes is None:
        sizes = np.arange(8, 128 + 1, 4)

    rows = []
    for s in sizes:
        try:
            m = compute_mtf(img_hu, wire_xy, dcm_info, psfroi=int(s), _sweep=True)
            rows.append(
                dict(
                    size=int(s),
                    f50=float(m["mtf_eval"][2]),
                    f10=float(m["mtf_eval"][3]),
                    f02=float(m["mtf_eval"][4]),
                    f_tail=float(m["mtf_eval"][5]),
                    fwhm_mm=float(m["fwhm_psf"]) * 10.0,
                )
            )
        except Exception as e:
            print(f"  ROI {s}: skipped ({e})")

    if not rows:
        raise RuntimeError("No ROI size produced a valid MTF.")

    sz = np.array([r["size"] for r in rows])
    f50 = np.array([r["f50"] for r in rows])
    f10 = np.array([r["f10"] for r in rows])
    f02 = np.array([r["f02"] for r in rows])
    fwhm = np.array([r["fwhm_mm"] for r in rows])

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-12)

    optimal, reason = None, ""
    for i in range(len(sz) - n_consecutive):
        stable = True
        for j in range(1, n_consecutive + 1):
            for arr in (f50, f10, f02):
                if not np.isfinite(arr[i]) or not np.isfinite(arr[i + j]):
                    stable = False
                    break
                if rel(arr[i], arr[i + j]) > stability_tol:
                    stable = False
                    break
            if not stable:
                break
        if stable:
            optimal = int(sz[i])
            reason = (
                f"Smallest ROI where f50/f10/f02 each agree within "
                f"{stability_tol * 100:.1f}% across the next {n_consecutive} sizes."
            )
            break

    if optimal is None:
        optimal = int(sz[-1])
        reason = (
            "No convergence within the tested range — "
            "extend `sizes` upward or verify the wire ROI placement."
        )

    result = {
        "sizes": sz,
        "f50": f50,
        "f10": f10,
        "f02": f02,
        "fwhm_mm": fwhm,
        "optimal_size": optimal,
        "reason": reason,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=120)
        for ax, y, name in zip(
            axes.flat,
            [f50, f10, f02, fwhm],
            ["f50 (1/mm)", "f10 (1/mm)", "f02 (1/mm)", "FWHM (mm)"],
        ):
            ax.plot(sz, y, "o-", lw=1.5)
            ax.axvline(
                optimal, color="r", ls="--", alpha=0.6, label=f"optimal = {optimal} px"
            )
            ax.set_xlabel("ROI side (px)")
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"ROI convergence — {reason}", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    return result


parent_dir = r"Z:\eb028936\CTPro_Images\images\Phan_Wire"
# dcm_dirs = get_dcm_dirs_from_path(parent_dir)
# save_dcm_dirs_to_txt(dcm_dirs, "dcm_dirs.txt")
dcm_dirs = get_dcm_dirs_from_txt("dcm_dirs.txt")
out_dir = Path("mtf_reports")
if not out_dir.exists():
    out_dir.mkdir(parents=True)
for dcm_dir in dcm_dirs:
    print(f"Processing directory: {dcm_dir}")
    dcm_files = [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir)]
    dcm_files = order_dcm_slices(dcm_files)
    dcm_info = get_dicom_info(dcm_files[0])

    stem = dcm_dir.replace(parent_dir + "\\", "").replace("\\", "^")

    img_hu = get_ensemble_image(dcm_files)
    wire_xy = find_wire(dcm_files, dcm_info)

    roi_sweep = find_optimal_roi(
        img_hu,
        wire_xy,
        dcm_info,
        output_path=out_dir / f"{stem}_roi_sweep.pdf",
    )
    print(f"  optimal ROI = {roi_sweep['optimal_size']} px  ({roi_sweep['reason']})")

    mtf = compute_mtf(img_hu, wire_xy, dcm_info, psfroi=roi_sweep["optimal_size"])
    save_mtf_report(
        mtf,
        dcm_info,
        dcm_dir,
        out_dir / stem,
        title=f"MTF — {dcm_info.get('ConvolutionKernel', '')} "
        f"(ROI={roi_sweep['optimal_size']} px)",
    )
