"""DICOM loading + wire auto-detection (Hough circle fallback)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2


@dataclass
class DicomSlice:
    path: str
    image_hu: np.ndarray
    rFOV_cm: float
    rescale_slope: float
    rescale_intercept: float
    instance_number: Optional[int] = None
    slice_location: Optional[float] = None
    kernel: Optional[str] = None
    series_description: Optional[str] = None


def load_dicom_path(path: str) -> DicomSlice:
    """Read a single DICOM file and return a DicomSlice (image in HU)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    try:
        img_hu = apply_modality_lut(arr, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img_hu = arr * slope + intercept

    rFOV_cm = float(getattr(ds, "ReconstructionDiameter", 50.0)) / 10.0
    instance_number = getattr(ds, "InstanceNumber", None)
    try:
        instance_number = int(instance_number) if instance_number is not None else None
    except (TypeError, ValueError):
        instance_number = None
    slice_location = getattr(ds, "SliceLocation", None)
    try:
        slice_location = float(slice_location) if slice_location is not None else None
    except (TypeError, ValueError):
        slice_location = None

    return DicomSlice(
        path=path,
        image_hu=img_hu.astype(np.float32),
        rFOV_cm=rFOV_cm,
        rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
        rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
        instance_number=instance_number,
        slice_location=slice_location,
        kernel=getattr(ds, "ConvolutionKernel", None),
        series_description=getattr(ds, "SeriesDescription", None),
    )


def load_dicom_folder(folder: str) -> List[DicomSlice]:
    """Load every DICOM in a folder, sorted by InstanceNumber/SliceLocation/name."""
    folder = Path(folder)
    slices: List[DicomSlice] = []
    for f in folder.iterdir():
        if not f.is_file() or f.name.lower() == "params":
            continue
        try:
            slices.append(load_dicom_path(str(f)))
        except Exception:
            continue

    def keyfn(s: DicomSlice):
        if s.instance_number is not None:
            return (0, s.instance_number)
        if s.slice_location is not None:
            return (1, s.slice_location)
        return (2, s.path)

    slices.sort(key=keyfn)
    return slices


def _disk_kernel(radius: int) -> np.ndarray:
    """Disk kernel for dilation (matches the original code)."""
    k = np.zeros((2 * radius - 1, 2 * radius - 1), np.uint8)
    y, x = np.ogrid[-radius + 1 : radius, -radius + 1 : radius]
    k[x**2 + y**2 <= (radius - 1) ** 2] = 1
    k[0, radius - 2 : k.shape[1] - radius + 2] = 1
    k[-1, radius - 2 : k.shape[1] - radius + 2] = 1
    k[radius - 2 : k.shape[0] - radius + 2, 0] = 1
    k[radius - 2 : k.shape[0] - radius + 2, -1] = 1
    return k


def auto_detect_wire(
    img_hu: np.ndarray, max_radius: int = 25
) -> Optional[Tuple[int, int]]:
    """Try to find a bright wire/bead near the image center via Hough circles.

    Returns (cy, cx) on success, None on failure.
    Falls back to a pure-numpy bright-spot search if OpenCV isn't installed.
    """

    img = img_hu.astype(np.float32)
    dil = cv2.dilate(img, _disk_kernel(7))
    u8 = cv2.normalize(dil, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    h, w = u8.shape
    cy, cx = h // 2, w // 2
    crop_h = crop_w = 0
    circles = None
    y0 = x0 = 0
    while circles is None:
        crop_h += 32 * max(h // 512, 1)
        crop_w += 32 * max(w // 512, 1)
        y0 = max(cy - crop_h // 2, 0)
        y1 = min(cy + crop_h // 2, h)
        x0 = max(cx - crop_w // 2, 0)
        x1 = min(cx + crop_w // 2, w)
        crop = u8[y0:y1, x0:x1]
        m, s = float(crop.mean()), float(crop.std())
        thresh = m + 1.5 * s
        _, binimg = cv2.threshold(
            crop, thresh, max(float(crop.max()), 1.0), cv2.THRESH_BINARY
        )
        binimg = cv2.GaussianBlur(binimg, (5, 5), 0)
        for p1 in range(100, 301, 20):
            for p2 in range(200, 19, -20):
                circles = cv2.HoughCircles(
                    binimg,
                    cv2.HOUGH_GRADIENT,
                    1,
                    20,
                    param1=p1,
                    param2=p2,
                    minRadius=0,
                    maxRadius=max_radius,
                )
                if circles is not None:
                    break
            if circles is not None:
                break
        if crop_h >= h and crop_w >= w:
            break

    if circles is None:
        return None

    cxc, cyc, _ = circles[0, 0]
    cyc = int(round(cyc)) + y0
    cxc = int(round(cxc)) + x0

    # Local peak refinement (Hough centers are not pixel-perfect)
    win = 5
    y0r = max(cyc - win, 0)
    y1r = min(cyc + win + 1, h)
    x0r = max(cxc - win, 0)
    x1r = min(cxc + win + 1, w)
    sub = img_hu[y0r:y1r, x0r:x1r]
    py, px = np.unravel_index(np.argmax(sub), sub.shape)
    return (int(y0r + py), int(x0r + px))


# ============================================================================
# Metadata extraction (for the DICOM Info dialog).
# ============================================================================

# Ordered list of (label, extractor) pairs. Extractors take a pydicom Dataset
# and return a display string (or "—" if not found).

_PLACEHOLDER = "—"


def _fmt(val, fmt="{}", default=_PLACEHOLDER):
    """Format a value, gracefully handling None/missing."""
    if val is None or val == "":
        return default
    try:
        return fmt.format(val)
    except Exception:
        return str(val)


def _get(ds, name, default=None):
    """Safely read a DICOM element by keyword."""
    try:
        v = getattr(ds, name, None)
        if v is None:
            return default
        # MultiValue / Decimal types convert cleanly to float/str
        return v
    except Exception:
        return default


def _search_private(ds, *needles):
    """Best-effort scan of all elements (incl. private) for a keyword/name
    containing any of the given substrings. Returns the first hit's value."""
    needles_l = tuple(n.lower() for n in needles)
    for elem in ds:
        try:
            text = (str(elem.keyword) + " " + str(elem.name)).lower()
        except Exception:
            continue
        if any(n in text for n in needles_l):
            try:
                return elem.value
            except Exception:
                continue
    return None


def _extract_make(ds):
    return _fmt(_get(ds, "Manufacturer"))


def _extract_model(ds):
    return _fmt(_get(ds, "ManufacturerModelName"))


def _extract_location(ds):
    for kw in ("InstitutionName", "InstitutionAddress", "StationName"):
        v = _get(ds, kw)
        if v:
            return str(v)
    return _PLACEHOLDER


def _extract_date(ds):
    for kw in ("AcquisitionDate", "SeriesDate", "StudyDate", "ContentDate"):
        v = _get(ds, kw)
        if v and len(str(v)) == 8:
            s = str(v)
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return _PLACEHOLDER


def _extract_collimation(ds):
    """Format as 'N × T = total' if possible (e.g. '128 × 0.60 = 76.8 mm')."""
    single = _get(ds, "SingleCollimationWidth")
    total = _get(ds, "TotalCollimationWidth")
    try:
        if single and total:
            s, t = float(single), float(total)
            n = int(round(t / s)) if s > 0 else None
            if n:
                return f"{n} × {s:.2f} = {t:.1f}"
            return f"{t:.1f}"
        if total:
            return f"{float(total):.1f}"
        if single:
            return f"{float(single):.2f}"
    except (ValueError, TypeError):
        pass
    return _PLACEHOLDER


def _extract_kvp(ds):
    v = _get(ds, "KVP")
    try:
        return f"{float(v):.0f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return str(v) if v else _PLACEHOLDER


def _extract_qrm(ds):
    """Quality Reference mAs — Siemens private. Try keywords first; if nothing,
    look for common element names and finally for a free-text mention."""
    # Try a few candidate keyword substrings
    v = _search_private(
        ds, "QRM", "QualityRef", "ReferenceMAs", "RefMAs", "ReferencemAs"
    )
    if v is not None:
        try:
            return f"{float(v):.0f}"
        except (ValueError, TypeError):
            return str(v)
    # Sometimes embedded in image comments (rare but possible)
    for kw in ("AcquisitionComments", "ImageComments", "DerivationDescription"):
        s = _get(ds, kw)
        if s and "ref" in str(s).lower() and "mas" in str(s).lower():
            return str(s)
    return _PLACEHOLDER


def _extract_effective_mas(ds):
    """Effective mAs. Prefer ExposureInmAs (0018,9332); fall back to
    XRayTubeCurrent × ExposureTime / 1000, or look in privates."""
    v = _get(ds, "ExposureInmAs")
    if v is not None:
        try:
            return f"{float(v):.1f}"
        except (ValueError, TypeError):
            return str(v)

    ma = _get(ds, "XRayTubeCurrent")
    et = _get(ds, "ExposureTime")  # in ms per DICOM standard
    if ma is not None and et is not None:
        try:
            return f"{float(ma) * float(et) / 1000.0:.1f}"
        except (ValueError, TypeError):
            pass

    # Some Siemens private tag may carry it explicitly
    v = _search_private(ds, "EffectivemAs", "EffectiveMAs", "Effective mAs")
    if v is not None:
        try:
            return f"{float(v):.1f}"
        except (ValueError, TypeError):
            return str(v)

    # Last resort: raw Exposure (mAs*s on some scanners — flag it)
    v = _get(ds, "Exposure")
    if v is not None:
        try:
            return f"{float(v):.1f} (Exposure tag)"
        except (ValueError, TypeError):
            return str(v)
    return _PLACEHOLDER


def _extract_kernel(ds):
    v = _get(ds, "ConvolutionKernel")
    if v is None:
        return _PLACEHOLDER
    # MultiValue → join
    try:
        return (
            ", ".join(str(x) for x in v)
            if hasattr(v, "__iter__") and not isinstance(v, str)
            else str(v)
        )
    except Exception:
        return str(v)


def _extract_recon_fov(ds):
    v = _get(ds, "ReconstructionDiameter")  # mm
    try:
        return f"{float(v):.1f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return str(v) if v else _PLACEHOLDER


def _extract_pixel_size(ds):
    v = _get(ds, "PixelSpacing")
    try:
        row, col = float(v[0]), float(v[1])
        return f"{row:.4f} × {col:.4f}"
    except (TypeError, ValueError, IndexError):
        return _PLACEHOLDER


def _extract_slice_thickness(ds):
    v = _get(ds, "SliceThickness")
    try:
        return f"{float(v):.2f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return _PLACEHOLDER


def _extract_slice_interval(ds):
    v = _get(ds, "SpacingBetweenSlices")
    try:
        return f"{float(v):.2f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return _PLACEHOLDER


def _extract_rotation_time(ds):
    """RevolutionTime is in seconds; convert to ms."""
    v = _get(ds, "RevolutionTime")
    if v is None:
        # Some scanners store this under non-standard keywords
        v = _search_private(ds, "RotationTime", "GantryRotationTime")
    if v is None:
        return _PLACEHOLDER
    try:
        f = float(v)
        # If it looks like it's already in ms, don't double-convert
        return f"{f * 1000:.0f}" if f < 10 else f"{f:.0f}"
    except (ValueError, TypeError):
        return str(v)


def _extract_pitch(ds):
    v = _get(ds, "SpiralPitchFactor")
    try:
        return f"{float(v):.3f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return _PLACEHOLDER


def _extract_ctdivol(ds):
    v = _get(ds, "CTDIvol")
    try:
        return f"{float(v):.2f}" if v is not None else _PLACEHOLDER
    except (ValueError, TypeError):
        return _PLACEHOLDER


# Order matches the user's requested layout.
METADATA_FIELDS = [
    ("Make", _extract_make),
    ("Model", _extract_model),
    ("Location", _extract_location),
    ("Date", _extract_date),
    ("Collimation (mm)", _extract_collimation),
    ("Tube Potential (kV)", _extract_kvp),
    ("QRM", _extract_qrm),
    ("Effective mAs", _extract_effective_mas),
    ("Kernel", _extract_kernel),
    ("Recon FOV (mm)", _extract_recon_fov),
    ("Pixel Size (mm)", _extract_pixel_size),
    ("Slice Thickness (mm)", _extract_slice_thickness),
    ("Slice Interval (mm)", _extract_slice_interval),
    ("Rotation Time (ms)", _extract_rotation_time),
    ("Helical Pitch", _extract_pitch),
    ("CTDIvol (mGy)", _extract_ctdivol),
]


def extract_metadata(ds_or_path) -> list:
    """Return a list of (label, value) pairs in the requested display order.

    Accepts either a pydicom Dataset or a path to a DICOM file.
    """
    if isinstance(ds_or_path, (str, os.PathLike)):
        ds = pydicom.dcmread(str(ds_or_path), stop_before_pixels=True)
    else:
        ds = ds_or_path
    return [(label, fn(ds)) for label, fn in METADATA_FIELDS]


def read_full_dicom_text(path: str) -> str:
    """Return the full DICOM dataset as a human-readable string (no pixels)."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return str(ds)
