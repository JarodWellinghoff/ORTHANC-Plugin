#!/usr/bin/env python
# coding: utf-8

import json
import re


import numpy as np
from numpy import random
from typing import Any
from scipy.fft import (
    rfft,
    fft2,
    ifft2,
    fftfreq,
    rfftfreq,
    fftshift,
    ifftshift,
    next_fast_len,
)
from scipy.ndimage import minimum_filter, maximum_filter
from numpy.linalg import inv
import os
from pydicom import dcmread
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.stats import linregress
from skimage import feature, measure
from skimage import exposure, util
import math
import pandas as pd
import scipy.io as sio
from scipy.signal import convolve2d
from skimage.transform import resize
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.interpolate import (
    interp1d,
    make_interp_spline,
    griddata,
)
from scipy.ndimage import binary_dilation
import gc
import pathlib
from matplotlib.widgets import Slider
from sklearn.metrics import mean_absolute_percentage_error
from dataclasses import dataclass
from typing import Optional
from scipy.interpolate import RegularGridInterpolator as RGI
from skimage.util import crop

KERNEL_RE = re.compile(r"^([A-Za-z]+?)(\d+)([A-Za-z]*)(?:_(\d+))?$")


@dataclass
class SmoothnessResult:
    high_freq_ratio: float  # fraction of power above cutoff
    spectral_centroid: float  # normalised power-weighted mean frequency
    sobolev_norm_H1: float  # H^1 semi-norm (unnormalised)
    spectral_slope: float  # power-law decay exponent s (higher = smoother)
    smoothness_score: float  # composite score in [0, 1]

    def __str__(self) -> str:
        return (
            f"Smoothness report\n"
            f"  High-freq ratio    : {self.high_freq_ratio:.4f}   (low is smooth)\n"
            f"  Spectral centroid  : {self.spectral_centroid:.4f}   (low is smooth)\n"
            f"  Sobolev H1 norm    : {self.sobolev_norm_H1:.4e}  (low is smooth)\n"
            f"  Spectral slope     : {self.spectral_slope:.4f}   (high is smooth)\n"
            f"  Smoothness score   : {self.smoothness_score:.4f}   [0=rough, 1=smooth]"
        )

    # ---------------------------------------------------------------------------
    # Core function
    # ---------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def scroll_plot_3d(
    data,
    rois=None,
    side=10,
    depth=1,
    colors=None,
    labels=None,
    titles=None,
    start_slice=None,
    layout=None,
    block=None,
):
    """Scroll through slices of one or more 3D arrays using the mouse wheel.

    Parameters
    ----------
    data : (S, R, C) array or sequence of such arrays
        One volume, or several to view side by side. All volumes must share
        the same number of slices S; scrolling is synced across them.
    rois : sequence, optional
        Square ROI centers (slice, row, col). Either:
          * a flat list of centers -> drawn on every panel, or
          * a list of length n_panels, each a sequence of centers (or None)
            -> per-panel ROIs.
    side, depth, colors, labels :
        ROI styling, as before. Each may be a scalar/sequence applied to all
        panels, or a list of length n_panels for per-panel control.
    titles : sequence of str, optional
        Per-panel titles.
    start_slice : int, optional
        First slice shown. Defaults to the middle slice.
    layout : (nrows, ncols), optional
        Subplot grid. Defaults to a single row.
    """

    # --- normalize the volumes -------------------------------------------
    arrays = [data] if isinstance(data, np.ndarray) else list(data)
    n_panels = len(arrays)

    n_slices = arrays[0].shape[0]
    for k, a in enumerate(arrays):
        if a.shape[0] != n_slices:
            raise ValueError(
                f"all volumes must have the same number of slices; "
                f"volume 0 has {n_slices}, volume {k} has {a.shape[0]}"
            )

    half_depth = max(depth - 1, 0) // 2

    # --- broadcast per-panel parameters ----------------------------------
    def _is_center(x):
        return (
            isinstance(x, (tuple, list, np.ndarray))
            and len(x) == 3
            and all(np.isscalar(v) for v in x)
        )

    def _per_panel(value):
        """Length-n_panels list: use as-is if it already matches, else broadcast."""
        if isinstance(value, (list, tuple)) and len(value) == n_panels:
            return list(value)
        return [value] * n_panels

    if rois is None or len(rois) == 0:
        panel_rois = [[] for _ in range(n_panels)]
    elif _is_center(rois[0]):
        panel_rois = [list(rois) for _ in range(n_panels)]  # broadcast
    else:
        panel_rois = [list(r) if r is not None else [] for r in rois]  # per-panel

    panel_side = _per_panel(side)
    panel_colors = _per_panel(colors)
    panel_labels = _per_panel(labels)
    panel_titles = titles if titles is not None else [None] * n_panels

    palette = ["red", "lime", "cyan", "yellow", "magenta", "orange"]

    def _build_rois(ax, centers, side_spec, color_spec, label_spec):
        """Create (rect, text, slice) entries for one panel; returns list."""
        m = len(centers)
        sides = [side_spec] * m if np.isscalar(side_spec) else list(side_spec)

        if color_spec is None:
            cols = [palette[i % len(palette)] for i in range(m)]
        elif isinstance(color_spec, str):
            cols = [color_spec] * m
        else:
            cols = list(color_spec)

        entries = []
        for i, (s, r, c) in enumerate(centers):
            hs = sides[i] / 2.0
            rect = Rectangle(
                (c - hs, r - hs),
                sides[i],
                sides[i],
                fill=False,
                edgecolor=cols[i],
                linewidth=1.5,
            )
            rect.set_visible(False)
            ax.add_patch(rect)

            txt = None
            if label_spec is not None:
                txt = ax.text(
                    c - hs,
                    r - hs - 1,
                    str(label_spec[i]),
                    color=cols[i],
                    fontsize=8,
                    va="bottom",
                )
                txt.set_visible(False)
            entries.append((rect, txt, s))
        return entries

    class _Panel:
        __slots__ = ("im", "X", "rois")

    class SyncedScroller:
        def __init__(self, fig, axes):
            self.fig = fig
            self.n_slices = n_slices
            self.half_depth = half_depth
            self.ind = n_slices // 2 if start_slice is None else start_slice

            self.panels = []
            for k, ax in enumerate(axes):
                p = _Panel()
                p.X = arrays[k]
                p.im = ax.imshow(p.X[self.ind], cmap="gray")
                p.rois = _build_rois(
                    ax,
                    panel_rois[k],
                    panel_side[k],
                    panel_colors[k],
                    panel_labels[k],
                )
                if panel_titles[k] is not None:
                    ax.set_title(panel_titles[k])
                self.panels.append(p)

            self.update()

        def onscroll(self, event):
            step = 1 if event.button == "up" else -1
            self.ind = (self.ind + step) % self.n_slices
            self.update()

        def update(self):
            for p in self.panels:
                p.im.set_data(p.X[self.ind])
                for rect, txt, s in p.rois:
                    on = abs(self.ind - s) <= self.half_depth
                    rect.set_visible(on)
                    if txt is not None:
                        txt.set_visible(on)
            self.fig.suptitle(
                f"slice {self.ind} / {self.n_slices - 1}    (scroll to navigate)"
            )
            self.fig.canvas.draw_idle()

    # --- figure / axes ---------------------------------------------------
    if layout is None:
        nrows, ncols = 1, n_panels
    else:
        nrows, ncols = layout

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax in axes[n_panels:]:  # hide unused cells
        ax.set_visible(False)

    scroller = SyncedScroller(fig, axes[:n_panels])
    fig.canvas.mpl_connect("scroll_event", scroller.onscroll)

    if block is None:
        block = not plt.isinteractive()
    plt.show(block=block)
    return scroller


def _parse_kernel(name):
    """Parse a Siemens kernel name -> (family, number, mode, iteration) or None."""
    s = re.sub(r"ext_?hu$", "", str(name).strip(), flags=re.IGNORECASE)  # drop extHU
    m = KERNEL_RE.match(s)
    if not m:
        return None
    fam, num, mode, it = m.groups()
    return (fam.lower(), int(num), (mode or "").lower(), int(it) if it else None)


def get_mtf_from_excel(kernel_file, manufacturer, model, kernel, verbose=True):
    df = pd.read_excel(kernel_file, header=0)
    for col in ("Manufacturer", "Model", "Kernel"):
        df[col] = df[col].astype(str).str.strip()
    # drop unparseable kernel names (catches the stray '0'/blank rows)
    df = df[df["Kernel"].apply(lambda k: _parse_kernel(k) is not None)].copy()

    man, mod, ker = str(manufacturer).strip(), str(model).strip(), str(kernel).strip()
    ker_norm = re.sub(
        r"ext_?hu$", "", ker, flags=re.IGNORECASE
    )  # normalized for name compare

    def _ret(row, how):
        if verbose:
            print(
                f"[{how}] {row['Manufacturer']} {row['Model']} {row['Kernel']} "
                f"-> MTF50={float(row['MTF50']):.3f}, MTF10={float(row['MTF10']):.3f}"
            )
        return float(row["MTF50"]), float(row["MTF10"])

    same_man = df["Manufacturer"].str.casefold() == man.casefold()
    same_mod = df["Model"].str.casefold() == mod.casefold()
    same_name = (
        df["Kernel"].str.replace(r"ext_?hu$", "", regex=True, case=False).str.casefold()
        == ker_norm.casefold()
    )

    # Tier 1: exact manufacturer + model + kernel
    hit = df[same_man & same_mod & same_name]
    if len(hit):
        return _ret(hit.iloc[0], "exact")

    # Tier 2: exact kernel name, any model (same manufacturer)
    hit = df[same_man & same_name]
    if len(hit):
        return _ret(hit.iloc[0], "exact name, other model")

    target = _parse_kernel(ker_norm)
    if target is None:
        if verbose:
            print(f"Could not parse '{ker}'; using default.")
        return 0.434, 0.730
    t_fam, t_num, t_mode, _ = target

    # Tiers 3+: same family, nearest sharpness number; prefer same model, then matching mode
    for scope_name, mask in (
        ("same model", same_man & same_mod),
        ("same manufacturer", same_man),
        ("any", pd.Series(True, index=df.index)),
    ):
        pool = df[mask].copy()
        if not len(pool):
            continue
        parsed = pool["Kernel"].apply(_parse_kernel)
        pool["_fam"] = parsed.apply(lambda p: p[0])
        pool["_num"] = parsed.apply(lambda p: p[1])
        pool["_mode"] = parsed.apply(lambda p: p[2])
        pool = pool[pool["_fam"] == t_fam]  # family is a hard constraint
        if not len(pool):
            continue
        pool["_score"] = (pool["_num"] - t_num).abs() + (pool["_mode"] != t_mode) * 0.5
        best = pool.sort_values(["_score", "_num"]).iloc[0]
        return _ret(best, f"closest ({scope_name}, family '{t_fam}')")

    if verbose:
        print(f"No same-family match for '{ker}'; using default.")
    return 0.434, 0.730


def normal_round(n):
    """Round a number to the nearest integer, with 0.5 rounding up."""
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def poly2fit(x, y, z, n):
    """Fit a 2D polynomial of degree n to data z at coordinates (x, y)."""
    if x.shape != y.shape or x.shape != z.shape:
        print("X, Y, and Z matrices must be the same size")
    x = x.T.flatten()
    y = y.T.flatten()
    z = z.T.flatten()
    n = n + 1
    k = 0
    A = np.zeros((x.shape[0], 6))
    i = n
    while i >= 1:
        for j in range(1, i + 1):
            temp1 = np.power(x, i - j)
            temp2 = np.power(y, j - 1)
            A[:, k] = np.multiply(temp1, temp2)
            k = k + 1
        i = i - 1
    p = np.linalg.lstsq(A, z, rcond=None)[0]
    return p


def subtractMean2D(im, method, psize):
    """Subtract a mean or polynomial background from a 2D image."""
    if method:
        FOV = np.zeros(2)
        im_sizeX, im_sizeY = im.shape
        FOV[0] = psize[0] * im_sizeX
        FOV[1] = psize[1] * im_sizeY
        x = np.arange(0, im_sizeX) * psize[0] - FOV[0] / 2
        y = np.arange(0, im_sizeY) * psize[1] - FOV[1] / 2
        X, Y = np.meshgrid(x, y)
        P = poly2fit(X, Y, im, 1)
        im = im - (P[0] * X + P[1] * Y + P[2])
    else:
        im = im - np.mean(im.flatten())
    return im


def NPS_statistics(nps1d, unit):
    """Calculate statistics of 1D NPS: average frequency, peak frequency, slope, and 10% frequency."""
    peakfrequencyIndex = np.where(nps1d == np.max(nps1d))[0][0]
    peakfrequency = peakfrequencyIndex * unit
    n = len(nps1d)
    Spatial_freq = np.arange(n) * unit
    p = nps1d / np.sum(nps1d)
    p = np.squeeze(p)
    Spatial_freq = np.squeeze(Spatial_freq)
    fav = np.sum(Spatial_freq * p)
    minfrequencyIndex = np.where(nps1d < 0.10 * np.max(nps1d))[0]
    freq = np.where(minfrequencyIndex > peakfrequencyIndex)[0][0]
    min10percent_frequency = minfrequencyIndex[freq] * unit
    npoint = 10
    k = np.polyfit(np.arange(npoint) * unit, nps1d[:npoint], 1)
    return fav, peakfrequency, k, min10percent_frequency


def ROI_to_NPS_Sum(roi_size, rois, dx, dy):
    """Compute 1D NPS from ROI images with optional 2× oversampling.

    - ROI_All shape: (ROI_size, ROI_size, num_rois)
    - Now performs 2× bilinear oversampling to double the frequency range.
    """

    # ====================== 2× OVERSAMPLING ======================
    oversample_factor = 2
    dx_os = dx / oversample_factor
    dy_os = dy / oversample_factor

    # Upsample all ROIs: (H, W, N) → (2H, 2W, N)
    rois = ndimage.zoom(
        rois, zoom=(1, oversample_factor, oversample_factor), order=1, mode="reflect"
    )

    # Update ROI_size to the new (oversampled) size
    roi_size = rois.shape[1]  # now even (e.g. 18, 22, etc.)
    # ============================================================

    # Size of the FFT padded array (8 times larger than ROI)
    nnn = next_fast_len(8 * max(20, roi_size))
    if nnn is None:
        raise ValueError("Failed to compute next fast length for FFT.")
    cc = int(np.rint(nnn / 2))
    unit = 1 / (dx_os * nnn)  # frequency step with oversampled pixel size

    oo = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.arange(cc).reshape((cc, 1))
    x_offset = radii * np.cos(oo)
    y_offset = radii * np.sin(oo)
    xi_all = np.rint(x_offset + cc - 1).astype(np.int32)
    yi_all = np.rint(y_offset + cc - 1).astype(np.int32)
    valid_mask = (xi_all >= 0) & (xi_all < nnn) & (yi_all >= 0) & (yi_all < nnn)

    NPS_1D_sum = np.zeros((cc, 1), dtype=np.float32)
    noise_level_sum = 0.0
    num_rois = rois.shape[0]

    for jj in range(num_rois):
        roi = rois[jj]

        # Subtract mean (important for NPS)
        psize = (dx_os, dy_os)
        roi = subtractMean2D(roi, 1, psize)

        noise_level_sum += np.std(roi)

        # 2D FFT
        roipad_fft = fftshift(np.abs(fft2(roi, s=(nnn, nnn))))  # type: ignore
        roipad_fft = roipad_fft**2 * dx_os * dy_os / (roi_size * roi_size)

        # Radial averaging
        polar1 = np.zeros((cc, 360), dtype=np.float32)
        polar1[valid_mask] = roipad_fft[xi_all[valid_mask], yi_all[valid_mask]]
        nps1d = np.mean(polar1, axis=1).reshape((cc, 1))
        NPS_1D_sum += nps1d

    Spatial_freq = np.arange(cc).astype(np.float32) * unit

    return Spatial_freq, NPS_1D_sum, noise_level_sum, unit


def Laguerre2D(order, a, b, cx, cy, X, Y):
    """Generate 2D Laguerre-Gauss channel filter."""
    val1 = np.zeros(X.size)
    ga = 2 * np.pi * ((X - cx) ** 2 / (a**2) + (Y - cy) ** 2 / (b**2))
    for jp in range(order + 1):
        val1 = val1 + (-1) ** jp * np.prod(np.linspace(1, order, order)) / (
            np.prod(np.linspace(1, jp, jp))
            * np.prod(np.linspace(1, order - jp, order - jp))
        ) * (ga**jp) / np.prod(np.linspace(1, jp, jp))
    channel_filter = np.exp(-ga / 2) * val1
    return channel_filter


def Gabor2D(fc, wd, theta, beta, cx, cy, X, Y):
    """Generate 2D Gabor channel filter."""
    channel_filter = np.exp(
        -4 * np.log(2) * ((X - cx) ** 2 + (Y - cy) ** 2) / (wd**2)
    ) * np.cos(
        2 * np.pi * fc * ((X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)) + beta
    )
    return channel_filter


def channel_selection(Chnl, inputArg1, inputArg2, inputArg3="0"):
    """Configure channel parameters for Laguerre-Gauss or Gabor filters."""
    if Chnl.Chnl_Toggle == "Laguerre-Gauss":
        if isinstance(inputArg1, int):
            Chnl.LG_order = inputArg1
        if isinstance(inputArg2, int):
            Chnl.LG_orien = inputArg2
    if Chnl.Chnl_Toggle == "Gabor":
        if isinstance(inputArg1, str):
            if inputArg1 == "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]":
                Chnl.Gabor_passband = np.transpose(
                    [
                        [1 / 64, 1 / 32],
                        [1 / 32, 1 / 16],
                        [1 / 16, 1 / 8],
                        [1 / 8, 1 / 4],
                    ]
                )
            if inputArg1 == "[[1/64,1/32], [1/32,1/16]]":
                Chnl.Gabor_passband = np.transpose([[1 / 64, 1 / 32], [1 / 32, 1 / 16]])
        if isinstance(inputArg2, str):
            if inputArg2 == "[0, pi/3, 2*pi/3]":
                Chnl.Gabor_theta = [0, np.pi / 3, 2 * np.pi / 3]
            if inputArg2 == "[0, pi/2]":
                Chnl.Gabor_theta = [0, np.pi / 2]
            if inputArg2 == "0":
                Chnl.Gabor_theta = [0]
        if isinstance(inputArg3, str):
            if inputArg3 == "[0, pi/2]":
                Chnl.Gabor_beta = [0, np.pi / 2]
            if inputArg3 == "0":
                Chnl.Gabor_beta = [0]
        Chnl.Gabor_fc = np.mean(Chnl.Gabor_passband, axis=0)
        Chnl.Gabor_wd = (
            4
            * np.log(2)
            / (np.pi * (Chnl.Gabor_passband[1, :] - Chnl.Gabor_passband[0, :]))
        )
    return Chnl


def ChannelMatrix_Generation(Chnl, roiSize_xy):
    """Generate channel matrix for CHO analysis based on filter type."""
    x = np.linspace(1, roiSize_xy, roiSize_xy) - (roiSize_xy + 1) / 2
    y = x
    X, Y = np.meshgrid(x, y)
    X = X.T.reshape(-1)
    Y = Y.T.reshape(-1)
    channelMatrix = []
    if Chnl.Chnl_Toggle == "Laguerre-Gauss":
        LG_ORIEN, LG_ORDER = np.meshgrid(
            np.arange(Chnl.LG_orien) + 1, np.arange(Chnl.LG_order) + 1
        )
        LG_ORIEN = LG_ORIEN.T.reshape(-1)
        LG_ORDER = LG_ORDER.T.reshape(-1)
        A = 5 * np.ones(LG_ORDER.size)
        B = 14 * np.ones(LG_ORDER.size)
        A[LG_ORIEN == 1] = 8
        B[LG_ORIEN == 1] = 8
        A[LG_ORIEN == 2] = 14
        B[LG_ORIEN == 2] = 5
        channelMatrix = np.zeros(
            (roiSize_xy * roiSize_xy, Chnl.LG_order * Chnl.LG_orien)
        )
        for ii in range(LG_ORDER.size):
            channelMatrix[:, ii] = Laguerre2D(LG_ORDER[ii], A[ii], B[ii], 0, 0, X, Y)
    elif Chnl.Chnl_Toggle == "Gabor":
        Gabor_THETA, Gabor_FC, Gabor_BETA = np.meshgrid(
            Chnl.Gabor_theta, Chnl.Gabor_fc, Chnl.Gabor_beta
        )
        Gabor_wd_matrix = np.zeros((Chnl.Gabor_wd.size, 1, 1))
        Gabor_wd_matrix[:, 0, 0] = Chnl.Gabor_wd
        Gabor_WD = np.tile(Gabor_wd_matrix, (1, Gabor_FC.shape[1], Gabor_FC.shape[2]))
        Gabor_FC = Gabor_FC.T.reshape(-1)
        Gabor_WD = Gabor_WD.T.reshape(-1)
        Gabor_THETA = Gabor_THETA.T.reshape(-1)
        Gabor_BETA = Gabor_BETA.T.reshape(-1)
        channelMatrix = np.zeros(
            (roiSize_xy * roiSize_xy, Gabor_FC.size), dtype=np.float32
        )
        for ii in range(Gabor_FC.size):
            channelMatrix[:, ii] = Gabor2D(
                Gabor_FC[ii], Gabor_WD[ii], Gabor_THETA[ii], Gabor_BETA[ii], 0, 0, X, Y
            )
    return channelMatrix


def CHO_patient_with_resampling(
    sig_true, bkg_ordered, channelMatrix, internalNoise, Resampling_method
):
    """Compute detectability (d') using Channelized Hotelling Observer with resampling."""

    N_total_bkg = bkg_ordered.shape[1]
    try:
        rand_scanSelect_bkg = np.arange(N_total_bkg)
        if Resampling_method == "Bootstrap":
            rand_scanSelect_bkg = np.random.randint(N_total_bkg, size=N_total_bkg)
        elif Resampling_method == "Shuffle":
            rand_scanSelect_bkg = np.random.permutation(N_total_bkg)
        sig = sig_true
        bkg = bkg_ordered[:, rand_scanSelect_bkg].astype(float)
        if (
            sig.shape[0] != channelMatrix.shape[0]
            or bkg.shape[0] != channelMatrix.shape[0]
        ):
            print("Numbers of pixels do not match.")
        vN = channelMatrix.T @ bkg
        sbar = np.squeeze(sig)
        S = np.cov(vN.T, rowvar=False)
        wCh = np.linalg.inv(S) @ channelMatrix.T @ sbar
        temp = channelMatrix.T @ sbar
        tsN_Mean = wCh.T @ temp
        tN0 = wCh.T @ vN
        tN = tN0 + np.random.randn(tN0.size) * internalNoise * np.std(tN0)
        dp = np.sqrt((tsN_Mean**2) / np.var(tN))
    except Exception as e:
        dp = np.nan  # Return NaN if an error occurs
    return dp


def interpolate_grid(X0, Y0, L0, XX, YY, method):
    """Interpolate 2D data to a new grid using specified method."""
    X0 = np.asarray(X0)
    Y0 = np.asarray(Y0)
    L0 = np.asarray(L0)
    XX = np.asarray(XX)
    YY = np.asarray(YY)
    grid_points = (X0.flatten(), Y0.flatten())
    values = L0.flatten()
    Lesion = griddata(grid_points, values, (XX, YY), method=method)
    return Lesion


def max_consecutive_ones_2d(arr):
    """Longest True run in a 2D array, scanning rows and columns. Scalar."""

    def max_consecutive_true(arr, axis):
        """Length of the longest consecutive run of True along `axis`, per line.

        Fully vectorized. Returns an array with `axis` removed.

        Trick: mark every False with its own index and every True with -1, then
        np.maximum.accumulate carries forward the index of the most recent False
        (-1 if none seen yet). `idx - last_false` is the run length ending at each
        cell; the max along the axis is the longest run.
        """
        arr = np.moveaxis(np.asarray(arr, dtype=bool), axis, -1)
        n = arr.shape[-1]
        idx = np.arange(n).reshape((1,) * (arr.ndim - 1) + (n,))
        last_false = np.maximum.accumulate(np.where(arr, -1, idx), axis=-1)
        counts = np.where(arr, idx - last_false, 0)
        return counts.max(axis=-1)

    r = max_consecutive_true(arr, axis=1).max()  # horizontal (within rows)
    c = max_consecutive_true(arr, axis=0).max()  # vertical   (within columns)
    return np.array([r, c])


# --------------------------------------------------------------
#  NEW: Damped-cosine PSF (replaces super-Gaussian)
# --------------------------------------------------------------
def simulate_psf_damped_cosine(mtf50, mtf10=np.nan, size=513, wire_fov_mm=50.0):
    """
    Damped-cosine PSF:   exp(-r/τ) * cos(k·r)
    • mtf10 is None → pure Gaussian (k=0) with analytic τ from mtf50
    • mtf10 given   → optimise τ and k to hit MTF(mtf50)=0.5 and MTF(mtf10)=0.1
    """
    working_size = max(
        int(2 ** np.ceil(np.log2(wire_fov_mm * np.nanmax([mtf50, mtf10]) * 2)) + 1),
        size,
    )
    dx = wire_fov_mm / working_size
    if np.isnan(mtf10):
        sigma = np.sqrt(np.log(2) / (2 * np.pi**2)) / mtf50  # physical units

        # PSF (image domain)
        c = np.arange(working_size) - working_size // 2
        xx, yy = np.meshgrid(c * dx, c * dx)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    else:
        n = np.log(np.log(10) / np.log(2)) / np.log(mtf10 / mtf50)
        fc = mtf50 / np.log(2) ** (1.0 / n)

        # Centered 2D radial frequency grid
        f = fftshift(fftfreq(working_size, d=dx))
        fx, fy = np.meshgrid(f, f)
        fr = np.hypot(fx, fy)

        mtf2d = np.exp(-((fr / fc) ** n))

        # MTF is real, even, non-negative -> PSF is real, even, non-negative*
        # *non-negativity not guaranteed for arbitrary MTF shapes; check at runtime
        psf = np.real(fftshift(ifft2(ifftshift(mtf2d))))

    psf /= psf.sum()
    return psf, dx


def compute_presampling_mtf(psf, unit_dx, maxfreq=100.0, normal_round=round):
    """
    Compute radial 1D MTF using fixed 2048-point FFT.
    """
    psf = np.asarray(psf, dtype=np.float64)

    # 1. Normalize PSF so sum = 1
    psf_norm = psf / np.sum(psf)

    # 2. Fixed 2048-point FFT size
    N_fft = 2048

    # 3. Center the PSF and pad to exactly 2048 x 2048
    nsize = psf.shape[0]
    pad_total = N_fft - nsize
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    psf_padded = np.pad(
        psf_norm,
        pad_width=((pad_left, pad_right), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )

    # 4. Compute 2D FFT with proper centering (using np.fft)
    psf_shift = ifftshift(psf_padded)
    FT = fft2(psf_shift)
    mtf2d = np.abs(FT)
    mtf2d = fftshift(mtf2d)
    mtf2d /= mtf2d.max()  # MTF(0) = 1.0

    # 5. Frequency axis
    df = 1.0 / (unit_dx * N_fft)  # frequency step in cycles/mm
    freq_full = fftshift(fftfreq(N_fft, d=unit_dx))
    freq_pos = freq_full[freq_full >= 0]  # only positive frequencies

    # 6. Radial averaging
    centre = N_fft // 2
    deg = 360
    oo = 2 * np.pi * np.linspace(0, deg - 1, deg) / deg

    # Limit to maxfreq
    max_bin = int(maxfreq / df) + 1
    n_freq = min(max_bin, len(freq_pos))

    mtf_polar = np.zeros((n_freq, deg))
    freq = freq_pos[:n_freq]

    for ii in range(n_freq):
        r = float(ii)
        for jj in range(deg):
            xx = r * np.cos(oo[jj])
            yy = r * np.sin(oo[jj])
            xi = normal_round(xx + centre)
            yi = normal_round(yy + centre)

            if 0 <= xi < N_fft and 0 <= yi < N_fft:
                mtf_polar[ii, jj] = mtf2d[yi, xi]

    # Final 1D radial MTF
    mtf = np.mean(mtf_polar, axis=1)

    # 7. Calculate MTF50, MTF10, MTF2
    def get_mtf_freq(target):
        idx = np.where(mtf <= target)[0]
        return freq[idx[0]] if len(idx) > 0 else np.nan

    mtf_p50 = get_mtf_freq(0.51)
    mtf_p10 = get_mtf_freq(0.105)
    mtf_p02 = get_mtf_freq(0.02)  # MTF at 2%

    mtf_eval = np.array([mtf_p50, mtf_p10, mtf_p02])

    return freq, mtf, mtf_eval


def interpolate_rgi(img, scale):
    h, w = img.shape

    x = np.arange(0, w)
    y = np.arange(0, h)

    xnew = np.arange(0, w, 1 / scale)
    ynew = np.arange(0, h, 1 / scale)

    xxnew, yynew = np.meshgrid(xnew, ynew, indexing="ij", sparse=True)

    r = RGI(
        (x, y),
        img.T,
        method="linear",
        bounds_error=False,
        fill_value=img[0, 0],
    )

    return r((xxnew, yynew)).T


def roi_pad(image, target_size=80):
    """Pad the center of an image to the target size."""
    pad_width = np.tile(target_size - np.array(image.shape[0:2]), (2, 1)).T / 2
    pad_width[:, 0] = np.floor(pad_width[:, 0])
    pad_width[:, 1] = np.ceil(pad_width[:, 1])

    return np.pad(image, pad_width.astype(int))


def roi_crop(image, target_size):
    """Crop the center of an image to the target size."""

    crop_width = np.tile(np.array(image.shape[0:2]) - target_size, (2, 1)).T / 2
    crop_width[:, 0] = np.floor(crop_width[:, 0])
    crop_width[:, 1] = np.ceil(crop_width[:, 1])
    return crop(image, crop_width.astype(int))


def ahe(images):
    images = images.transpose()
    kernel_size = np.array(
        [
            images.shape[0] // 5,
            images.shape[1] // 5,
            images.shape[2] // 2,
        ]
    )
    images = np.clip(images, np.percentile(images, 5), np.percentile(images, 95))
    images_min = images.min()
    images_max = images.max()
    images = (images - images_min) / (images_max - images_min)
    images = exposure.equalize_adapthist(images, kernel_size=kernel_size)
    images = images * (images_max - images_min) + images_min
    return images.transpose()


def prepare_Lesion_sig(
    lesion_file,
    contrast_target,
    width_mm_target,
    roi_size_mm,
    psf_sim,
    dx_psf,
    pixel_spacing_patient,
):
    """
    Prepare lesion signal (scale → contrast → PSF convolution).

    Now always simulates PSF using MTF50/MTF10 from Recon_Kernels.xlsx.
    """
    # --------------------------------------------------------------
    # 2. Load / create lesion VOI
    # --------------------------------------------------------------
    mat_data = sio.loadmat(lesion_file)
    patient_data = mat_data["Patient"]
    try:
        lesion = patient_data["Lesion"][0][0]["VOI"][0][0]
        lesion_mask = patient_data["Lesion"][0][0]["LesionMask"][0][0]
        lesion_dcm_info = patient_data["DicomHeader"][0][0]
        lesion_dcm_pixel_spacing = lesion_dcm_info["PixelSpacing"][0][0].flatten()
    except IndexError:
        lesion = patient_data[0][0][0][0]["Lesion"][0][0]["VOI"]
        lesion_mask = patient_data[0][0][0][0]["Lesion"][0][0]["LesionMask"]
        lesion_dcm_info = patient_data[0][0][0][0]["DicomHeader"][0][0]
        lesion_dcm_pixel_spacing = lesion_dcm_info["PixelSpacing"][0]

    mid_slice = round(lesion.shape[-1] / 2)
    lesion = lesion[:, :, mid_slice]
    mask_slice = lesion_mask[:, :, mid_slice].astype(bool)

    # --------------------------------------------------------------
    # 3. Scale lesion to target physical size
    # --------------------------------------------------------------
    lesion_pixels_width_input = max_consecutive_ones_2d(mask_slice)
    #! Would median be more robust than mean here?
    lesion_median_hu = np.median(lesion[mask_slice])
    background_median_hu = np.median(lesion[~mask_slice]) if np.any(~mask_slice) else 0

    if contrast_target is None:
        contrast_target = lesion_median_hu - background_median_hu
    if width_mm_target is None:
        lesion_mm_width_input = lesion_pixels_width_input * lesion_dcm_pixel_spacing
        width_mm_target = lesion_mm_width_input.max()
    if roi_size_mm is None:
        # A linear fit to the ROI size vs lesion width (mm) from our standard 3 lesion sizes
        roi_size_mm = width_mm_target * 0.833 + 11.67

    roi_size = int(roi_size_mm / pixel_spacing_patient) | 1
    lesion_pixels_width_input = lesion_pixels_width_input.max()

    lesion = lesion - lesion_median_hu + contrast_target
    lesion[~mask_slice] = 0

    lesion_pixels_width_target = width_mm_target / pixel_spacing_patient
    scale = lesion_pixels_width_target / lesion_pixels_width_input

    Lesion = interpolate_rgi(lesion, scale)

    # Compute presampling MTF (for verification)
    freq, mtf, _ = compute_presampling_mtf(psf_sim, dx_psf)

    # Resample simulated PSF to patient pixel spacing
    scaling_factor = dx_psf / pixel_spacing_patient
    PSF_end = interpolate_rgi(psf_sim, scaling_factor)
    PSF_end /= PSF_end.sum()  # final normalization

    # --------------------------------------------------------------
    # 5. Convolution & crop to ROI
    # --------------------------------------------------------------

    Lesion_ext = roi_pad(Lesion)
    Lesion_conv = convolve2d(Lesion_ext, PSF_end, mode="full")
    Lesion_conv_end = roi_crop(Lesion_conv, roi_size)

    return (
        Lesion_conv_end,
        freq,
        mtf,
        roi_size,
        contrast_target,
        width_mm_target,
        roi_size_mm,
    )


def integrate_image(image, window_size=[3, 3], _mode="edge"):
    """
    Compute integral image for fast mean calculations.
    circular -> wrap
    replicate -> edge
    symmetric -> symmetric
    constant -> constant (default)
    """
    window_size = np.tile(window_size, (2, 1)) // 2
    if len(image.shape) == 3:
        win = window_size
        window_size = np.zeros((3, 2), dtype=int)
        window_size[1:, :] = win
        return np.pad(image, window_size, mode=_mode).cumsum(axis=1).cumsum(axis=2)
    elif len(image.shape) != 2:
        raise ValueError("Input image must be 2D or 3D.")
    return np.pad(image, window_size, mode=_mode).cumsum(axis=0).cumsum(axis=1)


def calculate_std_dev(
    intel_images,
    intel_images_square,
    intel_edge,
    roi_size,
    padding_size,
    init_shape,
    thr1,
    thr2,
    corrected=False,
    min_acceptable_std=0.8,  # Recommended: 0.8 ~ 1.2 HU
):
    """Calculate standard deviation map for ROI selection."""
    roi_size_adj = int(roi_size + (roi_size % 2 != 0))
    slice_n, side, _ = intel_images.shape
    padding_a = round((padding_size - roi_size_adj) / 2)
    padding_b = padding_a + roi_size_adj
    padding_c = side - padding_a
    padding_d = padding_c - roi_size_adj

    roi_size_half = roi_size // 2
    min_acceptable_variance = min_acceptable_std**2

    if corrected:
        normalization_factor = roi_size_adj**2 / (roi_size_adj**2 - 1)
    else:
        normalization_factor = 1

    STD_all = np.zeros(init_shape, dtype=np.float32)

    def get_map(t):
        t1 = t[padding_b:padding_c, padding_b:padding_c]
        t2 = t[padding_a:padding_d, padding_a:padding_d]
        t3 = t[padding_b:padding_c, padding_a:padding_d]
        t4 = t[padding_a:padding_d, padding_b:padding_c]
        _map = (t1 + t2 - t3 - t4) / (roi_size_adj**2)
        return _map

    for slice_i in range(slice_n):
        mean_map = get_map(intel_images[slice_i])
        mean_square = get_map(intel_images_square[slice_i])
        edge_impact = get_map(intel_edge[slice_i])

        Mask_mean = (mean_map < thr1) | (mean_map > thr2)

        # Adaptive edge exclusion based on distribution
        valid_edge = edge_impact[~np.isnan(edge_impact)]
        if len(valid_edge) > 20:
            edge_threshold = np.percentile(valid_edge, 92)  # top ~8%
        else:
            edge_threshold = 0.05

        Mask_edge = (edge_impact >= edge_threshold).astype(np.float32)

        Mask = np.logical_or(Mask_mean, Mask_edge > 0)
        variance = np.maximum(mean_square - mean_map**2, 0.0)
        variance[variance < min_acceptable_variance] = np.nan

        SD_Map = np.sqrt(variance * normalization_factor)
        SD_Map[Mask] = np.nan
        SD_Map = SD_Map[roi_size_half:-roi_size_half, roi_size_half:-roi_size_half]
        SD_Map = np.pad(SD_Map, roi_size_half, constant_values=np.nan)

        STD_all[slice_i, :, :] = SD_Map
    return STD_all


def extract_ROIs(std_map, sd_thresh, roi_size_half, images, group=False):
    """Extract ROIs with low standard deviation for NPS and CHO analysis."""
    h = roi_size_half
    slices, rows, cols = images.shape
    size = (1, 4 * h, 4 * h)  # flat across slices -> slices stay independent

    r_idx, c_idx = np.mgrid[:rows, :cols]
    dist = np.minimum.reduce([r_idx, c_idx, rows - 1 - r_idx, cols - 1 - c_idx])
    if isinstance(sd_thresh, (np.ndarray)) and len(sd_thresh) == images.shape[0]:
        valid = (
            (~np.isnan(std_map))
            & (std_map < sd_thresh[:, None, None])
            & (dist[None, :, :] > h)
        )
    else:
        valid = (~np.isnan(std_map)) & (std_map < sd_thresh) & (dist[None, :, :] > h)

    work = np.where(valid, std_map, np.inf)
    selected = np.zeros(std_map.shape, dtype=bool)
    available = valid.copy()
    while available.any():
        local_min = minimum_filter(work, size=size, mode="constant", cval=np.inf)
        is_min = available & (work <= local_min)  # equals window-min => local minimum
        selected |= is_min
        suppressed = maximum_filter(is_min, size=size, mode="constant", cval=False)
        available &= ~suppressed
        work = np.where(available, std_map, np.inf)

    ss, rr, cc = np.where(selected)
    order = np.argsort(std_map[ss, rr, cc], kind="stable")
    rois = [
        images[s, r - h : r + h + 1, c - h : c + h + 1]
        for s, r, c in zip(ss[order], rr[order], cc[order])
    ]
    rois = np.array(rois)
    centers = np.column_stack((ss, rr, cc))
    centers = centers[order]
    # scroll_plot_3d(images, rois=centers, side=2 * roi_size_half + 1)
    total_nps = 0
    if rois.size > 0:
        total_nps = rois.shape[0]

    # ====================== FINAL SAFETY FILTER  ======================
    if total_nps > 0:
        # Compute actual std for each extracted ROI
        actual_std = np.std(rois, axis=(1, 2))

        # Keep only ROIs where actual std <= Thre_SD
        # accommodate the case where sd_thresh is an array of thresholds for each slice
        if isinstance(sd_thresh, (np.ndarray)) and len(sd_thresh) == images.shape[0]:
            slice_indices = centers[:, 0]
            good_mask = actual_std <= sd_thresh[slice_indices]
        else:
            good_mask = actual_std <= sd_thresh

        rois = rois[good_mask]
        centers = centers[good_mask]
        total_nps = rois.shape[0]

        # Quality report
        rejected = len(actual_std) - total_nps
        if rejected > 0:
            print(
                f"extract_ROIs: {rejected} ROIs rejected. "
                f"Final selected: {total_nps} ROIs"
            )

    if group:
        rois = [rois[centers[:, 0] == loc] for loc in range(images.shape[0])]
    return rois


def round_to_nearest_odd(x):
    """Round to nearest odd integer for ROI sizes."""
    rounded = np.round(x).astype(int)
    return np.where(
        rounded % 2 == 0, np.where(rounded > x, rounded - 1, rounded + 1), rounded
    )


def spline_interpolate(x, y, points=100):
    """Interpolate y values at new x positions using cubic spline."""
    cs = make_interp_spline(x, y)
    xs = np.linspace(x[0], x[-1], points)
    return xs, cs(xs)


def mean_rolling_windows(x, y):
    idx_windows = np.arange(2, (len(x) // 2) + 1, 2)

    rolling_x = [
        np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1) for w in idx_windows
    ]
    rolling_means = [
        np.lib.stride_tricks.sliding_window_view(y, w).mean(axis=1) for w in idx_windows
    ]
    rolling_stds = [
        (np.lib.stride_tricks.sliding_window_view(y, w).std(axis=1))
        for w in idx_windows
    ]
    mean_std = np.array([s.mean() for s in rolling_stds]).mean()

    idx_windows = np.arange(
        max(2, min(int(mean_std + 1), 5)),
        # max(5, min(np.ceil(len(x) * mean_std**2), len(x))),
        max(5, min(int(400 * mean_std), len(x))),
    )
    rolling_x = [
        np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1) for w in idx_windows
    ]
    rolling_means = [
        np.lib.stride_tricks.sliding_window_view(y, w).mean(axis=1) for w in idx_windows
    ]
    rolling_stds = [
        (np.lib.stride_tricks.sliding_window_view(y, w).std(axis=1))
        for w in idx_windows
    ]
    weights = [(1 / (s + 1)) ** 2 for s in rolling_stds]
    ys_interp = np.stack(
        [
            # np.interp(x, xs, ys, left=np.nan, right=np.nan)
            np.interp(x, xs, ys)
            for xs, ys in zip(rolling_x, rolling_means)
        ]
    )
    weights_interp = np.stack(
        [
            # np.interp(x, xs, ys, left=np.nan, right=np.nan)
            np.interp(x, xs, ys)
            for xs, ys in zip(rolling_x, weights)
        ]
    )

    y_avg = np.sum(ys_interp * weights_interp, axis=0) / weights_interp.sum(axis=0)
    return y_avg


def candidate_prep(x, y):
    if x[0] != 0:
        x = x[len(x) // 2 :]
    if y[0] != 1:
        y = y[len(y) // 2 :]

    dt = np.gradient(x)
    v = np.gradient(y) / dt
    a = np.gradient(v) / dt

    smoothed_y = mean_rolling_windows(x, y)
    smoothed_v = mean_rolling_windows(x, v)
    smoothed_a = mean_rolling_windows(x, a)

    return dict(
        x=x,
        smoothed_y=smoothed_y,
        smoothed_v=smoothed_v,
        smoothed_a=smoothed_a,
        v=v,
        a=a,
    )


def get_stable_idx(x, y):
    prep = candidate_prep(x, y)
    x = prep["x"]
    smoothed_y = prep["smoothed_y"]
    smoothed_v = prep["smoothed_v"]
    smoothed_a = prep["smoothed_a"]

    idx_windows = np.arange(1, len(x) // 5)
    maxes = np.array([smoothed_y, smoothed_v, smoothed_a]).max(axis=0)
    mins = np.array([smoothed_y, smoothed_v, smoothed_a]).min(axis=0)
    dists = maxes - mins
    rolling_stds = [
        np.lib.stride_tricks.sliding_window_view(dists, w).std(axis=1)
        for w in idx_windows
    ]
    means = np.array([rolling_std.mean() for rolling_std in rolling_stds])
    lens = np.array([len(rolling_std) for rolling_std in rolling_stds])
    tol = np.sum(means * (lens / lens.sum()))
    stabilized_indices = [
        np.where(rolling_std < tol)[0] for rolling_std in rolling_stds
    ]
    intersection_indices = sorted(
        list(set(stabilized_indices[0]).intersection(*stabilized_indices[1:]))
    )
    stable_idx = -1
    if len(intersection_indices) > 0:
        # Return the original index where stabilization window begins
        stable_idx = intersection_indices[0] + (idx_windows[-1] // 2)
    stable_x = x[stable_idx]

    return stable_x, stable_idx


def plot_lesion_signals(
    lesion_sig, freq, mtf, lesion_con_target, target_lesion_width, i
):

    stable_freq, stable_idx = get_stable_idx(freq, mtf)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, num=f"Presampling MTF - Lesion {i+1}", figsize=(12, 5)
    )
    show_freq = freq[:stable_idx]
    show_mtf = mtf[:stable_idx]
    if stable_idx <= 10:  # sanity check to avoid very small idx
        xs, cs_values = spline_interpolate(show_freq, show_mtf)
        ax1.plot(
            xs, cs_values, "b-", linewidth=2, label="Presampling MTF (Interpolated)"
        )
        ax1.plot(show_freq, show_mtf, "r--", linewidth=1, label="Presampling MTF")
    else:
        ax1.plot(show_freq, show_mtf, "r-", linewidth=2, label="Presampling MTF")
    show_lesion = np.round(lesion_sig)
    show_lesion = show_lesion + 127
    if show_lesion.max() > 255 or show_lesion.min() < 0:
        show_lesion = (
            (show_lesion - show_lesion.min())
            * 255
            / (show_lesion.max() - show_lesion.min())
        )
    # else:
    #     shift = (255 - show_lesion.max()) // 2
    #     show_lesion = show_lesion + shift
    print(
        show_lesion.min(),
        show_lesion.max(),
        (show_lesion.min() + show_lesion.max()) / 2,
        show_lesion[0, 0],
    )
    ax1.legend()
    ax2.imshow(
        show_lesion,
        cmap="gray",
        vmin=0,
        vmax=255,
    )

    ax1.set_title(f"Presampling MTF - Lesion {i+1}")
    ax1.set_xlabel("Frequency (cycles/mm)")
    ax1.set_ylabel("MTF")

    ax2.set_title(
        f"Lesion Signal (Contrast: {lesion_con_target} HU, Size: {target_lesion_width} mm)"
    )
    ax2.axis("off")

    fig.tight_layout()


def plot_CTDI_SSDE_with_image(slice_locations, CTDI_All, ssde_inc, coronal_image):
    fig, ax = plt.subplots(num="CTDI & SSDE", figsize=(10, 6))

    ax.plot(slice_locations, CTDI_All)
    ax.plot(slice_locations, ssde_inc)

    y_min, y_max = ax.get_ylim()

    x_min, x_max = slice_locations.min(), slice_locations.max()
    img_extent = [x_min, x_max, y_min, y_max]

    ax.imshow(
        coronal_image,
        extent=img_extent,
        aspect="auto",
        cmap="gray",
        vmin=-1024,
        vmax=400,
        origin="upper",
        zorder=0,
    )

    ax.set_xlabel("Z Location (mm)")
    ax.set_ylabel("CTDI & SSDE")
    ax.legend(["CTDI", "SSDE"])

    ax.tick_params(axis="y")

    fig.tight_layout()


def plot_Dw_with_image(slice_locations, dw, coronal_image):
    fig, ax = plt.subplots(num="Water Equivalent Diameter (Dw)", figsize=(10, 6))

    ax.plot(slice_locations, dw, "g-")

    y_min, y_max = ax.get_ylim()
    x_min, x_max = slice_locations.min(), slice_locations.max()

    img_extent = [x_min, x_max, y_min, y_max]

    ax.imshow(
        coronal_image,
        extent=img_extent,
        aspect="auto",
        cmap="gray",
        vmin=-1024,
        vmax=400,
        origin="upper",
        zorder=0,
    )

    ax.set_xlabel("Z Location (mm)")
    ax.set_ylabel("Dw (mm)")

    fig.tight_layout()


def plot_detectability_with_image(
    dps_location, Mean_loc_dps, coronal_image, slice_location
):
    fig, ax1 = plt.subplots(num="Detectability Index", figsize=(10, 6))
    # try:
    #     xs, cs_values = spline_interpolate(dps_location, Mean_loc_dps)
    #     ax1.plot(xs, cs_values, "b-", linewidth=2, label="DPS (Interpolated)")
    # except ValueError as e:
    #     pass
    ax1.plot(
        dps_location,
        Mean_loc_dps,
        "r--",
        linewidth=2,
        marker="o",
    )

    y_min, y_max = ax1.get_ylim()

    x_min, x_max = slice_location.min(), slice_location.max()
    img_extent = [x_min, x_max, y_min, y_max]

    ax1.imshow(
        coronal_image,
        extent=img_extent,
        aspect="auto",
        cmap="gray",
        vmin=-1024,
        vmax=400,
        origin="upper",
        zorder=0,
    )

    ax1.set_xlabel("Z Location (mm)")
    ax1.set_ylabel("Detectability Index")

    fig.tight_layout()


def plot_noise_with_image(
    dps_location, Noise_level_local, coronal_image, slice_location
):
    print("")
    fig, ax = plt.subplots(num="Noise Level", figsize=(10, 6))
    # try:
    #     xs, cs_values = spline_interpolate(dps_location, Noise_level_local)
    #     ax.plot(xs, cs_values, "b-", linewidth=2, label="Noise (Interpolated)")
    # except ValueError as e:
    #     pass
    ax.plot(
        dps_location,
        Noise_level_local,
        "g--",
        linewidth=2,
        label="Noise Level (HU)",
        marker="o",
    )

    y_min, y_max = ax.get_ylim()

    x_min, x_max = slice_location.min(), slice_location.max()
    img_extent = [x_min, x_max, y_min, y_max]

    ax.imshow(
        coronal_image,
        extent=img_extent,
        aspect="auto",
        cmap="gray",
        vmin=-1024,
        vmax=400,
        origin="upper",
        zorder=0,
    )

    ax.set_xlabel("Z Location (mm)")
    ax.set_ylabel("Noise Level (HU)")

    fig.tight_layout()
    # plt.show()


def plot_NPS(Spatial_freq, NPS_1D):
    fig, ax = plt.subplots(num="Noise Power Spectrum (NPS)")

    ax.plot(Spatial_freq, NPS_1D, "b")
    ax.set_xlabel("Spatial frequency (1/cm)", fontsize=16)
    ax.set_ylabel("NPS", fontsize=16)
    ax.grid(True)
    fig.tight_layout()
    # plt.show()


# ---------------------------------------------------------------------------
# ---- NEW: _save_all_figures  (only called when output_dir is set) --------
# ---------------------------------------------------------------------------


def _save_all_figures(output_dir: pathlib.Path):
    """Save every open matplotlib figure to output_dir as a PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        label = fig.get_label() or f"figure_{fig_num}"
        # Sanitise label for use as a filename
        safe_label = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in label
        ).strip()
        safe_label = safe_label.replace(" ", "_")
        filepath = output_dir / f"{safe_label}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved figure: {filepath}")
    plt.close("all")


def update_arrays(arr, remove_run_window, eval_run_window):
    new_arr = np.zeros_like(arr)
    new_arr[np.logical_not(eval_run_window)] = arr[np.logical_not(remove_run_window)]
    return new_arr


def group_boundaries(counts, threshold, overlap=0.0):
    bounds, totals, start, end = [], [], 0, 1

    while end <= len(counts):
        total = np.sum(counts[start:end])
        if total >= threshold:
            bounds.append((int(start), int(end)))
            totals.append(total)
            length = end - start + 1
            overlap_length = int(length * overlap)
            start = max(end - overlap_length, start + 1)
            end = start + 1
        elif end == len(counts):
            bounds.append((int(start), int(end)))
            totals.append(np.sum(counts[start:end]))
            break
        else:
            end += 1

    if totals[-1] < threshold and len(bounds) > 1:
        last_start, last_end = bounds[-1]
        for i in range(last_start - 1, 0, -1):
            new_total = np.sum(counts[i:last_end])
            if new_total >= threshold:
                bounds[-1] = (i, last_end)
                totals[-1] = new_total
                break

    return bounds


def collapse_slice(lesion_roi_maps):
    out = {}
    for roi_size, windows in lesion_roi_maps.items():
        out[roi_size] = {}
        for window, slices in enumerate(windows):
            arrays = [arr for _, arr in enumerate(slices)]
            out[roi_size][window] = np.concatenate(arrays, axis=0)
    return out


# ---------------------------------------------------------------------------
# ---- NEW: main()  –  wraps the original execution block ------------------
# ---------------------------------------------------------------------------
def main(
    dicom_dir: str,
    lesion_file: str,
    psf_file: str | None,
    mtf50: float | None,
    mtf10: float | None,
    output_dir: str | None,
    manufacturer="Siemens",
    model="EID Force",
    kernel="Br44",
):
    """
    Run the full CT image quality pipeline.

    Parameters
    ----------
    dicom_dir   : Directory containing DICOM slices (required).
    lesion_file : Path to the lesion .mat file (required).
    psf_file    : Path to the PSF .mat file, or None to simulate from MTF values.
    mtf50       : MTF50 (cycles/mm).  Required when psf_file is None.
    mtf10       : MTF10 (cycles/mm).  Optional; use None for pure Gaussian PSF.
    output_dir  : Directory to save figures and results JSON.  None → interactive show only.
    """

    # Demo Main Program
    import time

    Thr1 = 0
    Thr2 = 150
    numResample = 500
    internalNoise = 2.25
    Resampling_method = "Bootstrap"
    # lesion_contrasts = [
    #     -30,
    #     -30,
    #     -30,
    #     -10,
    #     -10,
    #     -10,
    #     -50,
    #     -50,
    #     -50,
    # ]  # HU contrast levels
    # lesion_sizes_mm = [3, 9, 6, 3, 9, 6, 3, 9, 6]  # Lesion diameters in mm
    # roi_sizes_mm = [
    #     14,
    #     19,
    #     17,
    #     14,
    #     19,
    #     17,
    #     14,
    #     19,
    #     17,
    # ]  # ROI sizes in mm for each condition
    lesion_contrasts = [-30, -30, -10, -30, -50]  # HU contrast levels
    lesion_sizes_mm = [3, 9, 6, 6, 6]  # Lesion diameters in mm
    roi_sizes_mm = [14, 19, 17, 17, 17]  # ROI sizes in mm for each condition
    # lesion_contrasts = [None, None, None, None]  # HU contrast levels
    # lesion_sizes_mm = [None, 3, 9, 6]  # Lesion diameters in mm
    # roi_sizes_mm = [None, None, None, None]  # ROI sizes in mm for each condition
    # lesion_contrasts = [None]  # HU contrast levels
    # lesion_sizes_mm = [None]  # Lesion diameters in mm
    # roi_sizes_mm = [None]  # ROI sizes in mm for each condition
    wire_fov_mm = 50
    wire_Matrix_size = 512

    out_path = pathlib.Path(output_dir) if output_dir else None
    start = time.time()

    # Load DICOM files from directory
    dir1 = dicom_dir
    # dir1 = r"\\mfad\researchMN\EB036541\YU\PUBLIC\PatientCT_monitoring\DICOMs\5W\ROLAND, THOMAS\IMAGES"
    scan_listing = sorted(os.listdir(dir1))
    n_images = len(scan_listing)
    filepaths = [os.path.join(dir1, scan) for scan in scan_listing]

    # Extract metadata from first two DICOM files
    dcm_info = dcmread(filepaths[0])
    RescaleIntercept = dcm_info.RescaleIntercept
    RescaleSlope = dcm_info.RescaleSlope
    SliceLocation1 = dcm_info.SliceLocation
    dcm_info2 = dcmread(filepaths[1])
    SliceLocation2 = dcm_info2.SliceLocation
    Sliceinterval = abs(SliceLocation1 - SliceLocation2)
    Im_Size = dcm_info.pixel_array.shape
    rFOV = dcm_info.ReconstructionDiameter / 10
    dx = rFOV / Im_Size[0]
    dy = rFOV / Im_Size[1]
    patient_FOV = dcm_info.ReconstructionDiameter  # FOV in mm from DICOM metadata
    patient_matrix = (
        dcm_info.Rows
    )  # Matrix size from DICOM metadata (assumes square matrix)

    # ------------------------------------------------------------------
    # Determine parameters a and b for SSDE calculation based on Body Part Examined
    # ------------------------------------------------------------------
    # Read Body Part Examined from the first image (Tag (0018,0015))
    body_part = dcm_info.get((0x0018, 0x0015), None)

    if body_part is not None:
        body_part_str = str(body_part.value).strip().upper()
        print(f"Body Part Examined from DICOM: {body_part_str}")

        if "ABDOMEN" in body_part_str or "PELVI" in body_part_str:
            # Abdomen / Pelvis / Torso (non-head)
            para_a = 3.704369
            para_b = 0.03671937
            print("Using abdomen parameters for SSDE calculation.")
        elif (
            "HEAD" in body_part_str
            or "BRAIN" in body_part_str
            or "SKULL" in body_part_str
        ):
            # Head / Brain / Skull
            para_a = 1.874799
            para_b = 0.03871313
            print("Using head parameters for SSDE calculation.")
        else:
            # Default to abdomen (safest for most body CT)
            para_a = 3.704369
            para_b = 0.03671937
            print(
                f"Unknown Body Part '{body_part_str}'. Defaulting to abdomen parameters."
            )
    else:
        # Fallback if tag is missing
        para_a = 3.704369
        para_b = 0.03671937
        print(
            "Body Part Examined tag (0018,0015) not found. Defaulting to abdomen parameters."
        )
    pixel_size = patient_FOV / patient_matrix  # Pixel size in mm (~0.6640625)

    # Define reconstruction kernel information
    kernel_file = r".\src\data\Recon_Kernels.xlsx"  # ← Update this path
    psf_size = wire_Matrix_size | 1
    if mtf50 is None:
        mtf50, mtf10 = get_mtf_from_excel(kernel_file, manufacturer, model, kernel)

    psf_sim, dx_psf = simulate_psf_damped_cosine(
        mtf50=mtf50, mtf10=mtf10, size=psf_size, wire_fov_mm=wire_fov_mm
    )

    lesion_sigs = []
    roi_sizes_px = []
    print(f"Using MTF50 = {mtf50:.3f}, MTF10 = {mtf10:.3f} for PSF simulation")

    for i in range(len(lesion_contrasts)):
        contrast_target = lesion_contrasts[i]
        width_mm_target = lesion_sizes_mm[i]
        roi_size_mm = roi_sizes_mm[i]

        (
            Lesion_sig,
            freq,
            mtf,
            roi_size_px,
            contrast_target,
            width_mm_target,
            roi_size_mm,
        ) = prepare_Lesion_sig(
            lesion_file,
            contrast_target,
            width_mm_target,
            roi_size_mm,
            psf_sim,
            dx_psf,
            pixel_size,
        )

        lesion_sigs.append(Lesion_sig)
        roi_sizes_px.append(roi_size_px)
        lesion_contrasts[i] = contrast_target
        roi_sizes_mm[i] = roi_size_mm
        lesion_sizes_mm[i] = width_mm_target
        # plot_lesion_signals(Lesion_sig, freq, mtf, contrast_target, width_mm_target, i)

    # Calculate CTDIvol and water-equivalent diameter (Dw)
    CTDI_All = np.zeros(n_images)
    slice_location = np.zeros(n_images)
    Dw = np.zeros(n_images)
    coronal_image = np.zeros((Im_Size[0], n_images))
    slice_location = np.zeros(n_images)
    pixel_roi = dx * dy * 100

    for i in range(n_images):
        dcm_info = dcmread(filepaths[i])
        CTDI_All[i] = dcm_info.CTDIvol
        im1 = dcm_info.pixel_array
        im = im1 * dcm_info.RescaleSlope + dcm_info.RescaleIntercept
        coronal_image[:, i] = im[Im_Size[1] // 2, :]
        Mask = im >= -260
        Mask22 = ndimage.binary_fill_holes(Mask).astype(bool)
        labeled_image, num_labels = measure.label(Mask22, return_num=True)
        if num_labels > 0:
            largest_region = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
            binaryImage = labeled_image == largest_region
        else:
            binaryImage = np.zeros_like(Mask22, dtype=bool)
        pixel_No = np.sum(binaryImage)
        A_roi = pixel_No * pixel_roi
        im10_pat = im[binaryImage]
        HU_Mean = np.mean(im10_pat)
        Dw[i] = 2 * np.sqrt((HU_Mean / 1000 + 1) * A_roi / np.pi)
        slice_location[i] = dcm_info.SliceLocation
        if i == n_images - 1:
            SliceLocation_end = dcm_info.SliceLocation
    slice_location = slice_location - slice_location.min()

    # Compute dose metrics: SSDE and DLP
    Mean_CTDI_All = np.mean(CTDI_All)
    Mean_Dw = np.mean(Dw) / 10

    # Now compute SSDE using the selected parameters
    f = para_a * np.exp(-para_b * Mean_Dw)
    ssde_inc = f * CTDI_All
    SSDE = f * Mean_CTDI_All

    Scan_len = (slice_location.max() - slice_location.min()) / 10
    DLP_CTDIvol_L = Scan_len * Mean_CTDI_All
    DLP_SSDE = Scan_len * SSDE

    val = 0.6 / dx
    ROI_size_N = int(2 * np.round((val - 1) / 2) + 1)  # round_to_nearest_odd

    Half_ROI_size_N = round(np.floor(ROI_size_N / 2))

    # Configure Gabor channels for CHO
    class Chnl:
        Chnl_Toggle = "Gabor"

    Chnl_Toggle = "Gabor"
    Chnl.Chnl_Toggle = Chnl_Toggle
    if Chnl.Chnl_Toggle == "Gabor":
        # Gabor_passband = '[[1/64,1/32], [1/32,1/16]]'
        # Gabor_theta = '[0, pi/2]'
        Gabor_passband = "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]"
        Gabor_theta = "[0, pi/3, 2*pi/3]"
        Gabor_beta = "0"
        Chnl = channel_selection(Chnl, Gabor_passband, Gabor_theta, Gabor_beta)
    elif Chnl.Chnl_Toggle == "Laguerre-Gauss":
        LG_order = 6
        LG_orien = 3
        Chnl = channel_selection(Chnl, LG_order, LG_orien)

    # # Process image sections for NPS and CHO analysis
    num_lesions = len(lesion_sigs)
    Padding_size = int(2 * round(1.2 / dx))
    Padding_size += Padding_size % 2
    window_size = [Padding_size, Padding_size]
    # Desired buffer zone around edges (mm). Tune if needed: 1.5~2.5
    physical_buffer_mm = 1.5
    # gives the kernel size in pixels for the desired physical buffer in mm.
    kernel_size = max(3, int(np.round(physical_buffer_mm / (dx * 10)))) | 1
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    window_size_mm = 150
    step_size_mm = 50
    window_size_idx = np.ceil((window_size_mm + 1e-6) / Sliceinterval).astype(int)
    step_size_idx = np.ceil((step_size_mm + 1e-6) / Sliceinterval).astype(int)

    algorithm = "old"  # new old
    if algorithm == "new":

        images = np.array(
            [
                dcmread(filepaths[i]).pixel_array * dcm_info.RescaleSlope
                + dcm_info.RescaleIntercept
                for i in range(n_images)
            ]
        )
        # images = ahe(images)
        images_integrated = integrate_image(images, window_size)
        images_squared_integrated = integrate_image(images**2, window_size)
        images_canny_edge_integrated = integrate_image(
            np.array(
                [
                    binary_dilation(feature.canny(im, sigma=5), structure=structure)
                    for im in images
                ]
            ),
            window_size,
        )
        nps_cropping = images.shape[0] // 3  # Reorder back to (z, y, x)
        nps_std_map = calculate_std_dev(
            images_integrated,
            images_squared_integrated,
            images_canny_edge_integrated,
            ROI_size_N,
            Padding_size,
            images.shape,
            Thr1,
            Thr2,
            corrected=False,
        )
        max_values = np.nanmax(nps_std_map, axis=(1, 2))
        max_values = np.lib.stride_tricks.sliding_window_view(
            np.pad(max_values, window_size_idx // 2, mode="reflect"), window_size_idx
        ).mean(axis=1)

        histos = [
            np.histogram(
                nps_std_map[i].flatten(), bins=np.arange(0, max_values[i], 0.2)
            )
            for i in range(nps_std_map.shape[0])
        ]
        h_Valuesss = [h[0] for h in histos]
        edgess = [h[1] for h in histos]
        whichbin_SDs = [np.argmax(h) for h in h_Valuesss]
        Noise_level_local = [
            edges[whichbin_SD] for edges, whichbin_SD in zip(edgess, whichbin_SDs)
        ]

        max_value = np.nanmax(nps_std_map[nps_cropping : 2 * nps_cropping])
        h_Values, edges = np.histogram(
            nps_std_map[nps_cropping : 2 * nps_cropping].flatten(),
            bins=np.arange(0, max_value, 0.2),
        )
        whichbin_SD = np.argmax(h_Values)
        sd_thresh = edges[whichbin_SD]
        ROI_All_NPS = extract_ROIs(
            nps_std_map,
            sd_thresh,
            Half_ROI_size_N,
            images,
            group=True,
        )
        nps_rois_counts = [ROI_All_NPS[loc].shape[0] for loc in range(len(ROI_All_NPS))]
        group_by = "lesion_count"  # slice_location lesion_count
        num_nps_rois = min(
            200, np.sum(nps_rois_counts[nps_cropping : 2 * nps_cropping])
        )
        Spatial_freq, NPS_1D_sum, noise_level_sum, unit = ROI_to_NPS_Sum(
            ROI_size_N,
            np.concatenate(ROI_All_NPS[nps_cropping : 2 * nps_cropping])[:num_nps_rois],
            dx,
            dy,
        )

        Left_Area = np.sum(h_Values[:whichbin_SD])
        Left_Areas = [np.sum(h_Valuesss[i][: whichbin_SDs[i]]) for i in range(n_images)]
        Two_sigma_area = np.ones(whichbin_SD) * Left_Area
        Two_sigma_areas = [
            np.ones(whichbin_SDs[i]) * Left_Areas[i] for i in range(n_images)
        ]
        cusu = np.cumsum(h_Values[: whichbin_SD - 1])
        cusus = [
            np.cumsum(h_Valuesss[i][: whichbin_SDs[i] - 1]) for i in range(n_images)
        ]
        Two_sigma_area[1:] = Left_Area - cusu
        for i in range(n_images):
            Two_sigma_areas[i][1:] = Left_Areas[i] - cusus[i]
        Ratios = Two_sigma_area / Left_Area
        Ratioss = [Two_sigma_areas[i] / Left_Areas[i] for i in range(n_images)]
        target_ratio = 0.9544
        temp_dis = np.abs(target_ratio - Ratios)
        temp_diss = [np.abs(target_ratio - Ratioss[i]) for i in range(n_images)]
        closest = Ratios[np.argmin(temp_dis)]
        closests = [Ratioss[i][np.argmin(temp_diss[i])] for i in range(n_images)]
        loc = np.where(Ratios == closest)[0][0]
        locs = [np.where(Ratioss[i] == closests[i])[0][0] for i in range(n_images)]
        gap = whichbin_SD - loc
        gaps = [whichbin_SDs[i] - locs[i] for i in range(n_images)]
        Thre_SD = edges[whichbin_SD + gap]
        Thre_SDs = np.array([edges[whichbin_SDs[i] + gaps[i]] for i in range(n_images)])
        Thre_SDs = np.lib.stride_tricks.sliding_window_view(
            np.pad(Thre_SDs, window_size_idx // 2, mode="reflect"), window_size_idx
        ).mean(axis=1)
        unique_roi_sizes = np.unique(roi_sizes_px)
        lesion_std_maps = {
            roi_size: calculate_std_dev(
                images_integrated,
                images_squared_integrated,
                images_canny_edge_integrated,
                roi_size,
                Padding_size,
                images.shape,
                Thr1,
                Thr2,
                corrected=False,
            )
            for roi_size in unique_roi_sizes
        }
        lesion_roi_maps = {
            roi_size: extract_ROIs(
                lesion_std_maps[roi_size], Thre_SDs, roi_size // 2, images, group=True
            )
            for roi_size in unique_roi_sizes
        }

        for roi_size in unique_roi_sizes:
            for si in range(len(lesion_roi_maps[roi_size])):
                for li in range(lesion_roi_maps[roi_size][si].shape[0]):
                    lesion_roi_maps[roi_size][si][li] = lesion_roi_maps[roi_size][si][
                        li
                    ] - np.mean(lesion_roi_maps[roi_size][si][li])

        lesion_counts = {
            roi_size: [
                lesion_roi_maps[roi_size][loc].shape[0]
                for loc in range(len(lesion_roi_maps[roi_size]))
            ]
            for roi_size in unique_roi_sizes
        }
        if group_by == "lesion_count":
            lesion_counts_grouped = {
                roi_size: group_boundaries(
                    lesion_counts[roi_size], threshold=500, overlap=0.667
                )
                for roi_size in unique_roi_sizes
            }
            lesion_roi_maps = {
                roi_size: [
                    lesion_roi_maps[roi_size][start:end]
                    for start, end in lesion_counts_grouped[roi_size]
                ]
                for roi_size in unique_roi_sizes
            }
            lesion_roi_locs = {
                roi_size: [
                    np.mean(slice_location[start:end])
                    for start, end in lesion_counts_grouped[roi_size]
                ]
                for roi_size in unique_roi_sizes
            }
            noise_level_local = {
                roi_size: [
                    np.mean(Noise_level_local[start:end])
                    for start, end in lesion_counts_grouped[roi_size]
                ]
                for roi_size in unique_roi_sizes
            }
        elif group_by == "slice_location":
            windows = np.lib.stride_tricks.sliding_window_view(
                np.arange(n_images), window_size_idx
            )[::step_size_idx]
            windows = [(w[0], w[-1] + 1) for w in windows]

            lesion_roi_maps = {
                roi_size: [
                    lesion_roi_maps[roi_size][start:end] for start, end in windows
                ]
                for roi_size in unique_roi_sizes
            }
            lesion_roi_locs = {
                roi_size: [np.mean(slice_location[start:end]) for start, end in windows]
                for roi_size in unique_roi_sizes
            }
            noise_level_local = {
                roi_size: [
                    np.mean(Noise_level_local[start:end]) for start, end in windows
                ]
                for roi_size in unique_roi_sizes
            }
        else:
            raise ValueError(
                "Invalid group_by value. Must be 'lesion_count' or 'slice_location'."
            )
        lesion_roi_maps = collapse_slice(lesion_roi_maps)
        sample_idxs = {
            roi_size: [
                np.random.permutation(lesion_roi_maps[roi_size][win].shape[0])
                for win in range(len(lesion_roi_maps[roi_size]))
            ]
            for roi_size in unique_roi_sizes
        }
        bkgs_ordered = {
            roi_size: [
                np.reshape(
                    lesion_roi_maps[roi_size][win][sample_idxs[roi_size][win]],
                    (roi_size**2, len(sample_idxs[roi_size][win])),
                )
                for win in range(len(lesion_roi_maps[roi_size]))
            ]
            for roi_size in unique_roi_sizes
        }
        sigs_true = {
            f"{lesion_contrast}_{roi_size}": np.reshape(lesion_sig, (roi_size**2, 1))
            for lesion_contrast, roi_size, lesion_sig in zip(
                lesion_contrasts, roi_sizes_px, lesion_sigs
            )
        }

        channel_matrices = {
            roi_size: ChannelMatrix_Generation(Chnl, roi_size)
            for roi_size in unique_roi_sizes
        }
        dps = {
            f"{lesion_contrast}_{roi_size}": {
                "dps": [
                    CHO_patient_with_resampling(
                        sigs_true[f"{lesion_contrast}_{roi_size}"],
                        bkgs_ordered[roi_size][win],
                        channel_matrices[roi_size],
                        internalNoise,
                        Resampling_method,
                    )
                    for win in range(len(lesion_roi_maps[roi_size]))
                ],
                "loc": lesion_roi_locs[roi_size],
            }
            for lesion_contrast, roi_size in zip(lesion_contrasts, roi_sizes_px)
        }
        loc_min = -np.inf
        loc_max = np.inf
        for les in dps.keys():
            if les in ["avg", "all_lesions"]:
                continue
            loc_min = max(np.min(dps[les]["loc"]), loc_min)
            loc_max = min(np.max(dps[les]["loc"]), loc_max)
        locs = np.linspace(loc_min, loc_max, n_images)

        for lesion_contrast, roi_size in zip(lesion_contrasts, roi_sizes_px):
            dps[f"{lesion_contrast}_{roi_size}"]["interp"] = np.interp(
                locs,
                dps[f"{lesion_contrast}_{roi_size}"]["loc"],
                dps[f"{lesion_contrast}_{roi_size}"]["dps"],
                left=np.nan,
                right=np.nan,
            )
        for roi_size in unique_roi_sizes:
            noise_level_local[roi_size] = np.interp(
                locs,
                lesion_roi_locs[roi_size],
                noise_level_local[roi_size],
                left=np.nan,
                right=np.nan,
            )
        Noise_level_local = np.mean(
            [noise_level_local[roi_size] for roi_size in unique_roi_sizes], axis=0
        )

        dps["all_lesions"] = {
            "dps": np.mean(
                [
                    dps[f"{lesion_contrast}_{roi_size}"]["interp"]
                    for lesion_contrast, roi_size in zip(lesion_contrasts, roi_sizes_px)
                ],
                axis=0,
            ),
            "loc": locs,
        }
        dps["avg"] = np.nanmean(dps["all_lesions"]["dps"])

        Mean_loc_dps = dps["all_lesions"]["dps"]
        # Mean_All_dps = np.mean(Mean_loc_dps)
        Mean_All_dps = dps["avg"]

        dps_location = dps["all_lesions"]["loc"]
    elif algorithm == "old":
        # Average the DPS values across all lesions
        # Interpolate DPS values to a common set of locations for averaging

        window_size_mm = 150
        step_size_mm = 50
        window_size_idx = np.ceil((window_size_mm + 1e-6) / Sliceinterval).astype(int)
        step_size_idx = np.ceil((step_size_mm + 1e-6) / Sliceinterval).astype(int)
        windows = np.lib.stride_tricks.sliding_window_view(
            np.arange(n_images), window_size_idx
        )[::step_size_idx]
        n_windows = windows.shape[0]
        # Which elements in each window are unique (not in previous window).  Used to avoid double-counting slices.
        eval_run_windows = np.array(
            [
                (
                    [not val in set(windows[i - 1]) for val in windows[i]]
                    if i > 0
                    else [True] * len(windows[i])
                )
                for i in range(len(windows))
            ]
        )
        # Which elements in each window are unique (not in next window).  Used to avoid double-counting slices.
        remove_run_windows = np.array(
            [
                (
                    [not val in set(windows[i + 1]) for val in windows[i]]
                    if i > -1
                    else [True] * len(windows[i])
                )
                for i in range(-1, len(windows) - 1)
            ]
        )

        dps_location = np.mean(slice_location[windows], axis=-1)
        padded_shape = [
            window_size_idx,
            Im_Size[0] + Padding_size,
            Im_Size[1] + Padding_size,
        ]
        init_shape = [
            window_size_idx,
            Im_Size[0],
            Im_Size[1],
        ]
        Noise_level_local = np.zeros([n_windows, 1])
        NPS_Cal = True
        _, first_idx = np.unique(roi_sizes_mm, return_index=True)
        mask = np.ones(len(roi_sizes_mm), dtype=bool)
        mask[first_idx] = False
        same_roi_sizes = np.flatnonzero(mask)

        All_dps = np.zeros([n_windows, num_lesions])
        image_section = np.zeros(init_shape, dtype=np.float32)
        images_integrated = np.zeros(padded_shape, dtype=np.float32)
        images_squared_integrated = np.zeros(padded_shape, dtype=np.float32)
        images_canny_edge_integrated = np.zeros(padded_shape, dtype=np.float32)

        for window_idx, window in enumerate(windows):
            eval_run_window = eval_run_windows[window_idx]
            remove_run_window = remove_run_windows[window_idx]
            image_section = update_arrays(image_section, remove_run_window, eval_run_window)  # type: ignore
            images_integrated = update_arrays(
                images_integrated, remove_run_window, eval_run_window  # type: ignore
            )
            images_squared_integrated = update_arrays(
                images_squared_integrated, remove_run_window, eval_run_window  # type: ignore
            )
            images_canny_edge_integrated = update_arrays(
                images_canny_edge_integrated, remove_run_window, eval_run_window  # type: ignore
            )
            for element_idx, image_idx in enumerate(window):
                if not eval_run_window[element_idx]:
                    continue

                dcm_info = dcmread(filepaths[image_idx])
                im = (
                    dcm_info.pixel_array * dcm_info.RescaleSlope
                    + dcm_info.RescaleIntercept
                )

                canny_edges = feature.canny(im, sigma=5)
                dilated_edges = binary_dilation(canny_edges, structure=structure)

                image_section[element_idx] = im
                images_integrated[element_idx] = integrate_image(im, window_size)
                images_squared_integrated[element_idx] = integrate_image(
                    im**2, window_size
                )
                images_canny_edge_integrated[element_idx] = integrate_image(
                    dilated_edges, window_size
                )

            STD_all = calculate_std_dev(
                images_integrated,
                images_squared_integrated,
                images_canny_edge_integrated,
                ROI_size_N,
                Padding_size,
                init_shape,
                Thr1,
                Thr2,
                corrected=False,
            )
            max_value = np.nanmax(STD_all)
            h_Values, edges = np.histogram(
                STD_all.flatten(), bins=np.arange(0, max_value, 0.2)
            )
            whichbin_SD = np.argmax(h_Values)
            bin_edge = edges[whichbin_SD]
            Noise_level_local[window_idx] = bin_edge
            if NPS_Cal and (window_idx >= n_windows // 2):
                print(f"Calculating NPS from the middle period (group {window_idx+1})")
                ROI_All_NPS = extract_ROIs(
                    STD_all, bin_edge, Half_ROI_size_N, image_section
                )
                Total_NPS_No = ROI_All_NPS.shape[0]
                # Use up to 200 ROIs (or all available if fewer)
                num_nps_rois = min(200, Total_NPS_No)
                Spatial_freq, NPS_1D_sum, noise_level_sum, unit = ROI_to_NPS_Sum(
                    ROI_size_N, ROI_All_NPS[0:num_nps_rois], dx, dy
                )
                Total_NPS_Num = num_nps_rois
                NPS_Cal = False
            del STD_all
            gc.collect()
            Left_Area = np.sum(h_Values[:whichbin_SD])
            Two_sigma_area = np.ones(whichbin_SD) * Left_Area
            cusu = np.cumsum(h_Values[: whichbin_SD - 1])
            Two_sigma_area[1:] = Left_Area - cusu
            Ratios = Two_sigma_area / Left_Area
            target_ratio = 0.9544
            temp_dis = np.abs(target_ratio - Ratios)
            closest = Ratios[np.argmin(temp_dis)]
            loc = np.where(Ratios == closest)[0][0]
            gap = whichbin_SD - loc
            Thre_SD = edges[whichbin_SD + gap]
            all_dps = []
            for ii in range(num_lesions):
                lesion_sig = lesion_sigs[ii]
                roi_size_px = roi_sizes_px[ii]

                Half_ROI_size = roi_size_px // 2
                if ii in same_roi_sizes and ii > 0:
                    print("use same ROI_All")
                else:
                    STD_map_all = calculate_std_dev(
                        images_integrated,
                        images_squared_integrated,
                        images_canny_edge_integrated,
                        roi_size_px,
                        Padding_size,
                        # Im_Size,
                        # N1,
                        # N2,
                        init_shape,
                        # window_size_idx,
                        Thr1,
                        Thr2,
                        corrected=False,
                    )
                    ROI_All = extract_ROIs(
                        STD_map_all, Thre_SD, Half_ROI_size, image_section
                    )
                    Total_NPS_No = ROI_All.shape[0]
                Noise_ROI = ROI_All[:Total_NPS_No]  # type: ignore
                depth = Noise_ROI.shape[0]
                for i in range(depth):
                    temp = Noise_ROI[i]
                    Noise_ROI[i] = temp - np.mean(temp)
                N_total = Noise_ROI.shape[0]
                sample_idx = np.random.permutation(N_total)
                channelMatrix = ChannelMatrix_Generation(Chnl, roi_size_px)
                bkg_ordered = np.reshape(
                    Noise_ROI[sample_idx], (roi_size_px**2, len(sample_idx))
                )
                sig_true = np.reshape(lesion_sig, (roi_size_px**2, 1))
                dp = CHO_patient_with_resampling(
                    sig_true,
                    bkg_ordered,
                    channelMatrix,
                    internalNoise,
                    Resampling_method,
                )
                all_dps.append(dp)
            All_dps[window_idx, :] = all_dps
            # del (
            #     images_integrated,
            #     images_squared_integrated,
            #     images_canny_edge_integrated,
            #     image_section,
            # )
            gc.collect()

        # Compute final metrics and plot NPS
        # dps_location_array = np.array(dps_location)
        # dps_location_array = dps_location_array - dps_location_array.min()
        # dps_location_array = np.reshape(
        #     dps_location_array, (n_images // window_size_idx - 2, -1)
        # )
        # dps_location_array = np.mean(dps_location_array, axis=-1)
        Mean_loc_dps = np.mean(All_dps, axis=-1)
        Mean_All_dps = np.mean(Mean_loc_dps)

    # Mean_loc_dps = dps["all_lesions"]["dps"]
    # # Mean_All_dps = np.mean(Mean_loc_dps)
    # Mean_All_dps = dps["avg"]
    NPS_1D = NPS_1D_sum / num_nps_rois
    Ave_noise_level_NPS = noise_level_sum / num_nps_rois
    Ave_noise_level = np.mean(Noise_level_local)
    end = time.time()
    elapsed = end - start
    # dps_location = dps["all_lesions"]["loc"]

    # ---- original plot calls (unchanged) ----
    plot_Dw_with_image(slice_location, Dw, coronal_image)
    plot_CTDI_SSDE_with_image(slice_location, CTDI_All, ssde_inc, coronal_image)
    plot_noise_with_image(
        dps_location, Noise_level_local, coronal_image, slice_location
    )
    plot_detectability_with_image(
        dps_location, Mean_loc_dps, coronal_image, slice_location
    )
    fig, ax = plt.subplots(num="Detectability vs. Slice Location")
    for les in dps.keys():
        if les in ["avg"]:
            continue
        ax.plot(dps[les]["loc"], dps[les]["dps"], label=les)
    # for line in All_dps.T:
    #     ax.plot(dps_location, line)
    y_min, y_max = ax.get_ylim()
    x_min, x_max = slice_location.min(), slice_location.max()
    img_extent = [x_min, x_max, y_min, y_max]
    ax.imshow(
        coronal_image,
        extent=img_extent,
        aspect="auto",
        cmap="gray",
        vmin=-1024,
        vmax=400,
        origin="upper",
        zorder=0,
    )
    ax.set_xlabel("Slice Location (mm)", fontsize=16)
    ax.set_ylabel("Detectability", fontsize=16)
    ax.legend()
    # plt.show()
    plot_NPS(Spatial_freq, NPS_1D)

    fav, peakfrequency, k, min10percent_frequency = NPS_statistics(NPS_1D, unit)
    print(f"peakfrequency = {peakfrequency:.3f}")
    print(f"averagefrequency = {fav:.3f}")
    print(f"min10percent_frequency = {min10percent_frequency:.3f}")
    print(f"average noise level = {Ave_noise_level:.3f}")
    print(f"elapsed time = {elapsed:.1f} s")

    results = {
        "average_frequency": fav,
        "average_index_of_detectability": Mean_All_dps,
        "average_noise_level": float(Ave_noise_level),
        "cho_detectability": Mean_loc_dps.tolist(),
        "dps_location": dps_location.tolist(),
        "ctdivol": CTDI_All.tolist(),
        "ctdivol_avg": Mean_CTDI_All,
        "dlp": DLP_CTDIvol_L,
        "dlp_ssde": DLP_SSDE,
        "dw": Dw.tolist(),
        "dw_avg": Mean_Dw,
        "location": slice_location.tolist(),
        "noise_level": Noise_level_local,
        "nps": NPS_1D.flatten().tolist(),
        "peak_frequency": peakfrequency,
        "percent_10_frequency": min10percent_frequency,
        "spatial_frequency": Spatial_freq.tolist(),
        "ssde": SSDE,
        "ssde_inc": ssde_inc.tolist(),
        "elapsed_seconds": elapsed,
    }

    # ---- NEW: save figures + results when output_dir is provided ----
    if out_path is not None:
        _save_all_figures(out_path)
        results_path = out_path / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results: {results_path}")
    else:
        plt.show()

    return results
