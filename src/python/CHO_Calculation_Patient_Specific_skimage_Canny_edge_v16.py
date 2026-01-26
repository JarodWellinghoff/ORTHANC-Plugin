#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import random
from numpy.linalg import inv
import os
from pydicom import dcmread
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure
from scipy.interpolate import griddata
import scipy.io as sio
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import gc


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


def ROI_to_NPS_Sum(ROI_size, ROI_All, dx, dy):
    """Compute 1D NPS from ROI images, averaging over radial frequencies."""
    nnn = 8 * max(20, ROI_size)
    cc = int(np.rint(nnn / 2))
    unit = 1 / (dx * nnn)
    oo = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.arange(cc).reshape((cc, 1))
    x_offset = radii * np.cos(oo)
    y_offset = radii * np.sin(oo)
    xi_all = np.rint(x_offset + cc - 1).astype(np.int32)
    yi_all = np.rint(y_offset + cc - 1).astype(np.int32)
    valid_mask = (xi_all >= 0) & (xi_all < nnn) & (yi_all >= 0) & (yi_all < nnn)
    NPS_1D_sum = np.zeros((cc, 1), dtype=np.float32)
    noise_level_sum = 0.0
    num_slices = ROI_All.shape[-1]
    for jj in range(num_slices):
        roi = ROI_All[:, :, jj]
        psize = (dx, dy)
        roi = subtractMean2D(roi, 1, psize)
        noise_level_sum += np.std(roi)
        roipad = np.zeros((nnn, nnn), dtype=np.float32)
        half_roi = int(np.ceil(ROI_size / 2))
        start_index = int(np.rint(nnn / 2 - half_roi))
        end_index = int(np.rint(nnn / 2 + half_roi))
        roipad[start_index : end_index - 1, start_index : end_index - 1] = roi
        roipad_fft = np.fft.fftshift(np.abs(np.fft.fft2(roipad)))
        roipad_fft = roipad_fft**2 * dx * dy / (ROI_size * ROI_size)
        nps2D = roipad_fft
        polar1 = np.zeros((cc, 360), dtype=np.float32)
        polar1[valid_mask] = nps2D[xi_all[valid_mask], yi_all[valid_mask]]
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
        channelMatrix = np.zeros((roiSize_xy * roiSize_xy, Gabor_FC.size))
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
    if Resampling_method == "Bootstrap":
        rand_scanSelect_bkg = np.random.randint(N_total_bkg, size=N_total_bkg)
    elif Resampling_method == "Shuffle":
        rand_scanSelect_bkg = np.random.permutation(N_total_bkg)
    sig = sig_true
    bkg = bkg_ordered[:, rand_scanSelect_bkg].astype(float)
    if sig.shape[0] != channelMatrix.shape[0] or bkg.shape[0] != channelMatrix.shape[0]:
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


def max_consecutive_ones_2d(bool_array_2d):
    """Find maximum consecutive ones in any row of a 2D boolean array."""
    max_consecutive = 0
    for row in bool_array_2d:
        max_count = 0
        count = 0
        for value in row:
            if value:
                count += 1
            else:
                max_count = max(max_count, count)
                count = 0
        max_count = max(max_count, count)
        max_consecutive = max(max_consecutive, max_count)
    return max_consecutive


def simulate_psf(mtf50, mtf10=None, size=257, pixel_spacing=0.6640625):
    """
    Simulate a 2D PSF based on MTF50 and optionally MTF10.

    Parameters:
    - mtf50: Spatial frequency (cycles/mm) where MTF=0.5.
    - mtf10: Spatial frequency (cycles/mm) where MTF=0.1 (optional, None for Gaussian).
    - size: PSF grid size (odd number, default 257).
    - pixel_spacing: Pixel size in mm (default 0.6640625 mm).

    Returns:
    - psf: Normalized 2D PSF array.

    Methods:
    1. Gaussian PSF (mtf10=None):
       - Uses a Gaussian function where MTF(f) = exp(-2 * pi^2 * sigma^2 * f^2).
       - Sigma is derived from mtf50: sigma = sqrt(-ln(0.5) / (2 * pi^2 * mtf50^2)).
       - Suitable for systems with smooth MTF fall-off (e.g., soft reconstruction kernels).

    2. Damped Cosine PSF (mtf10 specified):
       - Uses a damped cosine function: psf(r) = exp(-r/tau) * cos(k*r + phi).
       - Optimizes tau, k, phi to match MTF at mtf50 (0.5) and mtf10 (0.1).
       - Suitable for systems with oscillatory MTF (e.g., sharper kernels with edge enhancement).
       - Optimization minimizes error in MTF values at specified frequencies.
    """
    # Ensure odd size for symmetric PSF
    if size % 2 == 0:
        size += 1
    half = size // 2

    # Create spatial grid in mm
    x = np.linspace(-half, half, size) * pixel_spacing
    y = np.linspace(-half, half, size) * pixel_spacing
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    if mtf10 is None:
        # Gaussian PSF: MTF(f) = exp(-2 * pi^2 * sigma^2 * f^2)
        # Solve for sigma where MTF(mtf50) = 0.5
        ln05 = np.log(0.5)
        sigma_sq = -ln05 / (2 * np.pi**2 * mtf50**2)
        sigma = np.sqrt(sigma_sq)
        psf = np.exp(-(R**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        psf /= np.sum(psf)  # Normalize to sum to 1
    else:
        # Damped Cosine PSF: psf(r) = exp(-r/tau) * cos(k*r + phi)
        def generate_psf(params):
            tau, k, phi = params
            psf = np.exp(-R / tau) * np.cos(k * R + phi)
            return psf

        def compute_radial_mtf(psf):
            """Compute radial MTF from PSF via 2D FFT."""
            psf_shifted = np.fft.ifftshift(psf)
            ft = np.fft.fft2(psf_shifted)
            mtf_2d = np.abs(np.fft.fftshift(ft)) / np.abs(ft[0, 0])
            fx = np.fft.fftfreq(size, pixel_spacing)
            fy = np.fft.fftfreq(size, pixel_spacing)
            Fx, Fy = np.meshgrid(fx, fy)
            F = np.sqrt(Fx**2 + Fy**2)
            center = size // 2
            f_rad = F[center, center:]
            mtf_rad = mtf_2d[center, center:]
            mtf_func = interp1d(f_rad, mtf_rad, bounds_error=False, fill_value=0)
            return mtf_func

        def objective(params):
            """Minimize error in MTF at mtf50 and mtf10."""
            psf = generate_psf(params)
            psf /= np.sum(np.abs(psf))  # Normalize
            mtf_func = compute_radial_mtf(psf)
            error = (mtf_func(mtf50) - 0.5) ** 2 + (mtf_func(mtf10) - 0.1) ** 2
            return error

        # Initial parameters: tau based on mtf50, k for oscillation, phi=0
        tau0 = 1 / (np.pi * mtf50)
        k0 = np.pi * (mtf10 - mtf50)
        phi0 = 0
        params0 = [tau0, k0, phi0]
        result = minimize(objective, params0, method="Nelder-Mead")
        psf = generate_psf(result.x)
        psf /= np.sum(psf)  # Normalize to sum to 1
    return psf


def prepare_Lesion_sig(
    lesion_file,
    psf_file,
    lesion_con_target,
    target_lesion_width,
    patient_FOV,
    patient_matrix,
    ROI_size,
    mtf50=None,
    mtf10=None,
):
    """Prepare lesion signal by scaling, adjusting contrast, and convolving with PSF."""
    patient_data = sio.loadmat(lesion_file)
    lesion = patient_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
    mask = patient_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]
    loc = round(lesion.shape[-1] / 2)
    L0 = lesion[:, :, loc]
    M0 = mask[:, :, loc]
    M0 = M0.astype(bool)
    lesion_pixels_width_input = max_consecutive_ones_2d(M0)
    lesion_HU = np.mean(L0[M0])
    Diff_HU = lesion_HU - lesion_con_target
    L0 = L0 - Diff_HU
    L0[~M0.astype(bool)] = 0
    lesion_pixels_width_target = target_lesion_width / (patient_FOV / patient_matrix)
    scale = lesion_pixels_width_target / lesion_pixels_width_input
    s1, s2 = L0.shape
    os0 = np.floor(np.array(L0.shape) * scale).astype(int)
    x0 = np.linspace(0, s2 - 1, s2)
    y0 = np.linspace(0, s1 - 1, s1)
    X0, Y0 = np.meshgrid(x0, y0)
    x_out = np.linspace(0, s2 - 1, os0[1])
    y_out = np.linspace(0, s1 - 1, os0[0])
    XX, YY = np.meshgrid(x_out, y_out)
    Lesion = interpolate_grid(X0, Y0, L0, XX, YY, "linear")
    pixel_spacing = patient_FOV / patient_matrix
    if psf_file is None:
        if mtf50 is None:
            raise ValueError("mtf50 must be provided when psf_file is None")
        psf_size = 257
        PSF_end = simulate_psf(
            mtf50=mtf50, mtf10=mtf10, size=psf_size, pixel_spacing=pixel_spacing
        )
    else:
        psf_data = sio.loadmat(psf_file)
        PSF = psf_data["PSF"]
        wire_FOV = 50
        scaling_factor = wire_FOV / patient_FOV / 4
        os0_psf = np.floor(np.array(PSF.shape) * scaling_factor).astype(int)
        PSF_x0 = np.linspace(0, PSF.shape[1] - 1, PSF.shape[1])
        PSF_y0 = np.linspace(0, PSF.shape[0] - 1, PSF.shape[0])
        PSF_X0, PSF_Y0 = np.meshgrid(PSF_x0, PSF_y0)
        PSF_x_out = np.linspace(0, PSF.shape[1] - 1, os0_psf[1])
        PSF_y_out = np.linspace(0, PSF.shape[0] - 1, os0_psf[0])
        PSF_XX, PSF_YY = np.meshgrid(PSF_x_out, PSF_y_out)
        PSF_end = interpolate_grid(PSF_X0, PSF_Y0, PSF, PSF_XX, PSF_YY, "linear")
    PSF_end /= np.sum(PSF_end)
    Lesion_ext = np.zeros((80, 80))
    start_row = (Lesion_ext.shape[0] - Lesion.shape[0]) // 2
    start_col = (Lesion_ext.shape[1] - Lesion.shape[1]) // 2
    Lesion_ext[
        start_row : start_row + Lesion.shape[0], start_col : start_col + Lesion.shape[1]
    ] = Lesion
    Lesion_conv = convolve2d(Lesion_ext, PSF_end, mode="full")
    start_row = (Lesion_conv.shape[0] - ROI_size) // 2
    start_col = (Lesion_conv.shape[1] - ROI_size) // 2
    Lesion_conv_end = Lesion_conv[
        start_row : start_row + ROI_size, start_col : start_col + ROI_size
    ]
    return Lesion_conv_end


def Integral_image(image, window_size=[3, 3], padding="constant"):
    """Compute integral image for fast mean calculations."""
    if len(image.shape) != 2:
        raise ValueError("The input image must be a two-dimensional array.")
    m, n = window_size
    pad_width = ((m // 2, m // 2), (n // 2, n // 2))
    if padding == "circular":
        padded_image = np.pad(image, pad_width, mode="wrap")
    elif padding == "replicate":
        padded_image = np.pad(image, pad_width, mode="edge")
    elif padding == "symmetric":
        padded_image = np.pad(image, pad_width, mode="symmetric")
    else:
        padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)
    imageD = padded_image.astype(np.float64)
    t = np.cumsum(np.cumsum(imageD, axis=0), axis=1)
    return t


def calculate_std_dev(
    Intel_images,
    Intel_images_Square,
    Intel_Edge,
    ROI_size,
    Padding_size,
    Im_Size,
    N1,
    N2,
    N_sub,
    Thr1,
    Thr2,
    Edges,
    corrected=False,
):
    """Calculate standard deviation map for ROI selection."""
    ROI_size2 = int(ROI_size) if ROI_size % 2 == 0 else int(ROI_size + 1)
    Half_Diff_Size = round((Padding_size - ROI_size2) / 2)
    Intel_Im_Sizex = Intel_Edge.shape[0]
    Intel_Im_Sizey = Intel_Edge.shape[1]
    Intel_Edge = Intel_Edge[
        Half_Diff_Size : Intel_Im_Sizex - Half_Diff_Size,
        Half_Diff_Size : Intel_Im_Sizey - Half_Diff_Size,
        :,
    ]
    Intel_images = Intel_images[
        Half_Diff_Size : Intel_Im_Sizex - Half_Diff_Size,
        Half_Diff_Size : Intel_Im_Sizey - Half_Diff_Size,
        :,
    ]
    Intel_images_Square = Intel_images_Square[
        Half_Diff_Size : Intel_Im_Sizex - Half_Diff_Size,
        Half_Diff_Size : Intel_Im_Sizey - Half_Diff_Size,
        :,
    ]
    m = ROI_size2
    n = ROI_size2
    Half_ROI_size = round(np.floor(ROI_size / 2))
    if corrected:
        normalization_factor = m * n / (m * n - 1)
    else:
        normalization_factor = 1
    STD_all = np.zeros([Im_Size[0], Im_Size[1], N2 + N_sub * 2 - N1], dtype=np.float32)
    Mask_edge = np.zeros_like(Edges, dtype=np.float32)
    for kk in range(Intel_images.shape[-1]):
        t = Intel_images[:, :, kk]
        mean_map = t[m:, n:] + t[:-m, :-n] - t[m:, :-n] - t[:-m, n:]
        t2 = Intel_images_Square[:, :, kk]
        mean_square = t2[m:, n:] + t2[:-m, :-n] - t2[m:, :-n] - t2[:-m, n:]
        t3 = Intel_Edge[:, :, kk]
        edge_impact = t3[m:, n:] + t3[:-m, :-n] - t3[m:, :-n] - t3[:-m, n:]
        mean_map /= m * n
        mean_square /= m * n
        edge_impact /= m * n
        Mask_mean = (mean_map < Thr1) | (mean_map > Thr2)
        Mask_edge[edge_impact >= 1 / ROI_size] = 1
        Mask_edge[edge_impact < 1 / ROI_size] = 0
        Mask = np.logical_or(np.logical_or(Mask_mean, Mask_edge), Edges)
        SD_Map = np.sqrt(normalization_factor * (mean_square - mean_map**2))
        SD_Map[Mask] = np.nan
        SD_Map[0:Half_ROI_size, :] = np.nan
        SD_Map[-Half_ROI_size:, :] = np.nan
        SD_Map[:, 0:Half_ROI_size] = np.nan
        SD_Map[:, -Half_ROI_size:] = np.nan
        STD_all[:, :, kk] = SD_Map
    return STD_all


def extract_ROIs(STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size):
    """Extract ROIs with low standard deviation for NPS and CHO analysis."""
    ROIs = []
    for ii in range(STD_map_all.shape[-1]):
        im10 = Images_Section[:, :, ii]
        SD_Map = STD_map_all[:, :, ii]
        valid_mask = (SD_Map < Thre_SD) & (~np.isnan(SD_Map))
        rows, cols = np.where(valid_mask)
        if rows.size == 0:
            continue
        std_vals = SD_Map[rows, cols]
        sorted_indices = np.argsort(std_vals)
        sorted_rows = rows[sorted_indices]
        sorted_cols = cols[sorted_indices]
        selection_mask = np.zeros_like(SD_Map, dtype=bool)
        selected_centers = []
        for r, c in zip(sorted_rows, sorted_cols):
            if selection_mask[r, c]:
                continue
            selected_centers.append((r, c))
            r_start = max(r - Half_ROI_size, 0)
            r_end = min(r + Half_ROI_size + 1, selection_mask.shape[0])
            c_start = max(c - Half_ROI_size, 0)
            c_end = min(c + Half_ROI_size + 1, selection_mask.shape[1])
            selection_mask[r_start:r_end, c_start:c_end] = True
        for row, col in selected_centers:
            if (
                row - Half_ROI_size < 0
                or row + Half_ROI_size + 1 > Im_Size[0]
                or col - Half_ROI_size < 0
                or col + Half_ROI_size + 1 > Im_Size[1]
            ):
                continue
            roi = im10[
                row - Half_ROI_size : row + Half_ROI_size + 1,
                col - Half_ROI_size : col + Half_ROI_size + 1,
            ]
            ROIs.append(roi)
    ROIs_array = np.array(ROIs)
    Total_NPS_No = 0
    if ROIs_array.size:
        ROIs_array = np.transpose(ROIs_array, (1, 2, 0))
        Total_NPS_No = ROIs_array.shape[-1]
    return ROIs_array, Total_NPS_No


# Demo Main Program
import time

start = time.time()

# Load DICOM files from directory
Map_dir = r"\\mfad.mfroot.org\rchapp\eb028591\CT_CIC_Group_Server\Staff_Folders\Zhou_Zhongxing\Zhou_ZX"
dir1 = Map_dir + "\\For_Jarod\\L067_FD_1_0_B30F_0001\\"
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

# Prepare lesion signals for multiple contrast and size conditions
lesion_file = (
    Map_dir + "\\For_Jarod\\Liver_lesion_sample\\Patient02-411-920_Lesion1.mat"
)
psf_file = None  #'N:\\Zhou_ZX\\For_Jarod\\Liver_lesion_sample\\EID_PSF_Br44.mat'  # Set to None to simulate PSF
mtf50 = (
    0.434  # Spatial frequency (cycles/mm) where MTF=0.5; replace with CT-specific value
)
mtf10 = 0.730  # Spatial frequency (cycles/mm) where MTF=0.1; replace or set to None for Gaussian
Lesion_Contrasts = [-30, -30, -10, -30, -50]  # HU contrast levels
Lesion_Size = [3, 9, 6, 6, 6]  # Lesion diameters in mm
ROI_sizes_mm = [14, 19, 17, 17, 17]  # ROI sizes in mm for each condition
pixel_size = patient_FOV / patient_matrix  # Pixel size in mm (~0.6640625)


def round_to_nearest_odd(x):
    """Round to nearest odd integer for ROI sizes."""
    rounded = np.round(x).astype(int)
    return np.where(
        rounded % 2 == 0, np.where(rounded > x, rounded - 1, rounded + 1), rounded
    )


ROI_sizes = round_to_nearest_odd(
    np.array(ROI_sizes_mm) / pixel_size
)  # Convert mm to pixels
Lesion_sigs = []
for i in range(len(Lesion_Contrasts)):
    lesion_con_target = Lesion_Contrasts[i]
    target_lesion_width = Lesion_Size[i]
    ROI_size = ROI_sizes[i]
    # Generate lesion signal: scale lesion, adjust contrast, convolve with PSF
    Lesion_sig = prepare_Lesion_sig(
        lesion_file,
        psf_file,
        lesion_con_target,
        target_lesion_width,
        patient_FOV,
        patient_matrix,
        ROI_size,
        mtf50=mtf50,
        mtf10=mtf10,
    )
    Lesion_sigs.append(Lesion_sig)

# Calculate CTDIvol and water-equivalent diameter (Dw)
SliceLocation_end = 0
CTDI_All = np.zeros(n_images)
Dw = np.zeros(n_images)
pixel_roi = dx * dy * 100
for i in range(n_images):
    dcm_info = dcmread(filepaths[i])
    CTDI_All[i] = dcm_info.CTDIvol
    im1 = dcm_info.pixel_array
    im10 = im1 * dcm_info.RescaleSlope + dcm_info.RescaleIntercept
    Mask = im10 >= -260
    Mask22 = ndimage.binary_fill_holes(Mask).astype(bool)
    labeled_image, num_labels = measure.label(Mask22, return_num=True)
    if num_labels > 0:
        largest_region = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
        binaryImage = labeled_image == largest_region
    else:
        binaryImage = np.zeros_like(Mask22, dtype=bool)
    pixel_No = np.sum(binaryImage)
    A_roi = pixel_No * pixel_roi
    im10_pat = im10[binaryImage]
    HU_Mean = np.mean(im10_pat)
    Dw[i] = 2 * np.sqrt((HU_Mean / 1000 + 1) * A_roi / np.pi)
    if i == n_images - 1:
        SliceLocation_end = dcm_info.SliceLocation

# Compute dose metrics: SSDE and DLP
Mean_CTDI_All = np.mean(CTDI_All)
Mean_Dw = np.mean(Dw) / 10
para_a = 3.704369
para_b = 0.03671937
f = para_a * np.exp(-para_b * Mean_Dw)
SSDE = f * Mean_CTDI_All
Scan_len = (SliceLocation_end - SliceLocation1) / 10
DLP_CTDIvol_L = Scan_len * Mean_CTDI_All
DLP_SSDE = Scan_len * SSDE
MTF_10p = 7.30  # Replace with actual MTF 10% value
ROI_size_N = round(0.6 / dx)
Half_ROI_size_N = round(np.floor(ROI_size_N / 2))
Thr1 = 0
Thr2 = 150
numResample = 500
internalNoise = 2.25
Resampling_method = "Bootstrap"


# Configure Gabor channels for CHO
class Chnl:
    Chnl_Toggle = "Gabor"


Chnl_Toggle = "Gabor"
Chnl.Chnl_Toggle = Chnl_Toggle
if Chnl.Chnl_Toggle == "Gabor":
    Gabor_passband = "[[1/64,1/32], [1/32,1/16]]"
    Gabor_theta = "[0, pi/2]"
    Gabor_beta = "0"
    Chnl = channel_selection(Chnl, Gabor_passband, Gabor_theta, Gabor_beta)
elif Chnl.Chnl_Toggle == "Laguerre-Gauss":
    LG_order = 6
    LG_orien = 3
    Chnl = channel_selection(Chnl, LG_order, LG_orien)

# Process image sections for NPS and CHO analysis
N_sub = round(50 / Sliceinterval)
All_dps = np.zeros([n_images // N_sub - 2, 5])
Noise_level_local = np.zeros([n_images // N_sub - 2, 1])
Total_NPS_Num = 0
NPS_1D_sum = 0
Spatial_freq = []
noise_level_sum = 0
NPS_Cal = True
Padding_size = 2 * round(1.2 / dx)
Padding_size = int(Padding_size) if Padding_size % 2 == 0 else int(Padding_size + 1)

for mm in range(1, n_images // N_sub - 1):
    N1 = (mm - 1) * N_sub
    N2 = mm * N_sub
    Intel_images = np.zeros(
        [Im_Size[0] + Padding_size, Im_Size[1] + Padding_size, N2 + N_sub * 2 - N1],
        dtype=np.float32,
    )
    Intel_images_Square = np.zeros(
        [Im_Size[0] + Padding_size, Im_Size[1] + Padding_size, N2 + N_sub * 2 - N1],
        dtype=np.float32,
    )
    Intel_Edge = np.zeros(
        [Im_Size[0] + Padding_size, Im_Size[1] + Padding_size, N2 + N_sub * 2 - N1],
        dtype=np.float32,
    )
    Images_Section = np.zeros(
        [Im_Size[0], Im_Size[1], N2 + N_sub * 2 - N1], dtype=np.float32
    )
    for nn in range(N1, N2 + N_sub * 2):
        dcm_info = dcmread(filepaths[nn])
        im1 = dcm_info.pixel_array
        im10 = im1 * dcm_info.RescaleSlope + dcm_info.RescaleIntercept
        Edges = feature.canny(im10, sigma=5)
        Images_Section[:, :, nn - N1] = im10
        Intel_Edge[:, :, nn - N1] = Integral_image(
            Edges, [Padding_size, Padding_size], "replicate"
        )
        Intel_images[:, :, nn - N1] = Integral_image(
            im10, [Padding_size, Padding_size], "replicate"
        )
        Intel_images_Square[:, :, nn - N1] = Integral_image(
            im10**2, [Padding_size, Padding_size], "replicate"
        )
    STD_all = calculate_std_dev(
        Intel_images,
        Intel_images_Square,
        Intel_Edge,
        ROI_size_N,
        Padding_size,
        Im_Size,
        N1,
        N2,
        N_sub,
        Thr1,
        Thr2,
        Edges,
        corrected=False,
    )
    max_value = np.nanmax(STD_all)
    h_Values, edges = np.histogram(STD_all.flatten(), bins=np.arange(0, max_value, 0.2))
    whichbin_SD = np.argmax(h_Values)
    bin_edge = edges[whichbin_SD]
    Noise_level_local[mm - 1] = bin_edge
    if NPS_Cal:
        ROI_All_NPS, Total_NPS_No = extract_ROIs(
            STD_all, bin_edge, Half_ROI_size_N, Images_Section, Im_Size
        )
        Spatial_freq, NPS_1D_sum, noise_level_sum, unit = ROI_to_NPS_Sum(
            ROI_size_N, ROI_All_NPS[:, :, 0:200], dx, dy
        )
        Total_NPS_Num = 200
        NPS_Cal = False
    del STD_all
    gc.collect()
    Left_Area = np.sum(h_Values[:whichbin_SD])
    Two_sigma_area = np.zeros(whichbin_SD)
    Two_sigma_area[0] = Left_Area
    temp_area = Left_Area
    for b_idx in range(1, whichbin_SD):
        Two_sigma_area[b_idx] = temp_area - h_Values[b_idx - 1]
        temp_area = temp_area - h_Values[b_idx - 1]
    Ratios = Two_sigma_area / Left_Area
    target_ratio = 0.9544
    temp_dis = np.abs(target_ratio - Ratios)
    closest = Ratios[np.argmin(temp_dis)]
    loc = np.where(Ratios == closest)[0][0]
    gap = whichbin_SD - loc
    Thre_SD = edges[whichbin_SD + gap]
    all_dps = []
    for ii in range(len(Lesion_sigs)):
        lesion_sig = Lesion_sigs[ii]
        ROI_size = ROI_sizes[ii]
        lesion_con_target = Lesion_Contrasts[ii]
        Half_ROI_size = round(np.floor(ROI_size / 2))
        if ii in [3, 4]:
            print("use same ROI_All")
        else:
            STD_map_all = calculate_std_dev(
                Intel_images,
                Intel_images_Square,
                Intel_Edge,
                ROI_size,
                Padding_size,
                Im_Size,
                N1,
                N2,
                N_sub,
                Thr1,
                Thr2,
                Edges,
                corrected=False,
            )
            ROI_All, Total_NPS_No = extract_ROIs(
                STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size
            )
        Noise_ROI = ROI_All[:, :, 0:Total_NPS_No]
        depth = Noise_ROI.shape[2]
        for i in range(depth):
            temp = Noise_ROI[:, :, i]
            Noise_ROI[:, :, i] = temp - np.mean(temp)
        N_total = Noise_ROI.shape[2]
        sample_idx = np.random.permutation(N_total)
        channelMatrix = ChannelMatrix_Generation(Chnl, ROI_size)
        bkg_ordered = np.reshape(
            Noise_ROI[:, :, sample_idx], (ROI_size**2, len(sample_idx))
        )
        sig_true = np.reshape(lesion_sig, (ROI_size**2, 1))
        dp = CHO_patient_with_resampling(
            sig_true, bkg_ordered, channelMatrix, internalNoise, Resampling_method
        )
        all_dps.append(dp)
    All_dps[mm - 1, :] = all_dps
    del Intel_images, Intel_images_Square, Intel_Edge, Images_Section
    gc.collect()

# Compute final metrics and plot NPS
Mean_loc_dps = np.mean(All_dps, axis=-1)
Mean_All_dps = np.mean(Mean_loc_dps)
NPS_1D = NPS_1D_sum / Total_NPS_Num
Ave_noise_level = noise_level_sum / Total_NPS_Num
end = time.time()
print(end - start)

# Plot NPS
fig, ax = plt.subplots()
ax.plot(Spatial_freq, NPS_1D, "b", label="average")
ax.set_xlabel("Spatial frequency (1/cm)", fontsize=16)
ax.set_ylabel("NPS (HU^2 cm^2)", fontsize=16)
ax.grid(True)
ax.legend()
plt.show()
fav, peakfrequency, k, min10percent_frequency = NPS_statistics(NPS_1D, unit)
print(f"peakfrequency = {peakfrequency:.3f}")
print(f"averagefrequency = {fav:.3f}")
print(f"min10percent_frequency = {min10percent_frequency:.3f}")
print(f"average noise level = {Ave_noise_level:.3f}")
