import os
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy.interpolate import griddata, interp1d
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

LESION_FILE = os.path.join(SRC_DIR, "src", "data", "Patient02-411-920_Lesion1.mat")
PSF_FILE = os.path.join(SRC_DIR, "src", "data", "EID_PSF_Br44.mat")


def simulate_psf_damped_cosine(mtf50, mtf10=None, size=513, wire_fov_mm=50.0):
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
    dx = wire_fov_mm / size

    # Create spatial grid in mm
    x = np.linspace(-half, half, size) * dx
    y = np.linspace(-half, half, size) * dx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # ----- Case 1: only mtf50 → Gaussian (k = 0) -----
    if mtf10 is None:
        # Analytic σ for a Gaussian that gives MTF(mtf50) = 0.5
        sigma = np.sqrt(-np.log(0.5) / (2 * np.pi**2 * mtf50**2))
        psf = np.exp(-(R**2) / (2 * sigma**2))
        psf /= psf.sum()
        return psf, dx

    fx = np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    Fx, Fy = np.meshgrid(fx, fy)
    F = np.sqrt(Fx**2 + Fy**2)
    f_rad = F[half, half:]

    # ----- radial MTF from a 2-D PSF (same helper used by the old code) -----
    def compute_mtf_from_psf(psf):
        psf = psf / psf.sum()
        FT = np.fft.fft2(np.fft.ifftshift(psf))
        mtf2d = np.fft.fftshift(np.abs(FT))
        mtf2d /= np.max(mtf2d)
        mtf_rad = mtf2d[half, half:]
        return interp1d(
            f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
        )

    # ----- Case 2: optimise τ and k for both targets -----
    def objective(params):
        tau, k = params
        if tau <= 0 or k < 0:
            return 1e12
        # Build PSF (clip negatives – the papers keep them, but for low-contrast
        # lesion work a non-negative PSF is usually preferred)
        psf_trial = np.exp(-R / tau) * np.cos(k * R)
        psf_trial_clipped = np.maximum(psf_trial, 0.0)
        mtf_func = compute_mtf_from_psf(psf_trial_clipped)
        err50 = (100 * (mtf_func(mtf50) - 0.5)) ** 2
        err10 = (100 * (mtf_func(mtf10) - 0.1)) ** 2
        errTotal = err50 + err10
        print(f"Current parameters: τ = {tau:.4f} mm, k = {k:.4f} rad/mm")
        print(f"  Error (MTF50): {err50:.4f}")
        print(f"  Error (MTF10): {err10:.4f}")
        print(f"  Error (Total): {errTotal:.4f}")

        return errTotal  # balance the two

    # Reasonable starting point (Gaussian width + a mild oscillation)
    tau0 = 1.0 / (np.pi * mtf50)  # ~Gaussian width
    k0 = 0.6 * np.pi * mtf50  # small ringing
    res = minimize(
        objective,
        [tau0, k0],
        bounds=[(tau0 * 0.3, tau0 * 3.0), (0.0, 2.0 * np.pi * mtf50)],
        method="L-BFGS-B",
    )

    tau_opt, k_opt = res.x
    psf = np.exp(-R / tau_opt) * np.cos(k_opt * R)
    psf = np.maximum(psf, 0.0)  # keep non-negative
    psf /= psf.sum()

    # ----- verification (same style as the old function) -----
    mtf_func = compute_mtf_from_psf(psf)
    print(f"Damped-cosine PSF: τ = {tau_opt:.4f} mm, k = {k_opt:.4f} rad/mm")
    print(f"  MTF({mtf50:.3f}) = {mtf_func(mtf50):.4f} (target 0.5)")
    print(f"  MTF({mtf10:.3f}) = {mtf_func(mtf10):.4f} (target 0.1)")

    return psf, dx


def max_consecutive_ones_2d(bool_array_2d):
    """Find maximum consecutive ones in 2D array"""
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


def interpolate_grid(X0, Y0, L0, XX, YY, method):
    """Grid interpolation for lesion modeling"""
    X0 = np.asarray(X0)
    Y0 = np.asarray(Y0)
    L0 = np.asarray(L0)
    XX = np.asarray(XX)
    YY = np.asarray(YY)

    grid_points = (X0.flatten(), Y0.flatten())
    values = L0.flatten()
    Lesion = griddata(grid_points, values, (XX, YY), method=method)
    return Lesion


def compute_presampling_mtf(psf, dx_psf):
    """
    Compute radial pre-sampling MTF from a simulated PSF.

    Returns:
    - freq: starts at 0.0 (cycles/mm)
    - mtf:  starts at 1.0, decreases to 0
    """
    from scipy.interpolate import interp1d

    # 1. Normalize PSF to sum = 1
    psf_norm = psf / np.sum(np.abs(psf))

    # 2. FFT and shift zero-frequency to center
    psf_shift = np.fft.ifftshift(psf_norm)
    FT = np.fft.fft2(psf_shift)
    mtf2d = np.abs(np.fft.fftshift(FT)) / np.abs(FT[0, 0])
    # mtf2d = np.fft.fftshift(np.abs(FT))  # Center at [N//2, N//2]

    # 3. Normalize by MAX (which is DC component at center)
    # mtf2d /= np.max(mtf2d)  # Now max = 1.0

    # 4. Build radial frequency grid
    N = psf.shape[0]
    centre = N // 2

    # Frequency vectors (cycles/mm)
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx_psf))  # e.g., [-Nyq, ..., 0, ..., +Nyq]
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx_psf))
    Fx, Fy = np.meshgrid(fx, fy)
    F = np.sqrt(Fx**2 + Fy**2)

    # 5. Extract radial line from center to edge
    f_rad = F[centre, centre:]  # [0.0, df, 2df, ..., Nyq]
    mtf_rad = mtf2d[centre, centre:]  # [1.0, ..., ~0]

    # 6. Interpolant
    mtf_func = interp1d(
        f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
    )

    return f_rad, mtf_rad, mtf_func


def prepare_lesion_signal(
    lesion_data,
    psf_data,
    target_contrast_hu,
    target_size_mm,
    roi_size_mm,
    patient_fov_mm,
    patient_matrix_px,
    mtf50=None,
    mtf10=None,
):
    """Prepare lesion signal for CHO analysis"""
    patient_pixel_size = patient_fov_mm / patient_matrix_px
    roi_size_px = roi_size_mm / patient_pixel_size
    roi_size_px = np.round(roi_size_px).astype(int)
    if roi_size_px % 2 == 0:
        roi_size_px += 1

    # --- Extract lesion and mask volumes ---
    lesion_volume = lesion_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
    lesion_mask_volume = lesion_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

    # --- Select middle slice ---
    mid_slice_idx = round(lesion_volume.shape[-1] / 2)
    lesion_slice = lesion_volume[:, :, mid_slice_idx]
    mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

    # --- Compute lesion HU and adjust contrast ---
    lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)
    mean_lesion_hu = np.mean(lesion_slice[mask_slice])
    hu_difference = mean_lesion_hu - target_contrast_hu
    lesion_slice = lesion_slice - hu_difference
    lesion_slice[~mask_slice] = 0  # zero out background

    # --- Scale lesion to desired physical size ---
    target_width_pixels = target_size_mm / patient_pixel_size
    scale_factor = target_width_pixels / lesion_width_pixels_input

    input_height, input_width = lesion_slice.shape
    output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

    # --- Generate interpolation grids ---
    x_in = np.linspace(0, input_width - 1, input_width)
    y_in = np.linspace(0, input_height - 1, input_height)
    X_in, Y_in = np.meshgrid(x_in, y_in)

    x_out = np.linspace(0, input_width - 1, output_shape[1])
    y_out = np.linspace(0, input_height - 1, output_shape[0])
    X_out, Y_out = np.meshgrid(x_out, y_out)

    # --- Rescale lesion using interpolation ---
    scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

    # --- Load and rescale PSF ---
    if mtf50:
        psf_rescaled, dx_psf = simulate_psf_damped_cosine(mtf50, mtf10)

        # Compute presampling MTF
        freq, mtf, mtf_func = compute_presampling_mtf(psf_rescaled, dx_psf)

        # ----- resample simulated PSF to patient grid -----
        psf_rescaled_shape = psf_rescaled.shape
        scaling_factor = dx_psf / patient_pixel_size
        os0_psf = np.floor(np.array(psf_rescaled_shape) * scaling_factor).astype(int)
        # os0_psf = np.maximum(os0_psf, 15)  # safety
        # os0_psf = os0_psf + (os0_psf % 2 == 0)  # odd size

        # original grid (in *pixel* indices of the simulated PSF)
        x0_psf = np.linspace(0, psf_rescaled_shape[1] - 1, psf_rescaled_shape[1])
        y0_psf = np.linspace(0, psf_rescaled_shape[0] - 1, psf_rescaled_shape[0])
        X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

        # target grid
        x_out_psf = np.linspace(0, psf_rescaled_shape[1] - 1, os0_psf[1])
        y_out_psf = np.linspace(0, psf_rescaled_shape[0] - 1, os0_psf[0])
        XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

        psf_rescaled = interpolate_grid(
            X0_psf, Y0_psf, psf_rescaled, XX_psf, YY_psf, "linear"
        )
    elif psf_data:
        psf = psf_data["PSF"]
        wire_fov_mm = 50
        wire_matrix_size = 512
        dx_psf = wire_fov_mm / wire_matrix_size  # measured pixel size

        # Compute presampling MTF
        freq, mtf, mtf_func = compute_presampling_mtf(psf, dx_psf)

        scaling_factor = dx_psf / patient_pixel_size / 4

        os0_psf = np.floor(np.array(psf.shape) * scaling_factor).astype(int)
        # os0_psf = np.maximum(os0_psf, 15)
        # os0_psf = os0_psf + (os0_psf % 2 == 0)

        x0_psf = np.linspace(0, psf.shape[1] - 1, psf.shape[1])
        y0_psf = np.linspace(0, psf.shape[0] - 1, psf.shape[0])
        X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

        x_out_psf = np.linspace(0, psf.shape[1] - 1, os0_psf[1])
        y_out_psf = np.linspace(0, psf.shape[0] - 1, os0_psf[0])
        XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

        psf_rescaled = interpolate_grid(X0_psf, Y0_psf, psf, XX_psf, YY_psf, "linear")
    else:
        raise ValueError("Either PSF data or MTF50/MTF10 must be provided.")

    psf_rescaled /= psf_rescaled.sum()  # normalize PSF

    # --- Center lesion in padded array ---
    # lesion_padded = np.zeros((80, 80))
    lesion_padded = np.zeros((scaled_lesion.shape[0] + 50, scaled_lesion.shape[1] + 50))
    start_row = (lesion_padded.shape[0] - scaled_lesion.shape[0]) // 2
    start_col = (lesion_padded.shape[1] - scaled_lesion.shape[1]) // 2
    lesion_padded[
        start_row : start_row + scaled_lesion.shape[0],
        start_col : start_col + scaled_lesion.shape[1],
    ] = scaled_lesion

    # --- Convolve lesion with PSF (simulate system blur) ---
    lesion_convolved = convolve2d(lesion_padded, psf_rescaled, mode="full")

    # --- Crop to ROI size ---
    crop_start_row = (lesion_convolved.shape[0] - roi_size_px) // 2
    crop_start_col = (lesion_convolved.shape[1] - roi_size_px) // 2
    lesion_roi = lesion_convolved[
        crop_start_row : crop_start_row + roi_size_px,
        crop_start_col : crop_start_col + roi_size_px,
    ]

    return lesion_roi, freq, mtf


def create_lesion_model(set, index, mtf50, mtf10, recon_diameter_mm, rows):
    """Create a lesion model for analysis"""

    lesion_contrasts = [-30, -30, -10, -30, -50]
    lesion_sizes = [3, 100, 6, 6, 6]
    roi_sizes = [int(np.ceil((4 * x) / 3) + 17) for x in lesion_sizes]

    if set == "low-contrast":
        lesion_contrasts = [int(c / 2) for c in lesion_contrasts]
    elif set == "high-contrast":
        lesion_contrasts = [int(c * 2) for c in lesion_contrasts]

    lesion_file_data = sio.loadmat(LESION_FILE)
    psf_data = sio.loadmat(PSF_FILE)

    lesion_contrast = lesion_contrasts[index]
    lesion_size = lesion_sizes[index]
    roi_size = roi_sizes[index]

    lesion_roi, _, _ = prepare_lesion_signal(
        lesion_file_data,
        psf_data,
        lesion_contrast,
        lesion_size,
        roi_size,
        recon_diameter_mm,
        rows,
        mtf50,
        mtf10,
    )

    return lesion_roi, lesion_contrast, lesion_size, roi_size


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    lesion_set = "standard"
    recon_diameter_mm = 340
    rows = 512

    mtf50s = [None, 0.434, 0.434]
    mtf10s = [None, None, 0.730]
    figure_titles = [
        "Loaded PSF from file",
        "Gaussian PSF (mtf50=0.434)",
        "Damped Cosine PSF (mtf50=0.434, mtf10=0.730)",
    ]
    # mtf50 = 0.434  # cycles/mm
    # mtf10 = 0.730  # cycles/mm

    for mtf50, mtf10, title in zip(mtf50s, mtf10s, figure_titles):
        plt.figure(figsize=(15, 3))
        plt.suptitle(title)
        for lesion_index in range(5):
            plt.subplot(1, 5, lesion_index + 1)
            lesion_roi, lesion_contrast, lesion_size, roi_size = create_lesion_model(
                lesion_set, lesion_index, mtf50, mtf10, recon_diameter_mm, rows
            )

            plt.imshow(lesion_roi, cmap="gray")
            plt.title(
                f"Lesion {lesion_index + 1}: {lesion_contrast} HU, {lesion_size} mm"
            )
            plt.axis("off")
    plt.show()
