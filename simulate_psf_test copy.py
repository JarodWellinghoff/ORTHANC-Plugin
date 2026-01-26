import numpy as np
import matplotlib.pyplot as plt
import os
from pydicom import dcmread
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.io as sio
from scipy.signal import convolve2d
from scipy.optimize import minimize
from scipy.interpolate import interp1d


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


def simulate_psf_damped_cosine(mtf50, mtf10=None, size=513, wire_fov_mm=50.0):
    """
    Damped-cosine PSF:   exp(-r/τ) * cos(k·r)
    • mtf10 is None → pure Gaussian (k=0) with analytic τ from mtf50
    • mtf10 given   → optimise τ and k to hit MTF(mtf50)=0.5 and MTF(mtf10)=0.1
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    dx = wire_fov_mm / size
    x = np.linspace(-half, half, size) * dx
    y = np.linspace(-half, half, size) * dx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # ----- radial MTF from a 2-D PSF (same helper used by the old code) -----
    def compute_mtf_from_psf(psf):
        psf = psf / psf.sum()
        FT = np.fft.fft2(np.fft.ifftshift(psf))
        mtf2d = np.fft.fftshift(np.abs(FT))
        mtf2d /= np.max(mtf2d)
        N = psf.shape[0]
        centre = N // 2
        fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
        Fx, Fy = np.meshgrid(fx, fy)
        F = np.sqrt(Fx**2 + Fy**2)
        f_rad = F[centre, centre:]
        mtf_rad = mtf2d[centre, centre:]
        return interp1d(
            f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
        )

    # ----- Case 1: only mtf50 → Gaussian (k = 0) -----
    if mtf10 is None:
        # Analytic σ for a Gaussian that gives MTF(mtf50) = 0.5
        sigma = np.sqrt(-np.log(0.5) / (2 * np.pi**2 * mtf50**2))
        psf = np.exp(-(R**2) / (2 * sigma**2))
        psf /= psf.sum()
        return psf, dx

    # ----- Case 2: optimise τ and k for both targets -----
    def objective(params):
        tau, k = params
        if tau <= 0 or k < 0:
            return 1e12
        # Build PSF (clip negatives – the papers keep them, but for low-contrast
        # lesion work a non-negative PSF is usually preferred)
        psf_trial = np.exp(-R / tau) * np.cos(k * R)
        psf_trial = np.maximum(psf_trial, 0.0)
        mtf_func = compute_mtf_from_psf(psf_trial)
        err50 = (mtf_func(mtf50) - 0.5) ** 2
        err10 = (mtf_func(mtf10) - 0.1) ** 2
        return err50 + err10

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


def compute_presampling_mtf(psf, dx_psf):
    """
    Compute radial pre-sampling MTF from a simulated PSF.

    Returns:
    - freq: starts at 0.0 (cycles/mm)
    - mtf:  starts at 1.0, decreases to 0
    """
    from scipy.interpolate import interp1d

    # 1. Normalize PSF to sum = 1
    psf_norm = psf / psf.sum()

    # 2. FFT and shift zero-frequency to center
    psf_shift = np.fft.ifftshift(psf_norm)
    FT = np.fft.fft2(psf_shift)
    mtf2d = np.fft.fftshift(np.abs(FT))  # Center at [N//2, N//2]

    # 3. Normalize by MAX (which is DC component at center)
    mtf2d /= np.max(mtf2d)  # Now max = 1.0

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
    """
    Prepare lesion signal (scale → contrast → PSF convolution).

    *If psf_file is None* → a PSF is **simulated on a 50 mm FOV**
    and then resampled to the patient pixel spacing.
    """
    # --------------------------------------------------------------
    # 1. Load / create lesion VOI
    # --------------------------------------------------------------
    patient_data = sio.loadmat(lesion_file)
    lesion = patient_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
    mask = patient_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]
    loc = round(lesion.shape[-1] / 2)
    L0 = lesion[:, :, loc]
    M0 = mask[:, :, loc].astype(bool)

    # --------------------------------------------------------------
    # 2. Scale lesion to target physical size
    # --------------------------------------------------------------
    lesion_pixels_width_input = max_consecutive_ones_2d(M0)
    lesion_HU = np.mean(L0[M0])
    L0 = L0 - (lesion_HU - lesion_con_target)
    L0[~M0] = 0

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

    # --------------------------------------------------------------
    # 3. PSF handling
    # --------------------------------------------------------------
    pixel_spacing_patient = patient_FOV / patient_matrix  # mm/pixel

    if psf_file is None:  # ---- SIMULATED PSF ----
        if mtf50 is None:
            raise ValueError("mtf50 must be supplied when psf_file is None")
        PSF_sim, dx_psf = simulate_psf_damped_cosine(
            mtf50=mtf50, mtf10=mtf10, size=513, wire_fov_mm=50.0
        )  # 50 mm FOV / 513 ≈ 0.0973 mm/pixel → very fine sampling → accurate MTF up to ~5.14 cycles/mm (Nyquist)

        # Compute presampling MTF
        freq, mtf, mtf_func = compute_presampling_mtf(PSF_sim, dx_psf)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(7, 4.5))
        # plt.plot(freq, mtf, 'b-', linewidth=2, label='Presampling MTF')
        # plt.show()

        # ----- resample simulated PSF to patient grid -----
        scaling_factor = dx_psf / pixel_spacing_patient
        os0_psf = np.floor(np.array(PSF_sim.shape) * scaling_factor).astype(int)
        os0_psf = np.maximum(os0_psf, 15)  # safety
        os0_psf = os0_psf + (os0_psf % 2 == 0)  # odd size

        # original grid (in *pixel* indices of the simulated PSF)
        x0_psf = np.linspace(0, PSF_sim.shape[1] - 1, PSF_sim.shape[1])
        y0_psf = np.linspace(0, PSF_sim.shape[0] - 1, PSF_sim.shape[0])
        X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

        # target grid
        x_out_psf = np.linspace(0, PSF_sim.shape[1] - 1, os0_psf[1])
        y_out_psf = np.linspace(0, PSF_sim.shape[0] - 1, os0_psf[0])
        XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

        PSF_end = interpolate_grid(X0_psf, Y0_psf, PSF_sim, XX_psf, YY_psf, "linear")
    else:  # ---- MEASURED PSF ----
        psf_data = sio.loadmat(psf_file)
        PSF = psf_data["PSF"]

        wire_fov_mm = 50.0
        wire_matrix_size = 512
        dx_psf = wire_fov_mm / wire_matrix_size  # measured pixel size

        # Compute presampling MTF
        freq, mtf, mtf_func = compute_presampling_mtf(PSF, dx_psf)

        scaling_factor = wire_fov_mm / patient_FOV / 4

        os0_psf = np.floor(np.array(PSF.shape) * scaling_factor).astype(int)
        # os0_psf = np.maximum(os0_psf, 15)
        # os0_psf = os0_psf + (os0_psf % 2 == 0)

        x0_psf = np.linspace(0, PSF.shape[1] - 1, PSF.shape[1])
        y0_psf = np.linspace(0, PSF.shape[0] - 1, PSF.shape[0])
        X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

        x_out_psf = np.linspace(0, PSF.shape[1] - 1, os0_psf[1])
        y_out_psf = np.linspace(0, PSF.shape[0] - 1, os0_psf[0])
        XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

        PSF_end = interpolate_grid(X0_psf, Y0_psf, PSF, XX_psf, YY_psf, "linear")

    PSF_end /= PSF_end.sum()  # final normalisation

    # --------------------------------------------------------------
    # 4. Convolution & crop to ROI
    # --------------------------------------------------------------
    Lesion_ext = np.zeros((80, 80))
    r0 = (Lesion_ext.shape[0] - Lesion.shape[0]) // 2
    c0 = (Lesion_ext.shape[1] - Lesion.shape[1]) // 2
    Lesion_ext[r0 : r0 + Lesion.shape[0], c0 : c0 + Lesion.shape[1]] = Lesion

    Lesion_conv = convolve2d(Lesion_ext, PSF_end, mode="full")

    r0 = (Lesion_conv.shape[0] - ROI_size) // 2
    c0 = (Lesion_conv.shape[1] - ROI_size) // 2
    Lesion_conv_end = Lesion_conv[r0 : r0 + ROI_size, c0 : c0 + ROI_size]

    return Lesion_conv_end, freq, mtf


def round_to_nearest_odd(x):
    """Round to nearest odd integer for ROI sizes."""
    rounded = np.round(x).astype(int)
    return np.where(
        rounded % 2 == 0, np.where(rounded > x, rounded - 1, rounded + 1), rounded
    )


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
psf_file = Map_dir + "\\For_Jarod\\Liver_lesion_sample\\EID_PSF_Br44.mat"
# Set to None to simulate PSF
# Spatial frequency (cycles/mm) where MTF=0.5; replace with CT-specific value
mtf50 = 0.434
# Spatial frequency (cycles/mm) where MTF=0.1; replace or set to None for Gaussian
mtf10 = 0.730
Lesion_Contrasts = [-30, -30, -10, -30, -50]  # HU contrast levels
Lesion_Size = [3, 9, 6, 6, 6]  # Lesion diameters in mm
ROI_sizes_mm = [14, 19, 17, 17, 17]  # ROI sizes in mm for each condition
pixel_size = patient_FOV / patient_matrix  # Pixel size in mm (~0.6640625)

ROI_sizes = round_to_nearest_odd(
    np.array(ROI_sizes_mm) / pixel_size
)  # Convert mm to pixels
Lesion_sigs = []
for i in range(len(Lesion_Contrasts)):
    lesion_con_target = Lesion_Contrasts[i]
    target_lesion_width = Lesion_Size[i]
    ROI_size = ROI_sizes[i]
    # Generate lesion signal: scale lesion, adjust contrast, convolve with PSF
    Lesion_sig, freq, mtf = prepare_Lesion_sig(
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
    plt.imshow(Lesion_sig, cmap="gray")
    plt.title(
        f"Lesion Contrast: {lesion_con_target} HU, Size: {target_lesion_width} mm"
    )
    plt.colorbar(label="HU")

    # show the figure of Presampling MTF of loaded PSF or simulated PSF

    # plt.figure(figsize=(7, 4.5))
    # plt.plot(freq, mtf, "b-", linewidth=2, label="Presampling MTF")
    plt.show()
    print("Presampling MTF plotted.")
