# #!/usr/bin/env python
# # coding: utf-8

# """
# Unified CHO Patient-Specific Analysis Module

# This module combines both Global Noise and Full CHO analysis capabilities
# with optional progress tracking. It can run either:
# 1. Global Noise Analysis (simple, fast) - test=False
# 2. Full CHO Analysis (comprehensive) - test=True

# With optional progress reporting for live updates.
# """

# import numpy as np
# from typing import Callable, Optional, Tuple
# import os
# import time
# import math
# from scipy import ndimage
# from skimage import feature
# from skimage import measure
# from scipy.signal import convolve2d
# import json
# import scipy.io as sio
# from scipy.optimize import minimize
# from scipy.interpolate import griddata, interp1d
# from scipy.ndimage import zoom
# import gc
# from dicom_parser import DicomParser
# from numpy.lib.stride_tricks import sliding_window_view
# from ct_sliding_window import CTSlidingWindow
# from progress_tracker import progress_tracker

# # from python.CHO_Calculation_Patient_Specific_skimage_Canny_edge_v12 import Edges

# CONSTANT_RECON_FOV = 340
# CONSTANT_PATIENT_MATRIX = 512

# SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Prepare lesion signals
# LESION_FILE = os.path.join(SRC_DIR, "data", "Patient02-411-920_Lesion1.mat")
# PSF_FILE = os.path.join(SRC_DIR, "data", "EID_PSF_Br44.mat")


# def round_to_nearest_odd(x):
#     """Round to nearest odd integer for ROI sizes."""
#     rounded = np.round(x).astype(int)
#     return np.where(
#         rounded % 2 == 0, np.where(rounded > x, rounded - 1, rounded + 1), rounded
#     )


# def get_lesion_configuration(lesion_set):
#     """
#     Get lesion configuration based on the selected set
#     """
#     lesion_configs = {
#         "standard": {
#             "contrasts": [-30, -30, -10, -30, -50],
#             "sizes": [3, 9, 6, 6, 6],
#             "roi_sizes": [14, 19, 17, 17, 17],
#         },
#         "low-contrast": {
#             "contrasts": [-15, -15, -5, -15, -25],
#             "sizes": [3, 9, 6, 6, 6],
#             "roi_sizes": [14, 19, 17, 17, 17],
#         },
#         "high-contrast": {
#             "contrasts": [-60, -60, -20, -60, -50],
#             "sizes": [3, 9, 6, 6, 6],
#             "roi_sizes": [14, 19, 17, 17, 17],
#         },
#     }

#     return lesion_configs.get(lesion_set, lesion_configs["standard"])


# def normal_round(n):
#     if n - math.floor(n) < 0.5:
#         return math.floor(n)
#     return math.ceil(n)


# def convert_numpy_to_python(obj):
#     """Convert numpy data types to native Python types and handle NaN values"""
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         if np.isnan(obj):
#             return None  # Convert NaN to None (which becomes null in JSON)
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         # Handle NaN values in arrays
#         if obj.dtype.kind in ["f", "c"]:  # float or complex arrays
#             obj_clean = np.where(np.isnan(obj), None, obj)
#             return obj_clean.tolist()
#         else:
#             return obj.tolist()
#     elif isinstance(obj, (list, tuple)):
#         return [convert_numpy_to_python(item) for item in obj]
#     elif isinstance(obj, dict):
#         return {key: convert_numpy_to_python(value) for key, value in obj.items()}
#     elif (
#         np.isnan(obj)
#         if isinstance(obj, (int, float)) and not isinstance(obj, bool)
#         else False
#     ):
#         return None
#     else:
#         return obj


# def safe_json_dumps(obj):
#     """Convert object to JSON string, handling NaN values safely"""
#     converted = convert_numpy_to_python(obj)
#     return json.dumps(converted, allow_nan=False)


# def average_filter(image, window_size=[3, 3], padding="constant"):
#     """
#     AVERAGEFILTER 2-D mean filtering.
#     Performs mean filtering of a 2-dimensional matrix/image using the integral image method.
#     """
#     if len(image.shape) != 2:
#         raise ValueError("The input image must be a two-dimensional array.")

#     # Set up the window size
#     m, n = window_size

#     # Pad the image
#     pad_width = ((m // 2, m // 2), (n // 2, n // 2))

#     if padding == "circular":
#         padded_image = np.pad(image, pad_width, mode="wrap")
#     elif padding == "replicate":
#         padded_image = np.pad(image, pad_width, mode="edge")
#     elif padding == "symmetric":
#         padded_image = np.pad(image, pad_width, mode="symmetric")
#     else:  # Default is 'constant' padding (zero padding)
#         padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

#     # Convert the image to float
#     imageD = padded_image.astype(np.float64)

#     # Calculate the integral image
#     t = np.cumsum(np.cumsum(imageD, axis=0), axis=1)

#     # Calculate mean values based on the integral image
#     imageI = t[m:, n:] + t[:-m, :-n] - t[m:, :-n] - t[:-m, n:]

#     # Normalize the imageI by the area of the window
#     imageI /= m * n
#     return imageI.astype(image.dtype)


# def stdfilter(image, window_size=[3, 3], corrected=False, padding="constant"):
#     """2-D standard deviation filtering."""
#     if len(window_size) != 2:
#         raise ValueError("Window_size must be a tuple of 2 values (M, N).")

#     m, n = window_size
#     if len(image.shape) != 2:
#         raise ValueError("The input image must be a two-dimensional array.")

#     # Decide whether to use corrected variance estimate
#     if corrected:
#         normalization_factor = m * n / (m * n - 1)  # denominator is 'n-1'
#     else:
#         normalization_factor = 1  # denominator is 'n'

#     # Convert image to float
#     image = image.astype(float)
#     # Mean value
#     mean = average_filter(image, window_size=window_size, padding=padding)
#     # Mean square
#     mean_square = average_filter(image**2, window_size=window_size, padding=padding)
#     # Standard deviation calculation
#     deviation = np.sqrt(normalization_factor * (mean_square - mean**2))

#     return deviation


# def round_to_nearest_even(num):
#     return round(num + 0.5) if num % 2 != 0 else num


# # Full CHO Analysis Functions (only loaded when needed)
# def poly2fit(x, y, z, n):
#     """2D polynomial fitting for full CHO analysis"""
#     if (
#         x.shape[0] != y.shape[0]
#         or x.shape[1] != y.shape[1]
#         or x.shape[0] != z.shape[0]
#         or x.shape[1] != z.shape[1]
#     ):
#         print("X,Y, and Z matrices must be the same size")

#     x = np.transpose(x)
#     y = np.transpose(y)
#     z = np.transpose(z)
#     x = x.flatten()
#     y = y.flatten()
#     z = z.flatten()

#     n = n + 1
#     k = 0
#     A = np.zeros((x.shape[0], 6))
#     i = n
#     while i >= 1:
#         for j in range(1, i + 1):
#             temp1 = np.power(x, i - j)
#             temp2 = np.power(y, j - 1)
#             A[:, k] = np.multiply(temp1, temp2)
#             k = k + 1
#         i = i - 1

#     p = np.linalg.lstsq(A, z)[0]
#     return p


# def subtractMean2D(im, method, psize):
#     """Subtract mean from 2D image"""
#     if method:  # method = 1 -> "polynomial fitting"
#         FOV = np.zeros(2)
#         im_sizeX = im.shape[0]
#         im_sizeY = im.shape[1]
#         FOV[0] = psize[0] * im_sizeX
#         FOV[1] = psize[1] * im_sizeY

#         x = np.arange(0, im_sizeX, 1) * psize[0] - FOV[0] / 2
#         y = np.arange(0, im_sizeY, 1) * psize[1] - FOV[1] / 2
#         X, Y = np.meshgrid(x, y)
#         P = poly2fit(X, Y, im, 1)
#         im = im - (P[0] * X + P[1] * Y + P[2])
#     else:  # method = 0 -> "subtract the mean"
#         im = im - np.mean(im.flatten())

#     return im


# def NPS_statistics(nps1d, unit):
#     """Calculate NPS statistics"""
#     peakfrequencyIndex = np.where(nps1d == np.max(nps1d))[0][0]
#     peakfrequency = peakfrequencyIndex * unit

#     n = len(nps1d)
#     fre_idx = list(range(0, n))
#     Spatial_freq = [x * unit for x in fre_idx]
#     p = nps1d / sum(np.array(nps1d))
#     p = np.squeeze(p)
#     Spatial_freq = np.array(Spatial_freq)
#     Spatial_freq = np.squeeze(Spatial_freq)
#     fav = sum(np.multiply(Spatial_freq, p))

#     minfrequencyIndex = np.where(nps1d < 0.10 * np.max(nps1d))[0]
#     min10percent_frequency = -1
#     try:
#         freq = np.where(minfrequencyIndex > peakfrequencyIndex)[0][0]
#     except IndexError:
#         freq = -1
#     if freq >= 0:
#         min10percent_frequency = minfrequencyIndex[freq] * unit

#     npoint = 10
#     k = np.polyfit(np.arange(npoint) * unit, nps1d[:npoint], 1)

#     return fav, peakfrequency, k, min10percent_frequency


# def pad_to_shape_centered(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
#     """Pad array with zeros to target shape, centering the array"""
#     pad_width = []
#     for s, t in zip(arr.shape, target_shape):
#         total = max(t - s, 0)
#         before = total // 2
#         after = total - before
#         pad_width.append((before, after))
#     return np.pad(arr, pad_width, mode="constant", constant_values=0)


# def ROI_to_NPS_Sum(ROI_size, ROI_All, dx, dy):
#     """Calculate NPS from ROI data"""
#     if ROI_size < 20:
#         nnn = 8 * 20
#     else:
#         nnn = 8 * ROI_size

#     cc = int(np.rint(nnn / 2))
#     unit = 1 / (dx * nnn)

#     oo = np.linspace(0, 2 * np.pi, 360, endpoint=False)
#     radii = np.arange(cc).reshape((cc, 1))

#     x_offset = radii * np.cos(oo)
#     y_offset = radii * np.sin(oo)

#     xi_all = np.rint(x_offset + cc - 1).astype(np.int32)
#     yi_all = np.rint(y_offset + cc - 1).astype(np.int32)

#     valid_mask = (xi_all >= 0) & (xi_all < nnn) & (yi_all >= 0) & (yi_all < nnn)

#     NPS_1D_sum = np.zeros((cc, 1), dtype=np.float32)
#     noise_level_sum = 0.0

#     num_slices = ROI_All.shape[-1]
#     for jj in range(num_slices):
#         roi = ROI_All[:, :, jj]
#         psize = (dx, dy)
#         roi = subtractMean2D(roi, 1, psize)
#         noise_level_sum += np.std(roi)

#         roipad = pad_to_shape_centered(roi, (nnn, nnn))

#         roipad_fft = np.fft.fftshift(np.abs(np.fft.fft2(roipad)))
#         roipad_fft = roipad_fft**2 * dx * dy / (ROI_size * ROI_size)
#         nps2D = roipad_fft

#         polar1 = np.zeros((cc, 360), dtype=np.float32)
#         polar1[valid_mask] = nps2D[xi_all[valid_mask], yi_all[valid_mask]]

#         nps1d = np.mean(polar1, axis=1).reshape((cc, 1))
#         NPS_1D_sum += nps1d

#     Spatial_freq = np.arange(cc).astype(np.float32) * unit
#     return Spatial_freq, NPS_1D_sum, noise_level_sum, unit


# # Additional full CHO analysis functions would go here
# # (Laguerre2D, Gabor2D, channel_selection, etc. - included in full for brevity)


# def hu_to_image(hu_array):
#     """Convert HU values to image intensities"""
#     # Normalize HU values to [0, 1]
#     hu_min = np.min(hu_array)
#     hu_max = np.max(hu_array)
#     hu_normalized = (hu_array - hu_min) / (hu_max - hu_min)
#     # Scale to [0, 255] and convert to uint8
#     return (hu_normalized * 255).astype(np.uint8)


# def run_global_noise_analysis(files, config):
#     """Run Global Noise CHO Analysis"""

#     custom_params = config["custom_parameters"]
#     start = time.time()

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             15,
#             "Starting Global Noise CHO calculation...",
#             "preprocessing",
#         )

#     # Extract DICOM information
#     dcm_parser = DicomParser(files)
#     patient, study, scanner, series, ct = dcm_parser.extract_core()

#     n_images = len(files)
#     first_file = files[0]
#     second_file = files[1]
#     last_file = files[-1]

#     first_slice_location = first_file.SliceLocation
#     last_slice_location = last_file.SliceLocation
#     slice_interval = series["slice_interval_mm"] / 10
#     rows = int(series.get("rows", 512))
#     cols = int(series.get("columns", 512))
#     dx_mm, dy_mm = series["pixel_spacing_mm"]

#     dx_cm = dx_mm / 10
#     pixel_roi_mm2 = dx_mm * dy_mm

#     roi_diameter = round(0.6 / dx_cm)
#     roi_radius_round = round(np.floor(roi_diameter / 2))
#     Thr1 = 0
#     Thr2 = 150

#     img_size = (rows, cols)
#     sample_interval = round(3 / slice_interval)
#     if slice_interval >= 3:
#         sample_interval = 1

#     n_sel_images = (n_images + sample_interval - 1) // sample_interval

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             15,
#             f"Analyzing {n_sel_images} selected images for noise calculation",
#             "preprocessing",
#         )

#     ROI_size2 = round_to_nearest_even(roi_diameter)
#     window_gen = CTSlidingWindow(
#         window_length=15,  # 1 slice
#         step_size=0.3,  # 3mm = 0.3cm
#         padding_size=ROI_size2,
#         sigma=5,
#         window_unit="cm",
#         step_unit="cm",
#         use_cache=False,
#     )
#     total_sections = window_gen.get_num_windows(files)
#     ctdi_all = np.zeros(total_sections)
#     location = np.zeros(total_sections)
#     dw = np.zeros(total_sections)
#     coronal_view = np.zeros((cols, total_sections))
#     STD_all = np.zeros([rows, cols, total_sections], dtype=np.float32)
#     for window, metadata in window_gen.generate_windows(files):
#         # for i in range(0, n_images, sample_interval):
#         i = metadata["slice_indices"][0]
#         idx = metadata["window_idx"]
#         if (
#             config["report_progress"]
#             and config["series_uuid"]
#             and (i % 10 == 0 or i == n_images - 1)
#         ):
#             # 25 = progress during analysis
#             # 85 = progress after analysis
#             progress_percentage = 25 + ((i / n_images) * (85 - 25))
#             progress_tracker.update_progress(
#                 config["series_uuid"],
#                 int(progress_percentage),
#                 f"Processing slice {i+1}/{n_images}",
#                 "analysis",
#             )

#         curr_file = files[i]
#         ctdi_all[idx], dw[idx], coronal_view[:, idx] = compute_image_analysis(
#             curr_file, pixel_roi_mm2
#         )
#         location[idx] = metadata["relative_mid_cm"]
#         Edges = window["edges"]
#         Intel_Edge = window["integral_edges"]
#         Intel_images = window["integral_images"]
#         Intel_images_Square = window["integral_images_square"]
#         STD_all[:, :, idx] = np.squeeze(
#             calculate_std_dev(
#                 Intel_images,
#                 Intel_images_Square,
#                 Intel_Edge,
#                 roi_diameter,
#                 ROI_size2,
#                 img_size,
#                 Thr1,
#                 Thr2,
#                 Edges,
#                 corrected=False,
#             )
#         )
#     window_gen.clear_cache()

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 85, "Calculating final metrics...", "finalizing"
#         )

#     max_value = np.nanmax(STD_all)
#     h_Values, edges = np.histogram(STD_all.flatten(), bins=np.arange(0, max_value, 0.2))
#     maxcount = np.max(h_Values)
#     whichbin_SD = np.argmax(h_Values)
#     global_noise_level = edges[whichbin_SD]

#     # Calculate dose metrics
#     mean_ctdi_all = np.mean(ctdi_all)
#     mean_dw = np.mean(dw) / 10
#     # Body parameters for SSDE calculation
#     para_a = 3.704369
#     para_b = 0.03671937
#     f = para_a * math.exp(-para_b * mean_dw)
#     mean_ssde = f * mean_ctdi_all
#     f = para_a * math.exp(-para_b * (dw / 10))
#     ssde = f * ctdi_all
#     scan_length_cm = (last_slice_location - first_slice_location) / 10
#     dlp_ctdi_vol = scan_length_cm * mean_ctdi_all
#     dlp_ssde = scan_length_cm * mean_ssde
#     mtf_10p = 7.30
#     print(f"SSDE: {ssde}")
#     # location = np.linspace(0, scan_length_cm, num=len(ctdi_all)).tolist()

#     processing_time = time.time() - start

#     series["uuid"] = config["series_uuid"]
#     series["image_count"] = n_images
#     series["scan_length_cm"] = scan_length_cm

#     results = {
#         "average_frequency": None,
#         "average_index_of_detectability": None,
#         "average_noise_level": None,
#         "cho_detectability": None,
#         "ctdivol": ctdi_all,
#         "ctdivol_avg": mean_ctdi_all,
#         "dlp": dlp_ctdi_vol,
#         "dlp_ssde": dlp_ssde,
#         "dw": dw,
#         "dw_avg": mean_dw,
#         "location": location.tolist(),
#         "location_sparse": None,
#         "noise_level": None,
#         "nps": None,
#         "peak_frequency": None,
#         "percent_10_frequency": None,
#         "processing_time": processing_time,
#         "spatial_frequency": None,
#         "spatial_resolution": None,
#         "ssde": mean_ssde,
#         "coronal_view": coronal_view,
#         "global_noise_level": global_noise_level,
#     }

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 95, "Saving results to database...", "finalizing"
#         )

#     # Convert results and save to database
#     converted_results = convert_numpy_to_python(results)

#     if custom_params["saveResults"]:
#         try:
#             from results_storage import cho_storage

#             success = cho_storage.save_results(
#                 patient, study, scanner, series, ct, converted_results
#             )

#             if success:
#                 print(
#                     f"Global noise results saved to database for series {series['series_instance_uid']}"
#                 )
#             else:
#                 print(
#                     f"Failed to save global noise results to database for series {series['series_instance_uid']}"
#                 )

#         except Exception as e:
#             print(f"Error saving to database: {str(e)}")

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             100,
#             "Global Noise CHO calculation completed successfully",
#             "completed",
#         )
#         progress_tracker.complete_calculation(
#             config["series_uuid"],
#             {
#                 "patient": patient,
#                 "study": study,
#                 "scanner": scanner,
#                 "series": series,
#                 "ct": ct,
#                 "results": converted_results,
#             },
#         )

#     if custom_params["deleteAfterCompletion"]:
#         import orthanc

#         orthanc.RestApiDelete(f'/series/{config["series_uuid"]}')
#         print(f"Requested deletion of series {config['series_uuid']}")

#     print(f"Global Noise processing time: {processing_time:.2f} seconds")
#     # del results['coronal_view']
#     return {
#         "patient": patient,
#         "study": study,
#         "scanner": scanner,
#         "series": series,
#         "ct": ct,
#         "results": converted_results,
#     }


# # Full CHO Analysis Functions
# def Laguerre2D(order, a, b, cx, cy, X, Y):
#     """Generate Laguerre-Gauss channel function"""
#     val1 = np.zeros(X.size)
#     ga = 2 * np.pi * ((X - cx) ** 2 / (a**2) + (Y - cy) ** 2 / (b**2))
#     for jp in range(order + 1):
#         val1 = val1 + (-1) ** jp * np.prod(np.linspace(1, order, order)) / (
#             np.prod(np.linspace(1, jp, jp))
#             * np.prod(np.linspace(1, order - jp, order - jp))
#         ) * (ga**jp) / np.prod(np.linspace(1, jp, jp))
#     channel_filter = np.exp(-ga / 2) * val1
#     return channel_filter


# def Gabor2D(fc, wd, theta, beta, cx, cy, X, Y):
#     """Generate Gabor channel function"""
#     channel_filter = np.exp(
#         -4 * np.log(2) * ((X - cx) ** 2 + (Y - cy) ** 2) / (wd**2)
#     ) * np.cos(
#         2 * np.pi * fc * ((X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)) + beta
#     )
#     return channel_filter


# def channel_selection(Chnl, inputArg1, inputArg2, inputArg3="0"):
#     """Configure CHO channel parameters"""
#     if Chnl.Chnl_Toggle == "Laguerre-Gauss":
#         if isinstance(inputArg1, int):
#             Chnl.LG_order = inputArg1
#         if isinstance(inputArg2, int):
#             Chnl.LG_orien = inputArg2

#     if Chnl.Chnl_Toggle == "Gabor":
#         if isinstance(inputArg1, str):
#             if inputArg1 == "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]":
#                 Chnl.Gabor_passband = np.transpose(
#                     [
#                         [1 / 64, 1 / 32],
#                         [1 / 32, 1 / 16],
#                         [1 / 16, 1 / 8],
#                         [1 / 8, 1 / 4],
#                     ]
#                 )
#             if inputArg1 == "[[1/64,1/32], [1/32,1/16]]":
#                 Chnl.Gabor_passband = np.transpose([[1 / 64, 1 / 32], [1 / 32, 1 / 16]])

#         if isinstance(inputArg2, str):
#             if inputArg2 == "[0, pi/3, 2*pi/3]":
#                 Chnl.Gabor_theta = [0, np.pi / 3, 2 * np.pi / 3]
#             if inputArg2 == "[0, pi/2]":
#                 Chnl.Gabor_theta = [0, np.pi / 2]
#             if inputArg2 == "0":
#                 Chnl.Gabor_theta = [0]

#         if isinstance(inputArg3, str):
#             if inputArg3 == "[0, pi/2]":
#                 Chnl.Gabor_beta = [0, np.pi / 2]
#             if inputArg3 == "0":
#                 Chnl.Gabor_beta = [0]

#         Chnl.Gabor_fc = np.mean(Chnl.Gabor_passband, axis=0)
#         Chnl.Gabor_wd = (
#             4
#             * np.log(2)
#             / (np.pi * (Chnl.Gabor_passband[1, :] - Chnl.Gabor_passband[0, :]))
#         )

#     return Chnl


# def ChannelMatrix_Generation(Chnl, roiSize_xy):
#     """Generate CHO channel matrix"""
#     x = np.linspace(1, roiSize_xy, roiSize_xy) - (roiSize_xy + 1) / 2
#     y = x
#     X, Y = np.meshgrid(x, y)
#     X = X.T.reshape(-1)
#     Y = Y.T.reshape(-1)

#     if Chnl.Chnl_Toggle == "Laguerre-Gauss":
#         LG_ORIEN, LG_ORDER = np.meshgrid(
#             np.arange(Chnl.LG_orien) + 1, np.arange(Chnl.LG_order) + 1
#         )
#         LG_ORIEN = LG_ORIEN.T.reshape(-1)
#         LG_ORDER = LG_ORDER.T.reshape(-1)
#         A = 5 * np.ones(LG_ORDER.size)
#         B = 14 * np.ones(LG_ORDER.size)
#         A[LG_ORIEN == 1] = 8
#         B[LG_ORIEN == 1] = 8
#         A[LG_ORIEN == 2] = 14
#         B[LG_ORIEN == 2] = 5
#         channelMatrix = np.zeros(
#             (roiSize_xy * roiSize_xy, Chnl.LG_order * Chnl.LG_orien)
#         )
#         for ii in range(LG_ORDER.size):
#             channelMatrix[:, ii] = Laguerre2D(LG_ORDER[ii], A[ii], B[ii], 0, 0, X, Y)

#     elif Chnl.Chnl_Toggle == "Gabor":
#         Gabor_THETA, Gabor_FC, Gabor_BETA = np.meshgrid(
#             Chnl.Gabor_theta, Chnl.Gabor_fc, Chnl.Gabor_beta
#         )
#         Gabor_wd_matrix = np.zeros((Chnl.Gabor_wd.size, 1, 1))
#         Gabor_wd_matrix[:, 0, 0] = Chnl.Gabor_wd
#         Gabor_WD = np.tile(Gabor_wd_matrix, (1, Gabor_FC.shape[1], Gabor_FC.shape[2]))

#         Gabor_FC = Gabor_FC.T.reshape(-1)
#         Gabor_WD = Gabor_WD.T.reshape(-1)
#         Gabor_THETA = Gabor_THETA.T.reshape(-1)
#         Gabor_BETA = Gabor_BETA.T.reshape(-1)

#         channelMatrix = np.zeros((roiSize_xy * roiSize_xy, Gabor_FC.size))
#         for ii in range(Gabor_FC.size):
#             channelMatrix[:, ii] = Gabor2D(
#                 Gabor_FC[ii], Gabor_WD[ii], Gabor_THETA[ii], Gabor_BETA[ii], 0, 0, X, Y
#             )
#     else:
#         raise ValueError("Unknown channel type")
#     return channelMatrix


# def CHO_patient_with_resampling(
#     sig_true, bkg_ordered, channelMatrix, internalNoise, Resampling_method
# ):
#     """CHO observer calculation with resampling"""
#     N_total_bkg = bkg_ordered.shape[1]

#     if Resampling_method == "Bootstrap":
#         rand_scanSelect_bkg = np.random.randint(N_total_bkg, size=N_total_bkg)
#     elif Resampling_method == "Shuffle":
#         rand_scanSelect_bkg = np.random.permutation(N_total_bkg)
#     else:
#         raise ValueError("Unknown resampling method")

#     sig = sig_true
#     bkg = bkg_ordered[:, rand_scanSelect_bkg].astype(float)

#     if sig.shape[0] != channelMatrix.shape[0] or bkg.shape[0] != channelMatrix.shape[0]:
#         print("Numbers of pixels do not match.")

#     vN = channelMatrix.T @ bkg
#     sbar = np.squeeze(sig)
#     S = np.cov(vN.T, rowvar=False)
#     print("S", S)
#     print("rank(S):", np.linalg.matrix_rank(S))
#     print("det(S):", np.linalg.det(S))
#     print("cond(S):", np.linalg.cond(S))
#     print("channelMatrix.T", channelMatrix.T)
#     print("sbar", sbar)
#     wCh = np.linalg.inv(S) @ channelMatrix.T @ sbar
#     temp = channelMatrix.T @ sbar
#     tsN_Mean = wCh.T @ temp
#     tN0 = wCh.T @ vN
#     tN = tN0 + np.random.randn(tN0.size) * internalNoise * np.std(tN0)
#     dp = np.sqrt((tsN_Mean**2) / (np.var(tN)))

#     return dp


# def interpolate_grid(X0, Y0, L0, XX, YY, method):
#     """Grid interpolation for lesion modeling"""
#     X0 = np.asarray(X0)
#     Y0 = np.asarray(Y0)
#     L0 = np.asarray(L0)
#     XX = np.asarray(XX)
#     YY = np.asarray(YY)

#     grid_points = (X0.flatten(), Y0.flatten())
#     values = L0.flatten()
#     Lesion = griddata(grid_points, values, (XX, YY), method=method)
#     return Lesion


# def max_consecutive_ones_2d(bool_array_2d):
#     """Find maximum consecutive ones in 2D array"""
#     max_consecutive = 0
#     for row in bool_array_2d:
#         max_count = 0
#         count = 0
#         for value in row:
#             if value:
#                 count += 1
#             else:
#                 max_count = max(max_count, count)
#                 count = 0
#         max_count = max(max_count, count)
#         max_consecutive = max(max_consecutive, max_count)
#     return max_consecutive


# def simulate_psf_damped_cosine(mtf50, mtf10=None, size=513, wire_fov_mm=50.0):
#     """
#     Simulate a 2D PSF based on MTF50 and optionally MTF10.

#     Parameters:
#     - mtf50: Spatial frequency (cycles/mm) where MTF=0.5.
#     - mtf10: Spatial frequency (cycles/mm) where MTF=0.1 (optional, None for Gaussian).
#     - size: PSF grid size (odd number, default 257).
#     - pixel_spacing: Pixel size in mm (default 0.6640625 mm).

#     Returns:
#     - psf: Normalized 2D PSF array.

#     Methods:
#     1. Gaussian PSF (mtf10=None):
#        - Uses a Gaussian function where MTF(f) = exp(-2 * pi^2 * sigma^2 * f^2).
#        - Sigma is derived from mtf50: sigma = sqrt(-ln(0.5) / (2 * pi^2 * mtf50^2)).
#        - Suitable for systems with smooth MTF fall-off (e.g., soft reconstruction kernels).

#     2. Damped Cosine PSF (mtf10 specified):
#        - Uses a damped cosine function: psf(r) = exp(-r/tau) * cos(k*r + phi).
#        - Optimizes tau, k, phi to match MTF at mtf50 (0.5) and mtf10 (0.1).
#        - Suitable for systems with oscillatory MTF (e.g., sharper kernels with edge enhancement).
#        - Optimization minimizes error in MTF values at specified frequencies.
#     """
#     # Ensure odd size for symmetric PSF
#     if size % 2 == 0:
#         size += 1

#     half = size // 2
#     dx = wire_fov_mm / size

#     # Create spatial grid in mm
#     x = np.linspace(-half, half, size) * dx
#     y = np.linspace(-half, half, size) * dx
#     X, Y = np.meshgrid(x, y)
#     R = np.sqrt(X**2 + Y**2)

#     # ----- radial MTF from a 2-D PSF (same helper used by the old code) -----
#     def compute_mtf_from_psf(psf):
#         psf = psf / psf.sum()
#         FT = np.fft.fft2(np.fft.ifftshift(psf))
#         mtf2d = np.fft.fftshift(np.abs(FT))
#         mtf2d /= np.max(mtf2d)
#         N = psf.shape[0]
#         centre = N // 2
#         fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
#         fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
#         Fx, Fy = np.meshgrid(fx, fy)
#         F = np.sqrt(Fx**2 + Fy**2)
#         f_rad = F[centre, centre:]
#         mtf_rad = mtf2d[centre, centre:]
#         return interp1d(
#             f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
#         )

#     # ----- Case 1: only mtf50 → Gaussian (k = 0) -----
#     if mtf10 is None:
#         # Analytic σ for a Gaussian that gives MTF(mtf50) = 0.5
#         sigma = np.sqrt(-np.log(0.5) / (2 * np.pi**2 * mtf50**2))
#         psf = np.exp(-(R**2) / (2 * sigma**2))
#         psf /= psf.sum()
#         return psf, dx

#     # ----- Case 2: optimise τ and k for both targets -----
#     def objective(params):
#         tau, k = params
#         if tau <= 0 or k < 0:
#             return 1e12
#         # Build PSF (clip negatives – the papers keep them, but for low-contrast
#         # lesion work a non-negative PSF is usually preferred)
#         psf_trial = np.exp(-R / tau) * np.cos(k * R)
#         psf_trial = np.maximum(psf_trial, 0.0)
#         mtf_func = compute_mtf_from_psf(psf_trial)
#         err50 = (mtf_func(mtf50) - 0.5) ** 2
#         err10 = (mtf_func(mtf10) - 0.1) ** 2
#         return err50 + err10

#     # Reasonable starting point (Gaussian width + a mild oscillation)
#     tau0 = 1.0 / (np.pi * mtf50)  # ~Gaussian width
#     k0 = 0.6 * np.pi * mtf50  # small ringing
#     res = minimize(
#         objective,
#         [tau0, k0],
#         bounds=[(tau0 * 0.3, tau0 * 3.0), (0.0, 2.0 * np.pi * mtf50)],
#         method="L-BFGS-B",
#     )

#     tau_opt, k_opt = res.x
#     psf = np.exp(-R / tau_opt) * np.cos(k_opt * R)
#     psf = np.maximum(psf, 0.0)  # keep non-negative
#     psf /= psf.sum()

#     # ----- verification (same style as the old function) -----
#     mtf_func = compute_mtf_from_psf(psf)
#     print(f"Damped-cosine PSF: tao = {tau_opt:.4f} mm, k = {k_opt:.4f} rad/mm")
#     print(f"  MTF({mtf50:.3f}) = {mtf_func(mtf50):.4f} (target 0.5)")
#     print(f"  MTF({mtf10:.3f}) = {mtf_func(mtf10):.4f} (target 0.1)")

#     return psf, dx


# def compute_presampling_mtf(psf, dx_psf):
#     """
#     Compute radial pre-sampling MTF from a simulated PSF.

#     Returns:
#     - freq: starts at 0.0 (cycles/mm)
#     - mtf:  starts at 1.0, decreases to 0
#     """
#     from scipy.interpolate import interp1d

#     # 1. Normalize PSF to sum = 1
#     psf_norm = psf / psf.sum()

#     # 2. FFT and shift zero-frequency to center
#     psf_shift = np.fft.ifftshift(psf_norm)
#     FT = np.fft.fft2(psf_shift)
#     mtf2d = np.fft.fftshift(np.abs(FT))  # Center at [N//2, N//2]

#     # 3. Normalize by MAX (which is DC component at center)
#     mtf2d /= np.max(mtf2d)  # Now max = 1.0

#     # 4. Build radial frequency grid
#     N = psf.shape[0]
#     centre = N // 2

#     # Frequency vectors (cycles/mm)
#     fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx_psf))  # e.g., [-Nyq, ..., 0, ..., +Nyq]
#     fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx_psf))
#     Fx, Fy = np.meshgrid(fx, fy)
#     F = np.sqrt(Fx**2 + Fy**2)

#     # 5. Extract radial line from center to edge
#     f_rad = F[centre, centre:]  # [0.0, df, 2df, ..., Nyq]
#     mtf_rad = mtf2d[centre, centre:]  # [1.0, ..., ~0]

#     # 6. Interpolant
#     mtf_func = interp1d(
#         f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0
#     )

#     return f_rad, mtf_rad, mtf_func


# def prepare_lesion_signal(
#     lesion_data,
#     psf_data,
#     target_contrast_hu,
#     target_size_mm,
#     roi_size_mm,
#     patient_fov_mm,
#     patient_matrix_px,
#     mtf50=None,
#     mtf10=None,
# ):
#     """Prepare lesion signal for CHO analysis"""
#     patient_pixel_size = patient_fov_mm / patient_matrix_px
#     roi_size_px = roi_size_mm / patient_pixel_size
#     roi_size_px = np.round(roi_size_px).astype(int)
#     if roi_size_px % 2 == 0:
#         roi_size_px += 1

#     # --- Extract lesion and mask volumes ---
#     lesion_volume = lesion_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
#     lesion_mask_volume = lesion_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

#     # --- Select middle slice ---
#     mid_slice_idx = round(lesion_volume.shape[-1] / 2)
#     lesion_slice = lesion_volume[:, :, mid_slice_idx]
#     mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

#     # --- Compute lesion HU and adjust contrast ---
#     lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)
#     mean_lesion_hu = np.mean(lesion_slice[mask_slice])
#     hu_difference = mean_lesion_hu - target_contrast_hu
#     lesion_slice = lesion_slice - hu_difference
#     lesion_slice[~mask_slice] = 0  # zero out background

#     # --- Scale lesion to desired physical size ---
#     target_width_pixels = target_size_mm / patient_pixel_size
#     scale_factor = target_width_pixels / lesion_width_pixels_input

#     input_height, input_width = lesion_slice.shape
#     output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

#     # --- Generate interpolation grids ---
#     x_in = np.linspace(0, input_width - 1, input_width)
#     y_in = np.linspace(0, input_height - 1, input_height)
#     X_in, Y_in = np.meshgrid(x_in, y_in)

#     x_out = np.linspace(0, input_width - 1, output_shape[1])
#     y_out = np.linspace(0, input_height - 1, output_shape[0])
#     X_out, Y_out = np.meshgrid(x_out, y_out)

#     # --- Rescale lesion using interpolation ---
#     scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

#     # --- Load and rescale PSF ---
#     if mtf50:
#         psf_rescaled, dx_psf = simulate_psf_damped_cosine(mtf50, mtf10)

#         # Compute presampling MTF
#         freq, mtf, mtf_func = compute_presampling_mtf(psf_rescaled, dx_psf)

#         # ----- resample simulated PSF to patient grid -----
#         psf_rescaled_shape = psf_rescaled.shape
#         scaling_factor = dx_psf / patient_pixel_size
#         os0_psf = np.floor(np.array(psf_rescaled_shape) * scaling_factor).astype(int)
#         # os0_psf = np.maximum(os0_psf, 15)  # safety
#         # os0_psf = os0_psf + (os0_psf % 2 == 0)  # odd size

#         # original grid (in *pixel* indices of the simulated PSF)
#         x0_psf = np.linspace(0, psf_rescaled_shape[1] - 1, psf_rescaled_shape[1])
#         y0_psf = np.linspace(0, psf_rescaled_shape[0] - 1, psf_rescaled_shape[0])
#         X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

#         # target grid
#         x_out_psf = np.linspace(0, psf_rescaled_shape[1] - 1, os0_psf[1])
#         y_out_psf = np.linspace(0, psf_rescaled_shape[0] - 1, os0_psf[0])
#         XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

#         psf_rescaled = interpolate_grid(
#             X0_psf, Y0_psf, psf_rescaled, XX_psf, YY_psf, "linear"
#         )
#     elif psf_data:
#         psf = psf_data["PSF"]
#         wire_fov_mm = 50
#         wire_matrix_size = 512
#         dx_psf = wire_fov_mm / wire_matrix_size  # measured pixel size

#         # Compute presampling MTF
#         freq, mtf, mtf_func = compute_presampling_mtf(psf, dx_psf)

#         scaling_factor = dx_psf / patient_pixel_size / 4

#         os0_psf = np.floor(np.array(psf.shape) * scaling_factor).astype(int)
#         # os0_psf = np.maximum(os0_psf, 15)
#         # os0_psf = os0_psf + (os0_psf % 2 == 0)

#         x0_psf = np.linspace(0, psf.shape[1] - 1, psf.shape[1])
#         y0_psf = np.linspace(0, psf.shape[0] - 1, psf.shape[0])
#         X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

#         x_out_psf = np.linspace(0, psf.shape[1] - 1, os0_psf[1])
#         y_out_psf = np.linspace(0, psf.shape[0] - 1, os0_psf[0])
#         XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

#         psf_rescaled = interpolate_grid(X0_psf, Y0_psf, psf, XX_psf, YY_psf, "linear")
#     else:
#         raise ValueError("Either PSF data or MTF50/MTF10 must be provided.")

#     psf_rescaled /= psf_rescaled.sum()  # normalize PSF

#     # --- Center lesion in padded array ---
#     lesion_padded = np.zeros((80, 80))
#     # lesion_padded = np.zeros((scaled_lesion.shape[0] + 50, scaled_lesion.shape[1] + 50))
#     start_row = (lesion_padded.shape[0] - scaled_lesion.shape[0]) // 2
#     start_col = (lesion_padded.shape[1] - scaled_lesion.shape[1]) // 2
#     lesion_padded[
#         start_row : start_row + scaled_lesion.shape[0],
#         start_col : start_col + scaled_lesion.shape[1],
#     ] = scaled_lesion

#     # --- Convolve lesion with PSF (simulate system blur) ---
#     lesion_convolved = convolve2d(lesion_padded, psf_rescaled, mode="full")

#     # --- Crop to ROI size ---
#     crop_start_row = (lesion_convolved.shape[0] - roi_size_px) // 2
#     crop_start_col = (lesion_convolved.shape[1] - roi_size_px) // 2
#     lesion_roi = lesion_convolved[
#         crop_start_row : crop_start_row + roi_size_px,
#         crop_start_col : crop_start_col + roi_size_px,
#     ]

#     return lesion_roi, freq, mtf, roi_size_px


# # def prepare_lesion_signal(
# #     lesion_data,
# #     psf_data,
# #     target_contrast_hu,
# #     target_size_mm,
# #     roi_size_mm,
# #     patient_fov_mm,
# #     patient_matrix_px,
# #     mtf50=None,
# #     mtf10=None,
# # ):
# #     """Prepare lesion signal for CHO analysis"""
# #     patient_pixel_size = patient_fov_mm / patient_matrix_px
# #     roi_size_px = roi_size_mm / patient_pixel_size
# #     roi_size_px = np.round(roi_size_px).astype(int)
# #     if roi_size_px % 2 == 0:
# #         roi_size_px += 1

# #     # --- Extract lesion and mask volumes ---
# #     lesion_volume = lesion_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
# #     lesion_mask_volume = lesion_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

# #     # --- Select middle slice ---
# #     mid_slice_idx = round(lesion_volume.shape[-1] / 2)
# #     lesion_slice = lesion_volume[:, :, mid_slice_idx]
# #     mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

# #     # --- Compute lesion HU and adjust contrast ---
# #     lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)
# #     mean_lesion_hu = np.mean(lesion_slice[mask_slice])
# #     hu_difference = mean_lesion_hu - target_contrast_hu
# #     lesion_slice = lesion_slice - hu_difference
# #     lesion_slice[~mask_slice] = 0  # zero out background

# #     # --- Scale lesion to desired physical size ---
# #     target_width_pixels = target_size_mm / patient_pixel_size
# #     scale_factor = target_width_pixels / lesion_width_pixels_input

# #     input_height, input_width = lesion_slice.shape
# #     output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

# #     # --- Generate interpolation grids ---
# #     x_in = np.linspace(0, input_width - 1, input_width)
# #     y_in = np.linspace(0, input_height - 1, input_height)
# #     X_in, Y_in = np.meshgrid(x_in, y_in)

# #     x_out = np.linspace(0, input_width - 1, output_shape[1])
# #     y_out = np.linspace(0, input_height - 1, output_shape[0])
# #     X_out, Y_out = np.meshgrid(x_out, y_out)

# #     # --- Rescale lesion using interpolation ---
# #     scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

# #     # --- Load and rescale PSF ---
# #     if mtf50:
# #         psf_rescaled, dx_psf = simulate_psf_damped_cosine(mtf50, mtf10)

# #         # Compute presampling MTF
# #         freq, mtf, mtf_func = compute_presampling_mtf(psf_rescaled, dx_psf)

# #         # ----- resample simulated PSF to patient grid -----
# #         psf_rescaled_shape = psf_rescaled.shape
# #         scaling_factor = dx_psf / patient_pixel_size
# #         os0_psf = np.floor(np.array(psf_rescaled_shape) * scaling_factor).astype(int)
# #         # os0_psf = np.maximum(os0_psf, 15)  # safety
# #         # os0_psf = os0_psf + (os0_psf % 2 == 0)  # odd size

# #         # original grid (in *pixel* indices of the simulated PSF)
# #         x0_psf = np.linspace(0, psf_rescaled_shape[1] - 1, psf_rescaled_shape[1])
# #         y0_psf = np.linspace(0, psf_rescaled_shape[0] - 1, psf_rescaled_shape[0])
# #         X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

# #         # target grid
# #         x_out_psf = np.linspace(0, psf_rescaled_shape[1] - 1, os0_psf[1])
# #         y_out_psf = np.linspace(0, psf_rescaled_shape[0] - 1, os0_psf[0])
# #         XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

# #         psf_rescaled = interpolate_grid(
# #             X0_psf, Y0_psf, psf_rescaled, XX_psf, YY_psf, "linear"
# #         )
# #     elif psf_data:
# #         psf = psf_data["PSF"]
# #         wire_fov_mm = 50
# #         wire_matrix_size = 512
# #         dx_psf = wire_fov_mm / wire_matrix_size  # measured pixel size

# #         # Compute presampling MTF
# #         freq, mtf, mtf_func = compute_presampling_mtf(psf, dx_psf)

# #         scaling_factor = dx_psf / patient_pixel_size / 4

# #         os0_psf = np.floor(np.array(psf.shape) * scaling_factor).astype(int)
# #         # os0_psf = np.maximum(os0_psf, 15)
# #         # os0_psf = os0_psf + (os0_psf % 2 == 0)

# #         x0_psf = np.linspace(0, psf.shape[1] - 1, psf.shape[1])
# #         y0_psf = np.linspace(0, psf.shape[0] - 1, psf.shape[0])
# #         X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

# #         x_out_psf = np.linspace(0, psf.shape[1] - 1, os0_psf[1])
# #         y_out_psf = np.linspace(0, psf.shape[0] - 1, os0_psf[0])
# #         XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

# #         psf_rescaled = interpolate_grid(X0_psf, Y0_psf, psf, XX_psf, YY_psf, "linear")
# #     else:
# #         raise ValueError("Either PSF data or MTF50/MTF10 must be provided.")

# #     psf_rescaled /= psf_rescaled.sum()  # normalize PSF

# #     # --- Center lesion in padded array ---
# #     lesion_padded = np.zeros((80, 80))
# #     start_row = (lesion_padded.shape[0] - scaled_lesion.shape[0]) // 2
# #     start_col = (lesion_padded.shape[1] - scaled_lesion.shape[1]) // 2
# #     lesion_padded[
# #         start_row : start_row + scaled_lesion.shape[0],
# #         start_col : start_col + scaled_lesion.shape[1],
# #     ] = scaled_lesion

# #     # --- Convolve lesion with PSF (simulate system blur) ---
# #     lesion_convolved = convolve2d(lesion_padded, psf_rescaled, mode="full")

# #     # --- Crop to ROI size ---
# #     crop_start_row = (lesion_convolved.shape[0] - roi_size_px) // 2
# #     crop_start_col = (lesion_convolved.shape[1] - roi_size_px) // 2
# #     lesion_roi = lesion_convolved[
# #         crop_start_row : crop_start_row + roi_size_px,
# #         crop_start_col : crop_start_col + roi_size_px,
# #     ]

# #     return lesion_roi, freq, mtf


# def _integral_image(self, image, window_size=[3, 3], padding="constant"):
#     """
#     AVERAGEFILTER 2-D mean filtering.
#     Performs mean filtering of a 2-dimensional matrix/image using the integral image method.

#     Parameters:
#     - image: 2D array-like, the input image to be filtered.
#     - window_size: tuple, (M, N) defines the vertical and horizontal window size.
#     - padding: str, can be 'constant', 'reflect', 'symmetric', or 'wrap'.

#     Returns:
#     - image: 2D array, the filtered image.
#     """

#     if len(image.shape) != 2:
#         raise ValueError("The input image must be a two-dimensional array.")

#     # Set up the window size
#     m, n = window_size

#     # Pad the image
#     pad_width = (
#         (m // 2, m // 2),
#         (n // 2, n // 2),
#     )  # Calculate padding around the image

#     if padding == "circular":
#         padded_image = np.pad(image, pad_width, mode="wrap")

#     elif padding == "replicate":
#         padded_image = np.pad(image, pad_width, mode="edge")

#     elif padding == "symmetric":
#         padded_image = np.pad(image, pad_width, mode="symmetric")

#     else:  # Default is 'constant' padding (zero padding)
#         padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

#     # Convert the image to float
#     imageD = padded_image.astype(np.float64)

#     # Calculate the integral image
#     t = np.cumsum(np.cumsum(imageD, axis=0), axis=1)

#     # Cast the resulting image back to the original type
#     return t


# def _calculate_roi_parameters(roi_size: int, padding_size: int) -> Tuple[int, int, int]:
#     """Calculate ROI-related parameters."""
#     roi_size_even = roi_size if roi_size % 2 == 0 else roi_size + 1
#     half_diff_size = (padding_size - roi_size_even) // 2
#     half_roi_size = roi_size // 2
#     return roi_size_even, half_diff_size, half_roi_size


# def _compute_integral_difference(
#     integral_img: np.ndarray, m: int, n: int
# ) -> np.ndarray:
#     """Compute difference using integral image property."""
#     return (
#         integral_img[m:, n:]
#         + integral_img[:-m, :-n]
#         - integral_img[m:, :-n]
#         - integral_img[:-m, n:]
#     )


# def _crop_arrays(arrays: list, half_diff_size: int, height: int, width: int) -> list:
#     """Crop multiple arrays using the same indices."""
#     crop_slice = slice(half_diff_size, height - half_diff_size), slice(
#         half_diff_size, width - half_diff_size
#     )
#     return [arr[crop_slice] for arr in arrays]


# def _apply_boundary_mask(std_map: np.ndarray, half_roi_size: int) -> None:
#     """Apply NaN mask to boundaries in-place."""
#     if half_roi_size > 0:
#         std_map[:half_roi_size, :] = np.nan
#         std_map[-half_roi_size:, :] = np.nan
#         std_map[:, :half_roi_size] = np.nan
#         std_map[:, -half_roi_size:] = np.nan


# def calculate_std_dev(
#     integral_images: np.ndarray,
#     integral_images_square: np.ndarray,
#     integral_edge: np.ndarray,
#     roi_size: int,
#     padding_size: int,
#     image_size: Tuple[int, int],
#     threshold_low: float,
#     threshold_high: float,
#     edges: np.ndarray,
#     corrected: bool = False,
# ) -> np.ndarray:
#     """
#     Calculate standard deviation maps using integral images.

#     Args:
#         integral_images: Integral image array for mean calculations
#         integral_images_square: Integral image array for variance calculations
#         integral_edge: Integral image array for edge detection
#         roi_size: Region of interest size
#         padding_size: Padding size for image borders
#         image_size: Output image dimensions (height, width)
#         threshold_low: Lower threshold for mean filtering
#         threshold_high: Upper threshold for mean filtering
#         edges: Binary edge mask
#         corrected: Whether to apply Bessel's correction

#     Returns:
#         Standard deviation maps with shape (height, width, channels)
#     """
#     # Calculate ROI parameters
#     roi_size_even = roi_size if roi_size % 2 == 0 else roi_size + 1
#     half_diff_size = (padding_size - roi_size_even) // 2
#     half_roi_size = roi_size // 2

#     # Get original dimensions and crop
#     orig_height, orig_width = integral_edge.shape[:2]
#     end_h, end_w = orig_height - half_diff_size, orig_width - half_diff_size

#     integral_edge = integral_edge[half_diff_size:end_h, half_diff_size:end_w, :]
#     integral_images = integral_images[half_diff_size:end_h, half_diff_size:end_w, :]
#     integral_images_square = integral_images_square[
#         half_diff_size:end_h, half_diff_size:end_w, :
#     ]

#     m = n = roi_size_even
#     roi_area = m * n
#     normalization_factor = roi_area / (roi_area - 1) if corrected else 1.0
#     edge_threshold = 1.0 / roi_size

#     # Initialize output
#     std_all = np.zeros(
#         [image_size[0], image_size[1], integral_images.shape[-1]], dtype=np.float32
#     )

#     # Process each channel (keeping original loop structure for performance)
#     for k in range(integral_images.shape[-1]):
#         # Mean calculation using integral images
#         t = integral_images[:, :, k]
#         mean_map = (t[m:, n:] + t[:-m, :-n] - t[m:, :-n] - t[:-m, n:]) / roi_area

#         # Variance calculation
#         t2 = integral_images_square[:, :, k]
#         mean_square = (t2[m:, n:] + t2[:-m, :-n] - t2[m:, :-n] - t2[:-m, n:]) / roi_area

#         # Edge impact calculation
#         t3 = integral_edge[:, :, k]
#         edge_impact = (t3[m:, n:] + t3[:-m, :-n] - t3[m:, :-n] - t3[:-m, n:]) / roi_area

#         # Create masks
#         threshold_mask = (mean_map < threshold_low) | (mean_map > threshold_high)
#         edge_mask = edge_impact >= edge_threshold
#         combined_mask = threshold_mask | edge_mask | edges

#         # Calculate standard deviation with numerical stability
#         variance = mean_square - mean_map**2
#         variance = np.maximum(
#             variance, 0
#         )  # Prevent negative values from numerical errors
#         std_map = np.sqrt(normalization_factor * variance)

#         # Apply masks
#         std_map[combined_mask] = np.nan

#         # Apply boundary mask
#         if half_roi_size > 0:
#             std_map[:half_roi_size, :] = np.nan
#             std_map[-half_roi_size:, :] = np.nan
#             std_map[:, :half_roi_size] = np.nan
#             std_map[:, -half_roi_size:] = np.nan

#         std_all[:, :, k] = std_map

#     return std_all


# def extract_ROIs(STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size):
#     """Extract ROIs with low standard deviation for NPS and CHO analysis."""
#     ROIs = []
#     for ii in range(STD_map_all.shape[-1]):
#         im10 = Images_Section[:, :, ii]
#         SD_Map = STD_map_all[:, :, ii]
#         valid_mask = (SD_Map < Thre_SD) & (~np.isnan(SD_Map))
#         rows, cols = np.where(valid_mask)
#         if rows.size == 0:
#             continue
#         std_vals = SD_Map[rows, cols]
#         sorted_indices = np.argsort(std_vals)
#         sorted_rows = rows[sorted_indices]
#         sorted_cols = cols[sorted_indices]
#         selection_mask = np.zeros_like(SD_Map, dtype=bool)
#         selected_centers = []
#         for r, c in zip(sorted_rows, sorted_cols):
#             if selection_mask[r, c]:
#                 continue
#             selected_centers.append((r, c))
#             r_start = max(r - Half_ROI_size, 0)
#             r_end = min(r + Half_ROI_size + 1, selection_mask.shape[0])
#             c_start = max(c - Half_ROI_size, 0)
#             c_end = min(c + Half_ROI_size + 1, selection_mask.shape[1])
#             selection_mask[r_start:r_end, c_start:c_end] = True
#         for row, col in selected_centers:
#             if (
#                 row - Half_ROI_size < 0
#                 or row + Half_ROI_size + 1 > Im_Size[0]
#                 or col - Half_ROI_size < 0
#                 or col + Half_ROI_size + 1 > Im_Size[1]
#             ):
#                 continue
#             roi = im10[
#                 row - Half_ROI_size : row + Half_ROI_size + 1,
#                 col - Half_ROI_size : col + Half_ROI_size + 1,
#             ]
#             ROIs.append(roi)
#     ROIs_array = np.array(ROIs)
#     Total_NPS_No = 0
#     if ROIs_array.size:
#         ROIs_array = np.transpose(ROIs_array, (1, 2, 0))
#         Total_NPS_No = ROIs_array.shape[-1]
#     return ROIs_array, Total_NPS_No


# def compute_image_analysis(curr_file, pixel_roi_mm2):
#     img = curr_file.pixel_array
#     img_rescaled = img * curr_file.RescaleSlope + curr_file.RescaleIntercept
#     mask = img_rescaled >= -260
#     binary_mask = ndimage.binary_fill_holes(mask)
#     if binary_mask is not None:
#         binary_mask = binary_mask.astype(bool)
#     else:
#         binary_mask = np.zeros_like(mask, dtype=bool)

#     labels, num = measure.label(binary_mask, return_num=True)  # type: ignore
#     if num > 0:
#         largest_region = np.argmax(np.bincount(labels.flat)[1:]) + 1
#         binary_img = labels == largest_region
#     else:
#         binary_img = np.zeros_like(binary_mask, dtype=bool)

#     pixel_count = np.sum(binary_img)
#     roi_mm2 = pixel_count * pixel_roi_mm2
#     img_rescaled_masked = img_rescaled[binary_img]
#     hu_mean = np.mean(img_rescaled_masked)

#     coronal_slice = img_rescaled[img.shape[1] // 2, :]
#     ctdi = curr_file.get("CTDIvol", np.nan)
#     dw = 2 * np.sqrt((hu_mean / 1000 + 1) * roi_mm2 / np.pi)
#     return ctdi, dw, coronal_slice


# def zoom_centered(arr, zoom_factor):
#     h, w = arr.shape
#     zh, zw = int(np.round(h * zoom_factor)), int(np.round(w * zoom_factor))
#     zoomed = zoom(arr, zoom_factor, order=1)

#     # Crop or pad to original size
#     if zoom_factor > 1:
#         startx = (zh - h) // 2
#         starty = (zw - w) // 2
#         result = zoomed[startx : startx + h, starty : starty + w]
#     else:
#         pad_h = (h - zh) // 2
#         pad_w = (w - zw) // 2
#         result = np.zeros_like(arr)
#         result[pad_h : pad_h + zh, pad_w : pad_w + zw] = zoomed
#     return result


# # def create_lesion_model(set, index, mtf50, mtf10, recon_diameter_mm, rows):
# #     """Create a lesion model for analysis"""

# #     lesion_contrasts = [-30, -30, -10, -30, -50]
# #     lesion_sizes = [3, 9, 6, 6, 6]
# #     roi_sizes = [21, 29, 25, 25, 25]

# #     if set == "low-contrast":
# #         lesion_contrasts = [int(c / 2) for c in lesion_contrasts]
# #     elif set == "high-contrast":
# #         lesion_contrasts = [int(c * 2) for c in lesion_contrasts]

# #     lesion_file_data = sio.loadmat(LESION_FILE)
# #     psf_data = sio.loadmat(PSF_FILE)

# #     lesion_contrast = lesion_contrasts[index]
# #     lesion_size = lesion_sizes[index]
# #     roi_size = roi_sizes[index]

# #     patient_pixel_size = recon_diameter_mm / rows
# #     roi_size_px = roi_size / patient_pixel_size
# #     roi_size_px = np.round(roi_size_px).astype(int)
# #     if roi_size_px % 2 == 0:
# #         roi_size_px += 1

# #     # --- Extract lesion and mask volumes ---
# #     lesion_volume = lesion_file_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
# #     lesion_mask_volume = lesion_file_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

# #     # --- Select middle slice ---
# #     mid_slice_idx = round(lesion_volume.shape[-1] / 2)
# #     lesion_slice = lesion_volume[:, :, mid_slice_idx]
# #     mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

# #     # --- Compute lesion HU and adjust contrast ---
# #     lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)
# #     mean_lesion_hu = np.mean(lesion_slice[mask_slice])
# #     hu_difference = mean_lesion_hu - lesion_contrast
# #     lesion_slice = lesion_slice - hu_difference
# #     lesion_slice[~mask_slice] = 0  # zero out background

# #     # --- Scale lesion to desired physical size ---
# #     target_width_pixels = lesion_size / patient_pixel_size
# #     scale_factor = target_width_pixels / lesion_width_pixels_input

# #     input_height, input_width = lesion_slice.shape
# #     output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

# #     # --- Generate interpolation grids ---
# #     x_in = np.linspace(0, input_width - 1, input_width)
# #     y_in = np.linspace(0, input_height - 1, input_height)
# #     X_in, Y_in = np.meshgrid(x_in, y_in)

# #     x_out = np.linspace(0, input_width - 1, output_shape[1])
# #     y_out = np.linspace(0, input_height - 1, output_shape[0])
# #     X_out, Y_out = np.meshgrid(x_out, y_out)

# #     # --- Rescale lesion using interpolation ---
# #     scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

# #     lesion_roi = zoom_centered(scaled_lesion, 1.5)
# #     # lesion_roi, _, _ = prepare_lesion_signal(
# #     #     lesion_file_data,
# #     #     psf_data,
# #     #     lesion_contrast,
# #     #     lesion_size,
# #     #     roi_size,
# #     #     recon_diameter_mm,
# #     #     rows,
# #     #     mtf50,
# #     #     mtf10,
# #     # )

# #     return lesion_roi


# def center_crop_or_pad(img, out_size):
#     out_h, out_w = out_size, out_size
#     in_h, in_w = img.shape

#     out = np.zeros((out_h, out_w), dtype=img.dtype)

#     # source (input) crop box
#     src_y0 = max(0, (in_h - out_h) // 2)
#     src_x0 = max(0, (in_w - out_w) // 2)
#     src_y1 = min(in_h, src_y0 + out_h)
#     src_x1 = min(in_w, src_x0 + out_w)

#     # destination (output) paste box
#     dst_y0 = max(0, (out_h - in_h) // 2)
#     dst_x0 = max(0, (out_w - in_w) // 2)
#     dst_y1 = dst_y0 + (src_y1 - src_y0)
#     dst_x1 = dst_x0 + (src_x1 - src_x0)

#     out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
#     return out


# def create_lesion_model(set, index, mtf50, mtf10, recon_diameter_mm, rows):
#     """Create a lesion model for analysis"""

#     lesion_contrasts = [-30, -30, -10, -30, -50]
#     lesion_sizes = [3, 9, 6, 6, 6]
#     roi_sizes = [21, 29, 25, 25, 25]

#     if set == "low-contrast":
#         lesion_contrasts = [int(c / 2) for c in lesion_contrasts]
#     elif set == "high-contrast":
#         lesion_contrasts = [int(c * 2) for c in lesion_contrasts]

#     lesion_file_data = sio.loadmat(LESION_FILE)

#     lesion_contrast = lesion_contrasts[index]
#     lesion_size = lesion_sizes[index]
#     roi_size = roi_sizes[index]

#     patient_pixel_size = recon_diameter_mm / rows
#     roi_size_px = int(np.round(roi_size / patient_pixel_size))
#     if roi_size_px % 2 == 0:
#         roi_size_px += 1

#     lesion_volume = lesion_file_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
#     lesion_mask_volume = lesion_file_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

#     mid_slice_idx = round(lesion_volume.shape[-1] / 2)
#     lesion_slice = lesion_volume[:, :, mid_slice_idx]
#     mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

#     lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)

#     mean_lesion_hu = np.mean(lesion_slice[mask_slice])
#     hu_difference = mean_lesion_hu - lesion_contrast
#     lesion_slice = lesion_slice - hu_difference
#     lesion_slice[~mask_slice] = 0

#     # --- Scale lesion to desired physical size (this is GOOD) ---
#     target_width_pixels = lesion_size / patient_pixel_size
#     scale_factor = target_width_pixels / lesion_width_pixels_input

#     input_height, input_width = lesion_slice.shape
#     output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

#     x_in = np.linspace(0, input_width - 1, input_width)
#     y_in = np.linspace(0, input_height - 1, input_height)
#     X_in, Y_in = np.meshgrid(x_in, y_in)

#     x_out = np.linspace(0, input_width - 1, output_shape[1])
#     y_out = np.linspace(0, input_height - 1, output_shape[0])
#     X_out, Y_out = np.meshgrid(x_out, y_out)

#     scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

#     # ✅ NO zoom here (this was making sizes look too similar)
#     # ✅ force a consistent ROI frame without changing lesion scale
#     lesion_roi = center_crop_or_pad(scaled_lesion, roi_size_px)

#     return lesion_roi


# def run_full_cho_analysis(files, config):
#     """Run Full CHO Analysis with lesion detectability calculation"""
#     print("Config")
#     for key, val in config.items():
#         if key == "custom_parameters":
#             for param_key, param_val in val.items():
#                 print(f"  {param_key}: {param_val}")
#         else:
#             print(f"  {key}: {val}")

#     custom_params = config["custom_parameters"]
#     start = time.time()

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             15,
#             "Starting Full CHO calculation...",
#             "preprocessing",
#         )

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 10, "Loading lesion models...", "loading"
#         )

#     # Lesion parameters
#     dcm_parser = DicomParser(files)
#     patient, study, scanner, series, ct = dcm_parser.extract_core()

#     lesion_file_data = sio.loadmat(LESION_FILE)
#     psf_data = sio.loadmat(PSF_FILE)

#     lesion_config = get_lesion_configuration(custom_params["lesionSet"])
#     lesion_contrasts_hu = lesion_config["contrasts"]
#     lesion_sizes_mm = lesion_config["sizes"]
#     roi_sizes_mm = lesion_config["roi_sizes"]
#     pixel_size_x_y = series["pixel_spacing_mm"]
#     patient_fov_mm = float(ct["recon_diameter_mm"])
#     pixel_size_x = float(pixel_size_x_y[0])
#     # This assumes the pixel aspect ratio is 1:1
#     patient_matrix_px = int(series["rows"])
#     # roi_sizes_px = round_to_nearest_odd(np.array(roi_sizes_mm) / pixel_size_x)
#     spatial_resolution = custom_params.get("spatial_resolution", "auto")
#     mtf50 = None
#     mtf10 = None
#     if spatial_resolution == "custom":
#         mtf50 = custom_params.get("mtf50", None)
#         mtf10 = custom_params.get("mtf10", None)

#     # mtf50 = 0.434  # cycles/cm
#     # mtf10 = 0.730  # cycles/cm
#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             15,
#             "Preparing lesion models for analysis...",
#             "preprocessing",
#         )
#     lesion_data = []
#     for target_contrast_hu, target_size_mm, roi_size_mm in zip(
#         lesion_contrasts_hu, lesion_sizes_mm, roi_sizes_mm
#     ):
#         signal, freq, mtf, roi_size_px = prepare_lesion_signal(
#             lesion_file_data,
#             psf_data,
#             target_contrast_hu,
#             target_size_mm,
#             roi_size_mm,
#             patient_fov_mm,
#             patient_matrix_px,
#             mtf50,
#             mtf10,
#         )
#         lesion_data.append(
#             {
#                 "lesion_contrast": target_contrast_hu,
#                 "lesion_size": target_size_mm,
#                 "roi_size_mm": roi_size_mm,
#                 "roi_size_px": roi_size_px,
#                 "signal": signal,
#                 "frequency": freq,
#                 "mtf": mtf,
#             }
#         )
#     # import matplotlib.pyplot as plt

#     # plt.figure(figsize=(7, 4.5))
#     # for lesion in lesion_data:
#     #     plt.subplot(len(lesion_data), 1, lesion_data.index(lesion) + 1)
#     #     plt.plot(
#     #         lesion["frequency"],
#     #         lesion["mtf"],
#     #         "b-",
#     #         linewidth=2,
#     #         label="Presampling MTF",
#     #     )
#     #     plt.xlim(0, 1)
#     # plt.show()

#     # lesion_data = [
#     #     {
#     #         "lesion_contrast": target_contrast_hu,
#     #         "lesion_size": target_size_mm,
#     #         "roi_size": roi_size_mm,
#     #         "signal": prepare_lesion_signal(
#     #             lesion_data,
#     #             psf_data,
#     #             target_contrast_hu,
#     #             target_size_mm,
#     #             roi_size_mm,
#     #             patient_fov_mm,
#     #             patient_matrix_px,
#     #             mtf50,
#     #             mtf10,
#     #         ),
#     #     }
#     #     for target_contrast_hu, target_size_mm, roi_size_mm in zip(
#     #         lesion_contrasts_hu, lesion_sizes_mm, roi_sizes_mm
#     #     )
#     # ]

#     # Process DICOM files (similar to global noise but with additional CHO analysis)
#     n_images = series["number_of_frames"]

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             20,
#             f"Processing {n_images} DICOM images...",
#             "analysis",
#         )

#     # Get basic parameters
#     rows = int(series.get("rows", 512))
#     cols = int(series.get("columns", 512))
#     dx_mm, dy_mm = series["pixel_spacing_mm"]
#     img_size = (rows, cols)
#     dx_cm = dx_mm / 10
#     dy_cm = dy_mm / 10

#     # Calculate dose metrics (same as global noise)
#     ctdi_all = np.zeros(n_images)
#     dw = np.zeros(n_images)
#     coronal_view = np.zeros((cols, n_images))
#     pixel_roi_mm2 = dx_mm * dy_mm

#     # CHO Analysis setup
#     roi_diameter = round(custom_params["roiSize"] / dx_mm)
#     roi_radius_round = round(np.floor(roi_diameter / 2))
#     Thr1 = custom_params["thresholdLow"]
#     Thr2 = custom_params["thresholdHigh"]

#     # CHO parameters
#     n_resample = custom_params["resamples"]
#     internal_noise = custom_params["internalNoise"]
#     resampling_method = custom_params["resamplingMethod"]

#     for i in range(n_images):
#         if (
#             config["report_progress"]
#             and config["series_uuid"]
#             and (i % 10 == 0 or i == n_images - 1)
#         ):
#             # 25 = progress during analysis
#             # 40 = progress after analysis
#             progress_percentage = 25 + ((i / n_images) * (40 - 25))
#             progress_tracker.update_progress(
#                 config["series_uuid"],
#                 int(progress_percentage),
#                 f"Calculating dose metrics for slice {i+1}/{n_images}",
#                 "analysis",
#             )

#         curr_file = files[i]
#         ctdi_all[i], dw[i], coronal_view[:, i] = compute_image_analysis(
#             curr_file, pixel_roi_mm2
#         )
#     # Calculate basic dose/quality metrics
#     para_a = 3.704369
#     para_b = 0.03671937
#     exp_decay_inc = para_a * np.exp(-para_b * (dw / 10))
#     ssde_inc = exp_decay_inc * ctdi_all

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 40, "Dose metrics calculation completed", "analysis"
#         )

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 45, "Setting up CHO channels...", "analysis"
#         )

#     # CHO channels setup
#     class Chnl:
#         Chnl_Toggle = "Gabor"

#     Gabor_passband = "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]"
#     Gabor_theta = "[0, pi/3, 2*pi/3]"
#     Gabor_beta = "0"
#     Chnl = channel_selection(Chnl, Gabor_passband, Gabor_theta, Gabor_beta)

#     # Process image sections for NPS and CHO analysis
#     # dx = (patient_fov_mm / 10) / rows
#     # slice_interval_cm = series["slice_interval_mm"] / 10
#     # N_sub = round(50 / series["slice_interval_mm"])
#     # noise_level_sum = 0
#     # NPS_Cal = True
#     # padding_size = 2 * round(1.2 / dxif padding_size % 2 == 0 else int(padding_size + 1)

#     # Perform CHO analysis in sections
#     padding_size = 2 * round(1.2 / dx_cm)
#     padding_size = int(padding_size) if padding_size % 2 == 0 else int(padding_size + 1)
#     # padding_size = 2 ** math.ceil(math.log2(2 * round(1.2 / dx_cm)))
#     # padding_size = int(padding_size + padding_size % 2)
#     window_length = custom_params["windowLength"]
#     step_size = custom_params["stepSize"]
#     sigma = 5.0
#     window_gen = CTSlidingWindow(
#         window_length=window_length,
#         step_size=step_size,
#         padding_size=int(padding_size),
#         sigma=sigma,
#         use_cache=False,
#     )
#     total_sections = window_gen.get_num_windows(files)
#     # total_sections = n_images // N_sub - 2
#     dps_all = np.zeros([total_sections, len(lesion_data)])
#     noise_level_local = np.zeros([total_sections, 1])
#     location_sparse = np.zeros(total_sections)
#     total_nps_num = 0
#     nps_1d_sum = 0
#     noise_level_sum = 0
#     spatial_frequency = []
#     nps_cal = True

#     global_noise_level = 0  # Will be updated in the loop

#     # Process sections for CHO analysis
#     for window, metadata in window_gen.generate_windows(files):
#         # for mm in range(1, total_sections + 1):
#         # N1 = (mm - 1) * N_sub
#         # N2 = mm * N_sub
#         mm = metadata["window_idx"]
#         location_sparse[mm] = metadata["relative_mid_cm"]

#         if config["report_progress"] and config["series_uuid"]:
#             # 50 = progress during analysis
#             # 90 = progress after analysis
#             progress_percentage = 50 + ((mm / total_sections) * (90 - 50))
#             progress_tracker.update_progress(
#                 config["series_uuid"],
#                 int(progress_percentage),
#                 f"Processing CHO section {mm + 1}/{total_sections}",
#                 "analysis",
#             )

#         pixel_array = window["pixel_array"]
#         pixel_array_edges = window["edges"]
#         integral_edges = window["integral_edges"]
#         integral_images = window["integral_images"]
#         integral_images_squared = window["integral_images_square"]

#         STD_all = calculate_std_dev(
#             integral_images,
#             integral_images_squared,
#             integral_edges,
#             roi_diameter,
#             padding_size,
#             img_size,
#             Thr1,
#             Thr2,
#             pixel_array_edges,
#             corrected=False,
#         )

#         max_value = np.nanmax(STD_all)
#         h_Values, edges = np.histogram(
#             STD_all.flatten(), bins=np.arange(0, max_value, 0.2)
#         )
#         maxcount = np.max(h_Values)
#         whichbin_SD = np.argmax(h_Values)
#         bin_edge = edges[whichbin_SD]
#         noise_level_local[mm] = bin_edge
#         global_noise_level = bin_edge  # Update global noise level

#         # Calculate NPS if first time
#         if nps_cal and STD_all is not None:
#             roi_radius_round = round(np.floor(roi_diameter / 2))
#             ROI_All_NPS, Total_NPS_No = extract_ROIs(
#                 STD_all, bin_edge, roi_radius_round, pixel_array, img_size
#             )
#             if ROI_All_NPS.size > 0:
#                 max_rois = 200
#                 spatial_frequency, nps_1d_sum, noise_level_sum, unit = ROI_to_NPS_Sum(
#                     roi_diameter, ROI_All_NPS[:, :, 0:max_rois], dx_cm, dy_cm
#                 )
#                 total_nps_num = max_rois
#                 nps_cal = False

#         # Clean up temporary arrays
#         del STD_all
#         gc.collect()

#         # Calculate threshold for ROI selection
#         Left_Area = np.sum(h_Values[:whichbin_SD])
#         if whichbin_SD > 0:
#             Two_sigma_area = np.zeros(whichbin_SD)
#             Two_sigma_area[0] = Left_Area
#             temp_area = Left_Area

#             for b_idx in range(1, whichbin_SD):
#                 Two_sigma_area[b_idx] = temp_area - h_Values[b_idx - 1]
#                 temp_area = temp_area - h_Values[b_idx - 1]

#             Ratios = Two_sigma_area / max(Left_Area, 1)
#             target_ratio = 0.9544
#             temp_dis = np.abs(target_ratio - Ratios)
#             closest = Ratios[np.argmin(temp_dis)]
#             loc = np.where(Ratios == closest)[0][0]
#             gap = whichbin_SD - loc
#             Thre_SD = edges[min(whichbin_SD + gap, len(edges) - 1)]
#         else:
#             Thre_SD = bin_edge

#         # Perform CHO analysis for each lesion type
#         all_dps = []
#         cache = {}

#         for ii, lesion in enumerate(lesion_data):
#             lesion_sig = lesion["signal"]
#             roi_size = lesion["roi_size_px"]
#             Half_ROI_size = round(np.floor(roi_size / 2))
#             if roi_size in cache:
#                 # Skip to avoid redundant calls
#                 print(f"Using cached ROI_All for lesion {ii}")
#                 ROI_All, Total_NPS_No = cache[roi_size]
#             else:
#                 STD_map_all = calculate_std_dev(
#                     integral_images,
#                     integral_images_squared,
#                     integral_edges,
#                     roi_size,
#                     padding_size,
#                     img_size,
#                     Thr1,
#                     Thr2,
#                     pixel_array_edges,
#                     corrected=False,
#                 )
#                 ROI_All, Total_NPS_No = extract_ROIs(
#                     STD_map_all, Thre_SD, Half_ROI_size, pixel_array, img_size
#                 )
#                 cache[roi_size] = (ROI_All, Total_NPS_No)

#             Bkg_HU = 0  # make the background ROI to be 0 mean  #40 + abs(Lesion_Contrasts[ii])  #New_bkg_HU; #90 for -50HU lesion; 70 for -30HU lesion; 50 for -10HU lesion;
#             Noise_ROI = ROI_All[:, :, 0:Total_NPS_No]

#             depth = Noise_ROI.shape[2]
#             for i in range(depth):
#                 temp = Noise_ROI[:, :, i]
#                 Noise_ROI[:, :, i] = temp - np.mean(
#                     temp
#                 )  # Adjusting the noise ROI  to be 0 mean

#             N_total = Noise_ROI.shape[
#                 2
#             ]  # Getting the size of the third dimension (similar to MATLAB's size function)
#             dp = np.zeros(
#                 (n_resample, 1)
#             )  # Creating a zero array of shape (numResample, 1)
#             sample_idx = np.random.permutation(
#                 N_total
#             )  # Randomly permute the indices (equivalent to randperm in MATLAB)
#             channelMatrix = ChannelMatrix_Generation(Chnl, roi_size)
#             bkg_ordered = np.reshape(
#                 Noise_ROI[:, :, sample_idx], (roi_size**2, len(sample_idx))
#             )  # Reshape the Noise_ROI array
#             sig_true = np.reshape(
#                 lesion_sig, (roi_size**2, 1)
#             )  # Reshaping Lesion_sig to a 2D array
#             dp = CHO_patient_with_resampling(
#                 sig_true, bkg_ordered, channelMatrix, internal_noise, resampling_method
#             )

#             all_dps.append(dp)

#         del cache
#         dps_all[mm, :] = all_dps

#         # Clean up
#         gc.collect()
#         future_end = metadata["actual_end_cm"] + metadata["step_size"]
#         future_indices = window_gen._find_slices_in_range(
#             metadata["slice_positions_cm"], metadata["actual_start_cm"], future_end
#         )
#         window_gen.clear_cache(keep_indices=future_indices)
#     window_gen.clear_cache()
#     # Calculate final statistics
#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             90,
#             "Calculating statistics and preparing final results...",
#             "finalizing",
#         )

#     Mean_loc_dps = np.mean(dps_all, axis=-1) if dps_all.size > 0 else np.array([0])
#     Mean_All_dps = np.mean(Mean_loc_dps) if Mean_loc_dps.size > 0 else 0
#     Ave_noise_level = np.mean(noise_level_local)

#     # Calculate NPS statistics
#     ave_noise_level_nps = 0
#     fav = 0
#     peakfrequency = 0
#     min10percent_frequency = 0

#     if total_nps_num > 0 and nps_1d_sum is not None:
#         NPS_1D = nps_1d_sum / total_nps_num
#         ave_noise_level_nps = noise_level_sum / total_nps_num
#         fav, peakfrequency, k, min10percent_frequency = NPS_statistics(NPS_1D, unit)
#     else:
#         NPS_1D = np.array([])
#         spatial_frequency = np.array([])

#     # Calculate basic dose/quality metrics
#     para_a = 3.704369
#     para_b = 0.03671937

#     ctdi_mean = np.mean(ctdi_all)
#     dw_mean_cm = np.mean(dw) / 10
#     exp_decay = para_a * np.exp(-para_b * dw_mean_cm)
#     mean_ssde = exp_decay * ctdi_mean
#     exp_decay_inc = para_a * np.exp(-para_b * (dw / 10))
#     ssde_inc = exp_decay_inc * ctdi_all
#     scan_length_cm = series["series_length_mm"] / 10
#     dlp_ctdi_vol = scan_length_cm * ctdi_mean
#     dlp_ssde = scan_length_cm * mean_ssde
#     mtf_10_percent = 7.30

#     processing_time = time.time() - start

#     # Prepare results
#     series["uuid"] = config["series_uuid"]
#     series["image_count"] = n_images
#     series["scan_length_cm"] = scan_length_cm

#     # location_sparse = np.arange(step_size, scan_length_cm, step_size)
#     # chop = len(location_sparse) - total_sections
#     # start_chop = chop // 2
#     # end_chop = start_chop + chop % 2
#     # location_sparse = location_sparse[start_chop:-end_chop]
#     location = np.linspace(0, scan_length_cm, num=len(ctdi_all))

#     results = {
#         "average_frequency": fav,
#         "average_index_of_detectability": Mean_All_dps,
#         "average_noise_level": float(Ave_noise_level),
#         "cho_detectability": Mean_loc_dps.tolist(),
#         "ctdivol": ctdi_all.tolist(),
#         "ctdivol_avg": ctdi_mean,
#         "dlp": dlp_ctdi_vol,
#         "dlp_ssde": dlp_ssde,
#         "dw": dw.tolist(),
#         "dw_avg": dw_mean_cm,
#         "location": location.tolist(),
#         "location_sparse": location_sparse.tolist(),
#         "noise_level": noise_level_local.flatten().tolist(),
#         "nps": NPS_1D.flatten().tolist(),
#         "peak_frequency": peakfrequency,
#         "percent_10_frequency": min10percent_frequency,
#         "processing_time": processing_time,
#         "spatial_frequency": spatial_frequency.tolist(),
#         "spatial_resolution": mtf_10_percent,
#         "ssde": mean_ssde,
#         "ssde_inc": ssde_inc.tolist(),
#         "coronal_view": coronal_view,
#         "global_noise_level": global_noise_level,
#     }

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"], 95, "Saving results to database...", "finalizing"
#         )

#     # Convert results and save to database
#     converted_results = convert_numpy_to_python(results)
#     if custom_params["saveResults"]:
#         try:
#             from results_storage import cho_storage

#             success = cho_storage.save_results(
#                 patient, study, scanner, series, ct, converted_results
#             )

#             if success:
#                 print(
#                     f"Full analysis results saved to database for series {series['series_instance_uid']}"
#                 )
#             else:
#                 print(
#                     f"Failed to save full analysis results to database for series {series['series_instance_uid']}"
#                 )

#         except Exception as e:
#             print(f"Error saving to database: {str(e)}")

#     if config["report_progress"] and config["series_uuid"]:
#         progress_tracker.update_progress(
#             config["series_uuid"],
#             100,
#             "Full CHO calculation completed successfully",
#             "completed",
#         )
#         progress_tracker.complete_calculation(
#             config["series_uuid"],
#             {
#                 "patient": patient,
#                 "study": study,
#                 "scanner": scanner,
#                 "series": series,
#                 "ct": ct,
#                 "results": converted_results,
#             },
#         )

#     if custom_params["deleteAfterCompletion"]:
#         import orthanc

#         orthanc.RestApiDelete(f'/series/{config["series_uuid"]}')
#         print(f"Requested deletion of series {config['series_uuid']}")

#     print(f"Full CHO processing time: {processing_time:.2f} seconds")
#     print(f"Peak frequency = {peakfrequency:.3f}")
#     print(f"Average frequency = {fav:.3f}")
#     print(f"Min 10% frequency = {min10percent_frequency:.3f}")
#     print(f"Average noise level = {Ave_noise_level:.3f}")
#     print(f"Mean detectability = {Mean_All_dps:.3f}")
#     # del results['coronal_view']
#     return {
#         "patient": patient,
#         "study": study,
#         "scanner": scanner,
#         "series": series,
#         "ct": ct,
#         "results": converted_results,
#     }


# def main(files, config):
#     """
#     Main function for unified CHO analysis

#     Parameters:
#     - files: List of DICOM files
#     - config: Configuration dictionary with keys:
#         - series_uuid: Series UUID for progress reporting
#         - full_test: Boolean - True for full CHO analysis, False for global noise only
#         - report_progress: Boolean - Whether to report progress updates

#     Returns:
#     - results_dict: Dictionary with calculation results
#     """
#     print(
#         f"Starting CHO analysis - Full test: {config['full_test']}, Progress tracking: {config['report_progress']}"
#     )

#     if config["full_test"]:
#         return run_full_cho_analysis(files, config)
#     else:
#         return run_global_noise_analysis(files, config)


# if __name__ == "__main__":
#     import pydicom

#     # This section won't be used when imported as a module
#     # test_path = "V:\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\L067_FD_1_0_B30F_0001"  # Replace with actual path
#     test_path = "W:\\Liver Segmentation\\input\\TOBIN, SARA\\2025-06-26-001\\IMAGES"  # Replace with actual path
#     # test_path = "Y:\\Patient_data\\Alpha\\Radiomics\\healthy\\BARTZ_LYDIA_L\\DICOMS\\100%\\3\\Br44f\\1x1\\vmi_67"
#     slices = []
#     skipcount = 0
#     for f in os.listdir(test_path):
#         dcm = pydicom.dcmread(os.path.join(test_path, f))
#         if hasattr(dcm, "SliceLocation"):
#             slices.append(dcm)
#         else:
#             skipcount = skipcount + 1
#     print(f"file count: {len(slices)}")
#     print(f"skipped, no SliceLocation: {skipcount}")

#     slices = sorted(slices, key=lambda s: s.SliceLocation)
#     # ensure they are in the correct order
#     config = {
#         "series_uuid": None,
#         "full_test": True,
#         "report_progress": False,
#         "custom_parameters": {
#             "channelType": "Gabor",
#             "internalNoise": 2.25,
#             "lesionSet": "standard",
#             "resamples": 500,
#             "resamplingMethod": "Bootstrap",
#             "roiSize": 6,
#             "stepSize": 5,
#             "thresholdHigh": 300,
#             "thresholdLow": 0,
#             "windowLength": 15,
#             "saveResults": False,
#             "deleteAfterCompletion": False,
#         },
#     }
#     main(slices, config)


#!/usr/bin/env python
# coding: utf-8

"""
Unified CHO Patient-Specific Analysis Module

This module combines both Global Noise and Full CHO analysis capabilities
with optional progress tracking. It can run either:
1. Global Noise Analysis (simple, fast) - test=False
2. Full CHO Analysis (comprehensive) - test=True

With optional progress reporting for live updates.

Algorithm aligned with CHO_Calculation_Patient_Specific_skimage_Canny_edge_v26_copy.py:
  * MTF50/MTF10 looked up from Recon_Kernels.xlsx (manufacturer/model/kernel)
  * Damped-cosine PSF simulation via parametric MTF model (mtf2d = exp(-(fr/fc)^n))
  * Pre-sampling MTF computed via fixed 2048-point FFT with polar averaging
  * Body-part-aware SSDE parameters (abdomen vs head)
  * Dilated Canny edges (1.5 mm physical buffer)
  * NPS computed once at the middle window
  * extract_ROIs final safety filter (re-validates each ROI std)
"""

import numpy as np
from typing import Callable, Optional, Tuple
import os
import time
import math
import gc
import json

from scipy import ndimage
from skimage import feature
from skimage import measure
from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation
import scipy.io as sio
from scipy.optimize import minimize
from scipy.interpolate import griddata, interp1d, make_interp_spline

from numpy.lib.stride_tricks import sliding_window_view

from dicom_parser import DicomParser
from ct_sliding_window import CTSlidingWindow
from progress_tracker import progress_tracker

CONSTANT_RECON_FOV = 340
CONSTANT_PATIENT_MATRIX = 512

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lesion data + reconstruction kernel lookup table
LESION_FILE = os.path.join(SRC_DIR, "data", "Patient02-411-920_Lesion1.mat")
PSF_FILE = os.path.join(SRC_DIR, "data", "EID_PSF_Br44.mat")
KERNEL_FILE = os.path.join(SRC_DIR, "data", "Recon_Kernels.xlsx")


# ============================================================================
# Basic numerical helpers
# ============================================================================


def normal_round(n):
    """Round to nearest integer with .5 rounding up."""
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def round_to_nearest_odd(x):
    """Round to nearest odd integer for ROI sizes."""
    rounded = np.round(x).astype(int)
    return np.where(
        rounded % 2 == 0, np.where(rounded > x, rounded - 1, rounded + 1), rounded
    )


def round_to_nearest_even(num):
    return round(num + 0.5) if num % 2 != 0 else num


def get_lesion_configuration(lesion_set):
    """Get lesion configuration based on the selected set."""
    lesion_configs = {
        "standard": {
            "contrasts": [-30, -30, -10, -30, -50],
            "sizes": [3, 9, 6, 6, 6],
            "roi_sizes": [14, 19, 17, 17, 17],
        },
        "low-contrast": {
            "contrasts": [-15, -15, -5, -15, -25],
            "sizes": [3, 9, 6, 6, 6],
            "roi_sizes": [14, 19, 17, 17, 17],
        },
        "high-contrast": {
            "contrasts": [-60, -60, -20, -60, -50],
            "sizes": [3, 9, 6, 6, 6],
            "roi_sizes": [14, 19, 17, 17, 17],
        },
    }
    return lesion_configs.get(lesion_set, lesion_configs["standard"])


def convert_numpy_to_python(obj):
    """Convert numpy data types to native Python types and handle NaN values."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in ["f", "c"]:
            obj_clean = np.where(np.isnan(obj), None, obj)
            return obj_clean.tolist()
        else:
            return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif (
        np.isnan(obj)
        if isinstance(obj, (int, float)) and not isinstance(obj, bool)
        else False
    ):
        return None
    else:
        return obj


def safe_json_dumps(obj):
    """Convert object to JSON string, handling NaN values safely."""
    converted = convert_numpy_to_python(obj)
    return json.dumps(converted, allow_nan=False)


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


def interpolate_grid(X0, Y0, L0, XX, YY, method):
    """Interpolate 2D data to a new grid using specified method."""
    X0 = np.asarray(X0)
    Y0 = np.asarray(Y0)
    L0 = np.asarray(L0)
    XX = np.asarray(XX)
    YY = np.asarray(YY)
    grid_points = (X0.flatten(), Y0.flatten())
    values = L0.flatten()
    return griddata(grid_points, values, (XX, YY), method=method)


def center_crop_or_pad(img, out_size):
    out_h, out_w = out_size, out_size
    in_h, in_w = img.shape
    out = np.zeros((out_h, out_w), dtype=img.dtype)
    src_y0 = max(0, (in_h - out_h) // 2)
    src_x0 = max(0, (in_w - out_w) // 2)
    src_y1 = min(in_h, src_y0 + out_h)
    src_x1 = min(in_w, src_x0 + out_w)
    dst_y0 = max(0, (out_h - in_h) // 2)
    dst_x0 = max(0, (out_w - in_w) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


# ============================================================================
# NPS / variance helpers
# ============================================================================


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
    """Calculate statistics of 1D NPS."""
    peakfrequencyIndex = np.where(nps1d == np.max(nps1d))[0][0]
    peakfrequency = peakfrequencyIndex * unit
    n = len(nps1d)
    Spatial_freq = np.arange(n) * unit
    p = nps1d / np.sum(nps1d)
    p = np.squeeze(p)
    Spatial_freq = np.squeeze(Spatial_freq)
    fav = np.sum(Spatial_freq * p)
    minfrequencyIndex = np.where(nps1d < 0.10 * np.max(nps1d))[0]
    try:
        freq = np.where(minfrequencyIndex > peakfrequencyIndex)[0][0]
        min10percent_frequency = minfrequencyIndex[freq] * unit
    except IndexError:
        min10percent_frequency = np.nan
    npoint = 10
    k = np.polyfit(np.arange(npoint) * unit, nps1d[:npoint], 1)
    return fav, peakfrequency, k, min10percent_frequency


def pad_to_shape_centered(arr, target_shape):
    pad_width = []
    for s, t in zip(arr.shape, target_shape):
        total = max(t - s, 0)
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def ROI_to_NPS_Sum(ROI_size, ROI_All, dx, dy):
    """Compute 1D NPS from ROI images, averaging over radial frequencies."""
    if ROI_size < 20:
        nnn = 8 * 20
    else:
        nnn = 8 * ROI_size

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

        roipad = pad_to_shape_centered(roi, (nnn, nnn))
        roipad_fft = np.fft.fftshift(np.abs(np.fft.fft2(roipad)))
        roipad_fft = roipad_fft**2 * dx * dy / (ROI_size * ROI_size)
        nps2D = roipad_fft

        polar1 = np.zeros((cc, 360), dtype=np.float32)
        polar1[valid_mask] = nps2D[xi_all[valid_mask], yi_all[valid_mask]]

        nps1d = np.mean(polar1, axis=1).reshape((cc, 1))
        NPS_1D_sum += nps1d

    Spatial_freq = np.arange(cc).astype(np.float32) * unit
    return Spatial_freq, NPS_1D_sum, noise_level_sum, unit


# ============================================================================
# Recon kernel MTF lookup (NEW - from v26_copy)
# ============================================================================


def get_mtf_from_excel(kernel_file, manufacturer, model, kernel):
    """
    Read MTF50 and MTF10 values from Recon_Kernels.xlsx.

    Falls back to (0.434, 0.730) if the file or row is missing.
    """
    try:
        import pandas as pd

        df = pd.read_excel(kernel_file, header=0)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        for _, row in df.iterrows():
            if (
                row["Manufacturer"].upper() == str(manufacturer).strip().upper()
                and row["Model"].upper() == str(model).strip().upper()
                and row["Kernel"].upper() == str(kernel).strip().upper()
            ):
                mtf50 = float(row["MTF50"])
                mtf10 = float(row["MTF10"])
                print(
                    f"Loaded MTF from Excel: {manufacturer} {model} {kernel} "
                    f"-> MTF50={mtf50:.3f}, MTF10={mtf10:.3f}"
                )
                return mtf50, mtf10

        raise ValueError(f"No matching row found for {manufacturer} {model} {kernel}")

    except Exception as e:
        print(f"Error reading kernel file '{kernel_file}': {e}")
        print("Falling back to default values (MTF50=0.434, MTF10=0.730)")
        return 0.434, 0.730


# ============================================================================
# PSF simulation + presampling MTF (REPLACED with v26_copy versions)
# ============================================================================


def simulate_psf_damped_cosine(mtf50, mtf10=np.nan, size=513, wire_fov_mm=50.0):
    """
    Damped-cosine / Gaussian PSF derived from a parametric radial MTF.

    - mtf10 is None or NaN -> pure Gaussian using analytic sigma from mtf50
    - both mtf50 and mtf10 -> radial MTF mtf2d = exp(-(fr/fc)^n) with
      n = log(log(10)/log(2)) / log(mtf10/mtf50) and fc = mtf50 / log(2)^(1/n)
      The PSF is the inverse FFT of that MTF.

    This is the v26_copy parametric formulation (no L-BFGS-B optimization).
    """
    if mtf10 is None:
        mtf10 = np.nan

    # Working FFT size: at least Nyquist for max requested MTF, and at least `size`
    working_size = max(
        int(2 ** np.ceil(np.log2(wire_fov_mm * np.nanmax([mtf50, mtf10]) * 2)) + 1),
        size,
    )
    dx = wire_fov_mm / working_size

    if np.isnan(mtf10):
        # Pure Gaussian PSF: sigma analytic from mtf50
        sigma = np.sqrt(np.log(2) / (2 * np.pi**2)) / mtf50
        c = np.arange(working_size) - working_size // 2
        xx, yy = np.meshgrid(c * dx, c * dx)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    else:
        n = np.log(np.log(10) / np.log(2)) / np.log(mtf10 / mtf50)
        fc = mtf50 / np.log(2) ** (1.0 / n)

        f = np.fft.fftshift(np.fft.fftfreq(working_size, d=dx))
        fx, fy = np.meshgrid(f, f)
        fr = np.hypot(fx, fy)

        mtf2d = np.exp(-((fr / fc) ** n))
        psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mtf2d))))

    psf /= psf.sum()
    return psf, dx


def compute_presampling_mtf(psf, unit_dx, maxfreq=100.0):
    """
    Compute radial 1D MTF using fixed 2048-point FFT (v26_copy version).
    Returns (freq, mtf, mtf_eval) where mtf_eval = [MTF50, MTF10, MTF2].
    """
    psf = np.asarray(psf, dtype=np.float64)

    # 1. Normalize
    psf_norm = psf / np.sum(psf)

    # 2. Fixed 2048-point FFT size
    N_fft = 2048

    # 3. Center & pad
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

    # 4. 2D FFT
    psf_shift = np.fft.ifftshift(psf_padded)
    FT = np.fft.fft2(psf_shift)
    mtf2d = np.abs(FT)
    mtf2d = np.fft.fftshift(mtf2d)
    mtf2d /= mtf2d.max()

    # 5. Frequency axis
    df = 1.0 / (unit_dx * N_fft)
    freq_full = np.fft.fftshift(np.fft.fftfreq(N_fft, d=unit_dx))
    freq_pos = freq_full[freq_full >= 0]

    # 6. Polar averaging
    centre = N_fft // 2
    deg = 360
    oo = 2 * np.pi * np.linspace(0, deg - 1, deg) / deg

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

    mtf = np.mean(mtf_polar, axis=1)

    # 7. MTF50 / MTF10 / MTF2
    def get_mtf_freq(target):
        idx = np.where(mtf <= target)[0]
        return freq[idx[0]] if len(idx) > 0 else np.nan

    mtf_eval = np.array([get_mtf_freq(0.51), get_mtf_freq(0.105), get_mtf_freq(0.02)])

    return freq, mtf, mtf_eval


# ============================================================================
# Lesion signal preparation (REWRITTEN to v26_copy version)
# ============================================================================


def prepare_Lesion_sig(
    lesion_file,
    kernel_file,
    manufacturer,
    model,
    kernel,
    lesion_con_target,
    target_lesion_width,
    patient_FOV,
    patient_matrix,
    ROI_size,
    PSF_sim=None,
    dx_psf=None,
    wire_fov_mm=50,
    wire_Matrix_size=512,
    mtf50=None,
    mtf10=np.nan,
):
    """
    Prepare lesion signal: scale -> contrast -> PSF convolution.

    PSF is simulated from MTF50/MTF10 (looked up from Recon_Kernels.xlsx if
    not supplied). PSF_sim/dx_psf can be passed in to avoid re-simulating
    across multiple lesion contrast/size combinations.
    """
    # 1. MTF lookup
    if mtf50 is None:
        mtf50, mtf10 = get_mtf_from_excel(kernel_file, manufacturer, model, kernel)

    # 2. Load lesion VOI
    patient_data = sio.loadmat(lesion_file)
    lesion = patient_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
    mask = patient_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]
    loc = round(lesion.shape[-1] / 2)
    L0 = lesion[:, :, loc]
    M0 = mask[:, :, loc].astype(bool)

    # 3. Scale lesion to target physical size
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

    # 4. Simulate PSF (or reuse passed-in one)
    pixel_spacing_patient = patient_FOV / patient_matrix
    if PSF_sim is None or dx_psf is None:
        psf_size = wire_Matrix_size + (wire_Matrix_size % 2 == 0)
        PSF_sim, dx_psf = simulate_psf_damped_cosine(
            mtf50=mtf50, mtf10=mtf10, size=psf_size, wire_fov_mm=wire_fov_mm
        )

    freq, mtf, mtf_eval = compute_presampling_mtf(PSF_sim, dx_psf)

    # Resample simulated PSF to patient pixel spacing
    scaling_factor = dx_psf / pixel_spacing_patient
    os0_psf = np.floor(np.array(PSF_sim.shape) * scaling_factor).astype(int)

    x0_psf = np.linspace(0, PSF_sim.shape[1] - 1, PSF_sim.shape[1])
    y0_psf = np.linspace(0, PSF_sim.shape[0] - 1, PSF_sim.shape[0])
    X0_psf, Y0_psf = np.meshgrid(x0_psf, y0_psf)

    x_out_psf = np.linspace(0, PSF_sim.shape[1] - 1, os0_psf[1])
    y_out_psf = np.linspace(0, PSF_sim.shape[0] - 1, os0_psf[0])
    XX_psf, YY_psf = np.meshgrid(x_out_psf, y_out_psf)

    PSF_end = interpolate_grid(X0_psf, Y0_psf, PSF_sim, XX_psf, YY_psf, "linear")
    PSF_end /= PSF_end.sum()

    # 5. Convolve & crop to ROI
    Lesion_ext = np.zeros((80, 80))
    r0 = (Lesion_ext.shape[0] - Lesion.shape[0]) // 2
    c0 = (Lesion_ext.shape[1] - Lesion.shape[1]) // 2
    Lesion_ext[r0 : r0 + Lesion.shape[0], c0 : c0 + Lesion.shape[1]] = Lesion

    Lesion_conv = convolve2d(Lesion_ext, PSF_end, mode="full")

    r0 = (Lesion_conv.shape[0] - ROI_size) // 2
    c0 = (Lesion_conv.shape[1] - ROI_size) // 2
    Lesion_conv_end = Lesion_conv[r0 : r0 + ROI_size, c0 : c0 + ROI_size]

    return Lesion_conv_end, freq, mtf


# ============================================================================
# Integral image / std map / ROI extraction
# ============================================================================


def Integral_image(image, window_size=(3, 3), padding="constant"):
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
    return np.cumsum(np.cumsum(imageD, axis=0), axis=1)


def calculate_std_dev(
    integral_images,
    integral_images_square,
    integral_edge,
    roi_size,
    padding_size,
    image_size,
    threshold_low,
    threshold_high,
    edges,
    corrected=False,
):
    """
    Calculate per-pixel std from integral images, with edge & threshold masks.

    Edge buffer comes via integral_edge (already includes Canny + dilation
    when constructed in the main loop).
    """
    roi_size_even = int(roi_size) if roi_size % 2 == 0 else int(roi_size + 1)
    half_diff_size = round((padding_size - roi_size_even) / 2)
    half_roi_size = round(np.floor(roi_size / 2))

    end_h = integral_edge.shape[0] - half_diff_size
    end_w = integral_edge.shape[1] - half_diff_size

    integral_edge = integral_edge[half_diff_size:end_h, half_diff_size:end_w, :]
    integral_images = integral_images[half_diff_size:end_h, half_diff_size:end_w, :]
    integral_images_square = integral_images_square[
        half_diff_size:end_h, half_diff_size:end_w, :
    ]

    m = n = roi_size_even
    roi_area = m * n
    normalization_factor = roi_area / (roi_area - 1) if corrected else 1.0
    edge_threshold = 1.0 / roi_size

    std_all = np.zeros(
        [image_size[0], image_size[1], integral_images.shape[-1]], dtype=np.float32
    )

    for k in range(integral_images.shape[-1]):
        t = integral_images[:, :, k]
        mean_map = (t[m:, n:] + t[:-m, :-n] - t[m:, :-n] - t[:-m, n:]) / roi_area

        t2 = integral_images_square[:, :, k]
        mean_square = (t2[m:, n:] + t2[:-m, :-n] - t2[m:, :-n] - t2[:-m, n:]) / roi_area

        t3 = integral_edge[:, :, k]
        edge_impact = (t3[m:, n:] + t3[:-m, :-n] - t3[m:, :-n] - t3[:-m, n:]) / roi_area

        threshold_mask = (mean_map < threshold_low) | (mean_map > threshold_high)
        edge_mask = edge_impact >= edge_threshold
        combined_mask = threshold_mask | edge_mask | edges

        variance = mean_square - mean_map**2
        variance = np.maximum(variance, 0)
        std_map = np.sqrt(normalization_factor * variance)

        std_map[combined_mask] = np.nan

        if half_roi_size > 0:
            std_map[:half_roi_size, :] = np.nan
            std_map[-half_roi_size:, :] = np.nan
            std_map[:, :half_roi_size] = np.nan
            std_map[:, -half_roi_size:] = np.nan

        std_all[:, :, k] = std_map

    return std_all


def extract_ROIs(STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size):
    """
    Greedy non-overlapping ROI selection sorted by ascending std,
    followed by a final safety filter that re-validates each ROI's
    actual std vs Thre_SD (v26_copy behaviour).
    """
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
    if ROIs_array.size > 0:
        ROIs_array = np.transpose(ROIs_array, (1, 2, 0))
        Total_NPS_No = ROIs_array.shape[-1]

    # ============= FINAL SAFETY FILTER (v26_copy) =============
    if Total_NPS_No > 0:
        actual_std = np.array([np.std(roi) for roi in ROIs])
        good_mask = actual_std <= Thre_SD
        ROIs_array = ROIs_array[:, :, good_mask]
        Total_NPS_No = ROIs_array.shape[-1]
        rejected = len(actual_std) - Total_NPS_No
        if rejected > 0:
            print(
                f"extract_ROIs: {rejected} ROIs rejected (std > {Thre_SD:.3f} HU). "
                f"Final selected: {Total_NPS_No} ROIs"
            )

    return ROIs_array, Total_NPS_No


# ============================================================================
# CHO channels
# ============================================================================


def Laguerre2D(order, a, b, cx, cy, X, Y):
    """Laguerre-Gauss channel function."""
    val1 = np.zeros(X.size)
    ga = 2 * np.pi * ((X - cx) ** 2 / (a**2) + (Y - cy) ** 2 / (b**2))
    for jp in range(order + 1):
        val1 = val1 + (-1) ** jp * np.prod(np.linspace(1, order, order)) / (
            np.prod(np.linspace(1, jp, jp))
            * np.prod(np.linspace(1, order - jp, order - jp))
        ) * (ga**jp) / np.prod(np.linspace(1, jp, jp))
    return np.exp(-ga / 2) * val1


def Gabor2D(fc, wd, theta, beta, cx, cy, X, Y):
    """Gabor channel function."""
    return np.exp(-4 * np.log(2) * ((X - cx) ** 2 + (Y - cy) ** 2) / (wd**2)) * np.cos(
        2 * np.pi * fc * ((X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)) + beta
    )


def channel_selection(Chnl, inputArg1, inputArg2, inputArg3="0"):
    """Configure CHO channel parameters."""
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
    """Generate CHO channel matrix."""
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
    else:
        raise ValueError("Unknown channel type")
    return channelMatrix


def CHO_patient_with_resampling(
    sig_true, bkg_ordered, channelMatrix, internalNoise, Resampling_method
):
    """CHO observer detectability with resampling."""
    N_total_bkg = bkg_ordered.shape[1]

    if Resampling_method == "Bootstrap":
        rand_scanSelect_bkg = np.random.randint(N_total_bkg, size=N_total_bkg)
    elif Resampling_method == "Shuffle":
        rand_scanSelect_bkg = np.random.permutation(N_total_bkg)
    else:
        raise ValueError("Unknown resampling method")

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


# ============================================================================
# Per-image dose / Dw analysis
# ============================================================================


def compute_image_analysis(curr_file, pixel_roi_mm2):
    """Per-slice CTDIvol, Dw, and central coronal column."""
    img = curr_file.pixel_array
    img_rescaled = img * curr_file.RescaleSlope + curr_file.RescaleIntercept
    mask = img_rescaled >= -260
    binary_mask = ndimage.binary_fill_holes(mask)
    if binary_mask is not None:
        binary_mask = binary_mask.astype(bool)
    else:
        binary_mask = np.zeros_like(mask, dtype=bool)

    labels, num = measure.label(binary_mask, return_num=True)
    if num > 0:
        largest_region = np.argmax(np.bincount(labels.flat)[1:]) + 1
        binary_img = labels == largest_region
    else:
        binary_img = np.zeros_like(binary_mask, dtype=bool)

    pixel_count = np.sum(binary_img)
    roi_mm2 = pixel_count * pixel_roi_mm2
    img_rescaled_masked = img_rescaled[binary_img]
    hu_mean = np.mean(img_rescaled_masked) if img_rescaled_masked.size > 0 else 0.0

    coronal_slice = img_rescaled[img.shape[1] // 2, :]
    ctdi = curr_file.get("CTDIvol", np.nan)
    dw = 2 * np.sqrt((hu_mean / 1000 + 1) * roi_mm2 / np.pi) if roi_mm2 > 0 else 0.0
    return ctdi, dw, coronal_slice


def get_ssde_params_for_body_part(body_part_str):
    """
    Body-part-aware SSDE parameter selection (v26_copy).

    Returns (para_a, para_b, body_part_label).
    """
    if body_part_str:
        bp = body_part_str.strip().upper()
        if "ABDOMEN" in bp or "PELVI" in bp:
            return 3.704369, 0.03671937, "abdomen"
        if "HEAD" in bp or "BRAIN" in bp or "SKULL" in bp:
            return 1.874799, 0.03871313, "head"
    # Default to abdomen
    return 3.704369, 0.03671937, "abdomen-default"


def hu_to_image(hu_array):
    """Convert HU values to 8-bit image intensities."""
    hu_min = np.min(hu_array)
    hu_max = np.max(hu_array)
    if hu_max == hu_min:
        return np.zeros_like(hu_array, dtype=np.uint8)
    hu_normalized = (hu_array - hu_min) / (hu_max - hu_min)
    return (hu_normalized * 255).astype(np.uint8)


# ============================================================================
# Global Noise Analysis (preserved)
# ============================================================================


def run_global_noise_analysis(files, config):
    """Run Global Noise CHO Analysis (fast path)."""
    custom_params = config["custom_parameters"]
    start = time.time()

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            15,
            "Starting Global Noise CHO calculation...",
            "preprocessing",
        )

    dcm_parser = DicomParser(files)
    patient, study, scanner, series, ct = dcm_parser.extract_core()

    n_images = len(files)
    first_file = files[0]
    last_file = files[-1]

    first_slice_location = first_file.SliceLocation
    last_slice_location = last_file.SliceLocation
    slice_interval = series["slice_interval_mm"] / 10
    rows = int(series.get("rows", 512))
    cols = int(series.get("columns", 512))
    dx_mm, dy_mm = series["pixel_spacing_mm"]

    dx_cm = dx_mm / 10
    pixel_roi_mm2 = dx_mm * dy_mm

    roi_diameter = round(0.6 / dx_cm)
    Thr1 = 0
    Thr2 = 150

    img_size = (rows, cols)
    sample_interval = round(3 / slice_interval) if slice_interval > 0 else 1
    if slice_interval >= 3:
        sample_interval = 1

    n_sel_images = (n_images + sample_interval - 1) // sample_interval

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            15,
            f"Analyzing {n_sel_images} selected images for noise calculation",
            "preprocessing",
        )

    ROI_size2 = round_to_nearest_even(roi_diameter)
    window_gen = CTSlidingWindow(
        window_length=15,
        step_size=0.3,
        padding_size=ROI_size2,
        sigma=5,
        window_unit="cm",
        step_unit="cm",
        use_cache=False,
    )
    total_sections = window_gen.get_num_windows(files)
    ctdi_all = np.zeros(total_sections)
    location = np.zeros(total_sections)
    dw = np.zeros(total_sections)
    coronal_view = np.zeros((cols, total_sections))
    STD_all = np.zeros([rows, cols, total_sections], dtype=np.float32)

    for window, metadata in window_gen.generate_windows(files):
        i = metadata["slice_indices"][0]
        idx = metadata["window_idx"]
        if (
            config["report_progress"]
            and config["series_uuid"]
            and (i % 10 == 0 or i == n_images - 1)
        ):
            progress_percentage = 25 + ((i / n_images) * (85 - 25))
            progress_tracker.update_progress(
                config["series_uuid"],
                int(progress_percentage),
                f"Processing slice {i+1}/{n_images}",
                "analysis",
            )

        curr_file = files[i]
        ctdi_all[idx], dw[idx], coronal_view[:, idx] = compute_image_analysis(
            curr_file, pixel_roi_mm2
        )
        location[idx] = metadata["relative_mid_cm"]
        Edges = window["edges"]
        Intel_Edge = window["integral_edges"]
        Intel_images = window["integral_images"]
        Intel_images_Square = window["integral_images_square"]
        STD_all[:, :, idx] = np.squeeze(
            calculate_std_dev(
                Intel_images,
                Intel_images_Square,
                Intel_Edge,
                roi_diameter,
                ROI_size2,
                img_size,
                Thr1,
                Thr2,
                Edges,
                corrected=False,
            )
        )
    window_gen.clear_cache()

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"], 85, "Calculating final metrics...", "finalizing"
        )

    max_value = np.nanmax(STD_all) if STD_all.size > 0 else 1.0
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 1.0
    h_Values, edges = np.histogram(STD_all.flatten(), bins=np.arange(0, max_value, 0.2))
    whichbin_SD = int(np.argmax(h_Values))
    global_noise_level = float(edges[whichbin_SD]) if len(edges) > whichbin_SD else 0.0

    # Body-part-aware SSDE
    body_part_value = files[0].get((0x0018, 0x0015), None)
    body_part_str = str(body_part_value.value) if body_part_value is not None else ""
    para_a, para_b, body_part_label = get_ssde_params_for_body_part(body_part_str)
    print(f"SSDE body part: {body_part_label}")

    mean_ctdi_all = np.mean(ctdi_all)
    mean_dw = np.mean(dw) / 10
    f = para_a * math.exp(-para_b * mean_dw)
    mean_ssde = f * mean_ctdi_all
    f_inc = para_a * np.exp(-para_b * (dw / 10))
    ssde = f_inc * ctdi_all
    scan_length_cm = (last_slice_location - first_slice_location) / 10
    dlp_ctdi_vol = scan_length_cm * mean_ctdi_all
    dlp_ssde = scan_length_cm * mean_ssde

    processing_time = time.time() - start

    series["uuid"] = config["series_uuid"]
    series["image_count"] = n_images
    series["scan_length_cm"] = scan_length_cm

    results = {
        "average_frequency": None,
        "average_index_of_detectability": None,
        "average_noise_level": None,
        "cho_detectability": None,
        "ctdivol": ctdi_all,
        "ctdivol_avg": mean_ctdi_all,
        "dlp": dlp_ctdi_vol,
        "dlp_ssde": dlp_ssde,
        "dw": dw,
        "dw_avg": mean_dw,
        "location": location.tolist(),
        "location_sparse": None,
        "noise_level": None,
        "nps": None,
        "peak_frequency": None,
        "percent_10_frequency": None,
        "processing_time": processing_time,
        "spatial_frequency": None,
        "spatial_resolution": None,
        "ssde": mean_ssde,
        "ssde_inc": ssde.tolist() if isinstance(ssde, np.ndarray) else ssde,
        "coronal_view": coronal_view,
        "global_noise_level": global_noise_level,
        "body_part": body_part_label,
    }

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"], 95, "Saving results to database...", "finalizing"
        )

    converted_results = convert_numpy_to_python(results)

    if custom_params.get("saveResults"):
        try:
            from results_storage import cho_storage

            success = cho_storage.save_results(
                patient, study, scanner, series, ct, converted_results
            )
            if success:
                print(
                    f"Global noise results saved for series {series['series_instance_uid']}"
                )
            else:
                print(
                    f"Failed to save global noise results for series {series['series_instance_uid']}"
                )
        except Exception as e:
            print(f"Error saving to database: {str(e)}")

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            100,
            "Global Noise CHO calculation completed successfully",
            "completed",
        )
        progress_tracker.complete_calculation(
            config["series_uuid"],
            {
                "patient": patient,
                "study": study,
                "scanner": scanner,
                "series": series,
                "ct": ct,
                "results": converted_results,
            },
        )

    if custom_params.get("deleteAfterCompletion"):
        import orthanc

        orthanc.RestApiDelete(f'/series/{config["series_uuid"]}')
        print(f"Requested deletion of series {config['series_uuid']}")

    print(f"Global Noise processing time: {processing_time:.2f} seconds")
    return {
        "patient": patient,
        "study": study,
        "scanner": scanner,
        "series": series,
        "ct": ct,
        "results": converted_results,
    }


# ============================================================================
# Full CHO Analysis (algorithm aligned with v26_copy)
# ============================================================================


def run_full_cho_analysis(files, config):
    """
    Full CHO analysis with lesion detectability calculation.

    Algorithm from CHO_Calculation_Patient_Specific_skimage_Canny_edge_v26_copy.py
    adapted to operate on already-loaded DICOM file objects with progress
    tracking and database persistence.
    """
    print("Config")
    for key, val in config.items():
        if key == "custom_parameters":
            for param_key, param_val in val.items():
                print(f"  {param_key}: {param_val}")
        else:
            print(f"  {key}: {val}")

    custom_params = config["custom_parameters"]
    start = time.time()

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            10,
            "Loading lesion models...",
            "loading",
        )

    # ---- DICOM metadata ----
    dcm_parser = DicomParser(files)
    patient, study, scanner, series, ct = dcm_parser.extract_core()

    n_images = len(files)
    first_file = files[0]
    last_file = files[-1]

    SliceLocation1 = first_file.SliceLocation
    SliceLocation_end = last_file.SliceLocation

    rows = int(series.get("rows", 512))
    cols = int(series.get("columns", 512))
    Im_Size = (rows, cols)
    dx_mm, dy_mm = series["pixel_spacing_mm"]
    dx = dx_mm / 10  # cm/pixel
    dy = dy_mm / 10
    Sliceinterval = float(series["slice_interval_mm"])

    patient_FOV = float(first_file.ReconstructionDiameter)  # mm
    patient_matrix = int(first_file.Rows)
    pixel_size = patient_FOV / patient_matrix  # mm/pixel

    # ---- Lesion configuration ----
    lesion_config = get_lesion_configuration(custom_params["lesionSet"])
    Lesion_Contrasts = lesion_config["contrasts"]
    Lesion_Size = lesion_config["sizes"]
    ROI_sizes_mm = lesion_config["roi_sizes"]

    ROI_sizes = round_to_nearest_odd(np.array(ROI_sizes_mm) / pixel_size)

    wire_fov_mm = 50
    wire_Matrix_size = 512

    # ---- Reconstruction kernel info (for MTF lookup) ----
    manufacturer = scanner.get("manufacturer") or getattr(
        first_file, "Manufacturer", "Siemens"
    )
    model = scanner.get("model_name") or getattr(
        first_file, "ManufacturerModelName", "EID Force"
    )
    kernel = series.get("convolution_kernel") or getattr(
        first_file, "ConvolutionKernel", "Br44"
    )
    if isinstance(kernel, (list, tuple)):
        kernel = kernel[0] if len(kernel) > 0 else "Br44"

    # Allow caller to override via custom_params
    mtf50 = custom_params.get("mtf50")
    mtf10 = custom_params.get("mtf10", np.nan)
    if mtf50 is None:
        mtf50, mtf10 = get_mtf_from_excel(KERNEL_FILE, manufacturer, model, kernel)

    print(f"Using MTF50={mtf50:.3f}, MTF10={mtf10}")

    # ---- Simulate PSF ONCE, reuse across all lesion contrasts/sizes ----
    psf_size = wire_Matrix_size + (wire_Matrix_size % 2 == 0)
    PSF_sim, dx_psf = simulate_psf_damped_cosine(
        mtf50=mtf50,
        mtf10=mtf10 if mtf10 is not None else np.nan,
        size=psf_size,
        wire_fov_mm=wire_fov_mm,
    )

    # ---- Build lesion signals ----
    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"], 15, "Building lesion signals...", "preprocessing"
        )

    Lesion_sigs = []
    for i in range(len(Lesion_Contrasts)):
        Lesion_sig, freq, mtf = prepare_Lesion_sig(
            lesion_file=LESION_FILE,
            kernel_file=KERNEL_FILE,
            manufacturer=manufacturer,
            model=model,
            kernel=kernel,
            lesion_con_target=Lesion_Contrasts[i],
            target_lesion_width=Lesion_Size[i],
            patient_FOV=patient_FOV,
            patient_matrix=patient_matrix,
            ROI_size=int(ROI_sizes[i]),
            PSF_sim=PSF_sim,
            dx_psf=dx_psf,
            wire_fov_mm=wire_fov_mm,
            wire_Matrix_size=wire_Matrix_size,
            mtf50=mtf50,
            mtf10=mtf10 if mtf10 is not None else np.nan,
        )
        Lesion_sigs.append(Lesion_sig)

    # ---- CTDIvol, Dw, coronal projection per slice ----
    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            25,
            f"Computing dose metrics for {n_images} slices...",
            "analysis",
        )

    CTDI_All = np.zeros(n_images)
    Dw = np.zeros(n_images)
    coronal_image = np.zeros((Im_Size[0], n_images))
    slice_location = np.zeros(n_images)
    pixel_roi_mm2 = dx_mm * dy_mm

    for i in range(n_images):
        curr_file = files[i]
        CTDI_All[i], Dw[i], coronal_image[:, i] = compute_image_analysis(
            curr_file, pixel_roi_mm2
        )
        slice_location[i] = curr_file.SliceLocation

    slice_location = slice_location - slice_location.min()
    Mean_CTDI_All = float(np.mean(CTDI_All))
    Mean_Dw = float(np.mean(Dw) / 10)

    # ---- Body-part-aware SSDE (v26_copy) ----
    body_part_value = first_file.get((0x0018, 0x0015), None)
    body_part_str = str(body_part_value.value) if body_part_value is not None else ""
    para_a, para_b, body_part_label = get_ssde_params_for_body_part(body_part_str)
    print(f"SSDE body part: {body_part_label} (a={para_a}, b={para_b})")

    f = para_a * np.exp(-para_b * Mean_Dw)
    ssde_inc = f * CTDI_All
    SSDE = float(f * Mean_CTDI_All)
    Scan_len = (SliceLocation_end - SliceLocation1) / 10
    DLP_CTDIvol_L = Scan_len * Mean_CTDI_All
    DLP_SSDE = Scan_len * SSDE

    # ---- ROI / threshold parameters ----
    val = 0.6 / dx
    ROI_size_N = int(2 * np.round((val - 1) / 2) + 1)
    Half_ROI_size_N = round(np.floor(ROI_size_N / 2))
    Thr1 = custom_params.get("thresholdLow", 0)
    Thr2 = custom_params.get("thresholdHigh", 150)
    numResample = custom_params.get("resamples", 500)
    internalNoise = custom_params.get("internalNoise", 2.25)
    Resampling_method = custom_params.get("resamplingMethod", "Bootstrap")

    # ---- CHO Gabor channels ----
    class Chnl:
        Chnl_Toggle = "Gabor"

    Chnl.Chnl_Toggle = custom_params.get("channelType", "Gabor")
    if Chnl.Chnl_Toggle == "Gabor":
        Gabor_passband = "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]"
        Gabor_theta = "[0, pi/3, 2*pi/3]"
        Gabor_beta = "0"
        Chnl = channel_selection(Chnl, Gabor_passband, Gabor_theta, Gabor_beta)
    elif Chnl.Chnl_Toggle == "Laguerre-Gauss":
        Chnl = channel_selection(Chnl, 6, 3)

    # ---- Sliding-window CHO loop (v26_copy: 50mm windows, ~50% overlap) ----
    if Sliceinterval <= 0:
        raise ValueError("Slice interval must be > 0")
    N_sub = max(1, round(50 / Sliceinterval))
    num_groups = n_images // N_sub - 2

    if num_groups < 1:
        raise ValueError(
            f"Not enough slices ({n_images}) for sliding window of {N_sub} "
            f"slices/group. Need at least {3 * N_sub}."
        )

    All_dps = np.zeros([num_groups, len(Lesion_sigs)])
    dps_location_list = []
    Noise_level_local = np.zeros([num_groups, 1])
    Total_NPS_Num = 0
    NPS_1D_sum = 0
    Spatial_freq = np.array([])
    noise_level_sum = 0.0
    NPS_Cal = True
    unit = 0.0

    Padding_size = 2 * round(1.2 / dx)
    Padding_size = int(Padding_size) + (Padding_size % 2)

    for mm in range(1, n_images // N_sub - 1):
        if config["report_progress"] and config["series_uuid"]:
            progress_percentage = 30 + ((mm / num_groups) * (85 - 30))
            progress_tracker.update_progress(
                config["series_uuid"],
                int(progress_percentage),
                f"Processing CHO section {mm}/{num_groups}",
                "analysis",
            )

        N1 = (mm - 1) * N_sub
        N2 = mm * N_sub
        section_depth = N2 + N_sub * 2 - N1

        Intel_images = np.zeros(
            [Im_Size[0] + Padding_size, Im_Size[1] + Padding_size, section_depth],
            dtype=np.float32,
        )
        Intel_images_Square = np.zeros_like(Intel_images)
        Intel_Edge = np.zeros_like(Intel_images)
        Images_Section = np.zeros(
            [Im_Size[0], Im_Size[1], section_depth], dtype=np.float32
        )

        Edges = None
        for nn in range(N1, N2 + N_sub * 2):
            curr = files[nn]
            dps_location_list.append(curr.SliceLocation)
            im1 = curr.pixel_array
            im10 = im1 * curr.RescaleSlope + curr.RescaleIntercept

            # Canny + dilation with ~1.5mm physical buffer (v26_copy)
            canny_edges = feature.canny(im10, sigma=5)
            physical_buffer_mm = 1.5
            kernel_size = max(3, int(np.round(physical_buffer_mm / (dx * 10))))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            structure = np.ones((kernel_size, kernel_size), dtype=bool)
            dilated_edges = binary_dilation(canny_edges, structure=structure)
            Edges = dilated_edges  # last slice's edges used as fallback mask

            Images_Section[:, :, nn - N1] = im10
            Intel_Edge[:, :, nn - N1] = Integral_image(
                dilated_edges, [Padding_size, Padding_size], "replicate"
            )
            Intel_images[:, :, nn - N1] = Integral_image(
                im10, [Padding_size, Padding_size], "replicate"
            )
            Intel_images_Square[:, :, nn - N1] = Integral_image(
                im10**2, [Padding_size, Padding_size], "replicate"
            )

        # Initial std map at the full noise-ROI size, used for histogram & NPS
        STD_all = calculate_std_dev(
            Intel_images,
            Intel_images_Square,
            Intel_Edge,
            ROI_size_N,
            Padding_size,
            Im_Size,
            Thr1,
            Thr2,
            Edges,
            corrected=False,
        )

        max_value = np.nanmax(STD_all)
        if not np.isfinite(max_value) or max_value <= 0:
            max_value = 1.0
        h_Values, edges_hist = np.histogram(
            STD_all.flatten(), bins=np.arange(0, max_value, 0.2)
        )
        whichbin_SD = int(np.argmax(h_Values))
        bin_edge = float(edges_hist[whichbin_SD])
        Noise_level_local[mm - 1] = bin_edge

        # NPS calculated ONCE at the middle window (v26_copy)
        if NPS_Cal and (mm == (num_groups // 2 + 1)):
            print(f"Calculating NPS from middle period (group {mm})")
            ROI_All_NPS, Total_NPS_No = extract_ROIs(
                STD_all, bin_edge, Half_ROI_size_N, Images_Section, Im_Size
            )
            num_nps_rois = min(200, Total_NPS_No)
            if num_nps_rois > 0:
                Spatial_freq, NPS_1D_sum, noise_level_sum, unit = ROI_to_NPS_Sum(
                    ROI_size_N, ROI_All_NPS[:, :, 0:num_nps_rois], dx, dy
                )
                Total_NPS_Num = num_nps_rois
                NPS_Cal = False

        del STD_all
        gc.collect()

        # 95.44% (2-sigma) threshold from histogram
        Left_Area = float(np.sum(h_Values[:whichbin_SD]))
        if whichbin_SD > 0 and Left_Area > 0:
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
            loc = int(np.where(Ratios == closest)[0][0])
            gap = whichbin_SD - loc
            Thre_SD = float(edges_hist[min(whichbin_SD + gap, len(edges_hist) - 1)])
        else:
            Thre_SD = bin_edge

        # Per-lesion CHO d'
        all_dps = []
        ROI_All_cache = None
        Total_NPS_No_cache = 0

        for ii in range(len(Lesion_sigs)):
            lesion_sig = Lesion_sigs[ii]
            ROI_size = int(ROI_sizes[ii])
            Half_ROI_size = round(np.floor(ROI_size / 2))

            # v26_copy reuses extracted ROIs for indices [3, 4]
            if ii in (3, 4) and ROI_All_cache is not None:
                ROI_All = ROI_All_cache
                Total_NPS_No = Total_NPS_No_cache
            else:
                STD_map_all = calculate_std_dev(
                    Intel_images,
                    Intel_images_Square,
                    Intel_Edge,
                    ROI_size,
                    Padding_size,
                    Im_Size,
                    Thr1,
                    Thr2,
                    Edges,
                    corrected=False,
                )
                ROI_All, Total_NPS_No = extract_ROIs(
                    STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size
                )
                ROI_All_cache = ROI_All
                Total_NPS_No_cache = Total_NPS_No

            if Total_NPS_No <= 0:
                all_dps.append(np.nan)
                continue

            Noise_ROI = ROI_All[:, :, 0:Total_NPS_No].copy()
            for j in range(Noise_ROI.shape[2]):
                Noise_ROI[:, :, j] -= np.mean(Noise_ROI[:, :, j])

            sample_idx = np.random.permutation(Noise_ROI.shape[2])
            channelMatrix = ChannelMatrix_Generation(Chnl, ROI_size)
            bkg_ordered = np.reshape(
                Noise_ROI[:, :, sample_idx], (ROI_size**2, len(sample_idx))
            )
            sig_true = np.reshape(lesion_sig, (ROI_size**2, 1))
            dp = CHO_patient_with_resampling(
                sig_true, bkg_ordered, channelMatrix, internalNoise, Resampling_method
            )
            all_dps.append(float(np.asarray(dp).mean()))

        All_dps[mm - 1, :] = all_dps

        del Intel_images, Intel_images_Square, Intel_Edge, Images_Section
        gc.collect()

    # ---- Aggregate ----
    dps_location_array = np.array(dps_location_list)
    dps_location_array = dps_location_array - dps_location_array.min()
    # Each group contains (N_sub * 3) slices (N1..N2+2*N_sub); reshape to one
    # location per group by averaging
    slices_per_group = N_sub * 3
    if dps_location_array.size >= num_groups * slices_per_group:
        dps_location_array = np.reshape(
            dps_location_array[: num_groups * slices_per_group],
            (num_groups, slices_per_group),
        )
        dps_location_array = np.mean(dps_location_array, axis=-1)
    else:
        # fallback: linspace across the scan
        dps_location_array = np.linspace(
            slice_location.min(), slice_location.max(), num_groups
        )

    Mean_loc_dps = np.mean(All_dps, axis=-1)
    Mean_All_dps = float(np.mean(Mean_loc_dps))

    if Total_NPS_Num > 0:
        NPS_1D = NPS_1D_sum / Total_NPS_Num
        Ave_noise_level_NPS = noise_level_sum / Total_NPS_Num
    else:
        NPS_1D = np.zeros((1,))
        Ave_noise_level_NPS = 0.0
    Ave_noise_level = float(np.mean(Noise_level_local))

    if NPS_1D.size > 1 and unit > 0:
        fav, peakfrequency, k_slope, min10percent_frequency = NPS_statistics(
            NPS_1D, unit
        )
    else:
        fav = peakfrequency = min10percent_frequency = np.nan

    elapsed = time.time() - start
    processing_time = elapsed

    series["uuid"] = config["series_uuid"]
    series["image_count"] = n_images
    series["scan_length_cm"] = Scan_len

    results = {
        "average_frequency": fav,
        "average_index_of_detectability": Mean_All_dps,
        "average_noise_level": Ave_noise_level,
        "cho_detectability": Mean_loc_dps.tolist(),
        "dps_location": dps_location_array.tolist(),
        "ctdivol": CTDI_All.tolist(),
        "ctdivol_avg": Mean_CTDI_All,
        "dlp": DLP_CTDIvol_L,
        "dlp_ssde": DLP_SSDE,
        "dw": Dw.tolist(),
        "dw_avg": Mean_Dw,
        "location": slice_location.tolist(),
        "location_sparse": dps_location_array.tolist(),
        "noise_level": Noise_level_local.flatten().tolist(),
        "nps": NPS_1D.flatten().tolist(),
        "peak_frequency": peakfrequency,
        "percent_10_frequency": min10percent_frequency,
        "processing_time": processing_time,
        "spatial_frequency": (
            Spatial_freq.tolist() if isinstance(Spatial_freq, np.ndarray) else []
        ),
        "spatial_resolution": float(mtf10) if mtf10 is not None else None,
        "ssde": SSDE,
        "ssde_inc": ssde_inc.tolist(),
        "coronal_view": coronal_image,
        "global_noise_level": Ave_noise_level,
        "elapsed_seconds": elapsed,
        "body_part": body_part_label,
        "mtf50": float(mtf50),
        "mtf10": float(mtf10) if mtf10 is not None and not np.isnan(mtf10) else None,
    }

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"], 95, "Saving results to database...", "finalizing"
        )

    converted_results = convert_numpy_to_python(results)

    if custom_params.get("saveResults"):
        try:
            from results_storage import cho_storage

            success = cho_storage.save_results(
                patient, study, scanner, series, ct, converted_results
            )
            if success:
                print(
                    f"Full analysis results saved for series {series['series_instance_uid']}"
                )
            else:
                print(
                    f"Failed to save full analysis results for series {series['series_instance_uid']}"
                )
        except Exception as e:
            print(f"Error saving to database: {str(e)}")

    if config["report_progress"] and config["series_uuid"]:
        progress_tracker.update_progress(
            config["series_uuid"],
            100,
            "Full CHO calculation completed successfully",
            "completed",
        )
        progress_tracker.complete_calculation(
            config["series_uuid"],
            {
                "patient": patient,
                "study": study,
                "scanner": scanner,
                "series": series,
                "ct": ct,
                "results": converted_results,
            },
        )

    if custom_params.get("deleteAfterCompletion"):
        import orthanc

        orthanc.RestApiDelete(f'/series/{config["series_uuid"]}')
        print(f"Requested deletion of series {config['series_uuid']}")

    print(f"Full CHO processing time: {processing_time:.2f} seconds")
    print(f"Peak frequency = {peakfrequency}")
    print(f"Average frequency = {fav}")
    print(f"Min 10% frequency = {min10percent_frequency}")
    print(f"Average noise level = {Ave_noise_level:.3f}")
    print(f"Mean detectability = {Mean_All_dps:.3f}")

    return {
        "patient": patient,
        "study": study,
        "scanner": scanner,
        "series": series,
        "ct": ct,
        "results": converted_results,
    }


def create_lesion_model(set, index, mtf50, mtf10, recon_diameter_mm, rows):
    """Create a lesion model for analysis"""

    lesion_contrasts = [-30, -30, -10, -30, -50]
    lesion_sizes = [3, 9, 6, 6, 6]
    roi_sizes = [21, 29, 25, 25, 25]

    if set == "low-contrast":
        lesion_contrasts = [int(c / 2) for c in lesion_contrasts]
    elif set == "high-contrast":
        lesion_contrasts = [int(c * 2) for c in lesion_contrasts]

    lesion_file_data = sio.loadmat(LESION_FILE)

    lesion_contrast = lesion_contrasts[index]
    lesion_size = lesion_sizes[index]
    roi_size = roi_sizes[index]

    patient_pixel_size = recon_diameter_mm / rows
    roi_size_px = int(np.round(roi_size / patient_pixel_size))
    if roi_size_px % 2 == 0:
        roi_size_px += 1

    lesion_volume = lesion_file_data["Patient"]["Lesion"][0][0]["VOI"][0][0]
    lesion_mask_volume = lesion_file_data["Patient"]["Lesion"][0][0]["LesionMask"][0][0]

    mid_slice_idx = round(lesion_volume.shape[-1] / 2)
    lesion_slice = lesion_volume[:, :, mid_slice_idx]
    mask_slice = lesion_mask_volume[:, :, mid_slice_idx].astype(bool)

    lesion_width_pixels_input = max_consecutive_ones_2d(mask_slice)

    mean_lesion_hu = np.mean(lesion_slice[mask_slice])
    hu_difference = mean_lesion_hu - lesion_contrast
    lesion_slice = lesion_slice - hu_difference
    lesion_slice[~mask_slice] = 0

    # --- Scale lesion to desired physical size (this is GOOD) ---
    target_width_pixels = lesion_size / patient_pixel_size
    scale_factor = target_width_pixels / lesion_width_pixels_input

    input_height, input_width = lesion_slice.shape
    output_shape = np.floor(np.array(lesion_slice.shape) * scale_factor).astype(int)

    x_in = np.linspace(0, input_width - 1, input_width)
    y_in = np.linspace(0, input_height - 1, input_height)
    X_in, Y_in = np.meshgrid(x_in, y_in)

    x_out = np.linspace(0, input_width - 1, output_shape[1])
    y_out = np.linspace(0, input_height - 1, output_shape[0])
    X_out, Y_out = np.meshgrid(x_out, y_out)

    scaled_lesion = interpolate_grid(X_in, Y_in, lesion_slice, X_out, Y_out, "linear")

    # ✅ NO zoom here (this was making sizes look too similar)
    # ✅ force a consistent ROI frame without changing lesion scale
    lesion_roi = center_crop_or_pad(scaled_lesion, roi_size_px)

    return lesion_roi


# ============================================================================
# Entry point
# ============================================================================


def main(files, config):
    """
    Dispatch to global noise or full CHO analysis based on config['full_test'].

    Parameters
    ----------
    files : list of pydicom Datasets
    config : dict with keys
        - series_uuid : str | None
        - full_test : bool
        - report_progress : bool
        - custom_parameters : dict
    """
    print(
        f"Starting CHO analysis - Full test: {config['full_test']}, "
        f"Progress tracking: {config['report_progress']}"
    )

    if config["full_test"]:
        return run_full_cho_analysis(files, config)
    else:
        return run_global_noise_analysis(files, config)


if __name__ == "__main__":
    import pydicom

    test_path = "W:\\Liver Segmentation\\input\\TOBIN, SARA\\2025-06-26-001\\IMAGES"
    slices = []
    skipcount = 0
    for f in os.listdir(test_path):
        dcm = pydicom.dcmread(os.path.join(test_path, f))
        if hasattr(dcm, "SliceLocation"):
            slices.append(dcm)
        else:
            skipcount += 1
    print(f"file count: {len(slices)}")
    print(f"skipped, no SliceLocation: {skipcount}")

    slices = sorted(slices, key=lambda s: s.SliceLocation)

    config = {
        "series_uuid": None,
        "full_test": True,
        "report_progress": False,
        "custom_parameters": {
            "channelType": "Gabor",
            "internalNoise": 2.25,
            "lesionSet": "standard",
            "resamples": 500,
            "resamplingMethod": "Bootstrap",
            "roiSize": 6,
            "stepSize": 5,
            "thresholdHigh": 300,
            "thresholdLow": 0,
            "windowLength": 15,
            "saveResults": False,
            "deleteAfterCompletion": False,
        },
    }
    main(slices, config)
