#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import os
from pydicom import dcmread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
from scipy import ndimage
from skimage import feature
from skimage import measure


# In[2]:

#  **********************************************************************#
# function used in patient specfic CHO d' calculation 
def average_filter(image, window_size=[3, 3], padding='constant'):

    """
    AVERAGEFILTER 2-D mean filtering.  
    Performs mean filtering of a 2-dimensional matrix/image using the integral image method.    

    Parameters:
    - image: 2D array-like, the input image to be filtered.
    - window_size: tuple, (M, N) defines the vertical and horizontal window size.
    - padding: str, can be 'constant', 'reflect', 'symmetric', or 'wrap'.   

    Returns:
    - image: 2D array, the filtered image.
    """
    
    if len(image.shape) != 2:
        raise ValueError("The input image must be a two-dimensional array.")
       
    # Set up the window size
    m, n = window_size    

    # Pad the image
    pad_width = ((m // 2, m // 2), (n // 2, n // 2))  # Calculate padding around the image

    if padding == 'circular':
        padded_image = np.pad(image, pad_width, mode='wrap')

    elif padding == 'replicate':
        padded_image = np.pad(image, pad_width, mode='edge')

    elif padding == 'symmetric':
        padded_image = np.pad(image, pad_width, mode='symmetric')

    else:  # Default is 'constant' padding (zero padding)
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    # Convert the image to float
    imageD = padded_image.astype(np.float64)
    

    # Calculate the integral image
    t = np.cumsum(np.cumsum(imageD, axis=0), axis=1)

    # Calculate mean values based on the integral image
    imageI = (t[m:, n:] 
              + t[:-m, :-n]
              - t[m:, :-n]
              - t[:-m, n:])

    # Normalize the imageI by the area of the window
    imageI /= (m * n)
    # Cast the resulting image back to the original type
    return imageI.astype(image.dtype)


def stdfilter(image, window_size=[3, 3], corrected=False, padding='constant'):
    """
    2-D standard deviation filtering.

    Parameters:
    - image: 2D array (input image).
    - window_size: Tuple defining the neighborhood size (M, N).
    - corrected: Boolean value deciding whether to use corrected variance estimate.
    - padding: Type of padding to use ('constant', 'reflect', 'symmetric', 'wrap').

    Returns:
    - deviation: 2D array of standard deviation values.

    """
   
    # Parameter checking
    if len(window_size) != 2:
        raise ValueError("Window_size must be a tuple of 2 values (M, N).")    
    
    m, n = window_size
  
    if len(image.shape) != 2:  # Check for color images
        raise ValueError("The input image must be a two-dimensional array.")   

    # Decide whether to use corrected variance estimate
    if corrected:
        normalization_factor = m * n / (m * n - 1)  # denominator is 'n-1'
    else:
        normalization_factor = 1  # denominator is 'n'

    
    # Convert image to float
    image = image.astype(float)    
    # Mean value
    mean = average_filter(image, window_size=window_size, padding=padding)    
    # Mean square
    mean_square = average_filter(image ** 2,  window_size=window_size, padding=padding)    
    # Standard deviation calculation
    deviation = np.sqrt(normalization_factor * (mean_square - mean ** 2))    

    return deviation


def round_to_nearest_even(num):
    return round(num + 0.5) if num % 2 != 0 else num



# ## Demo Main Program

# ### Load Images
# Display images and Get filepaths, num_of_repeated_scan
# Default I_start and I_end values are given.
# In[77]:
import time
start = time.time()
# users should specif the location of files for CHO d' calculation in "dir1"
dir1 ='\\\\mfad.mfroot.org\\rchapp\\eb028591\\CT_CIC_Group_Server\\Staff_Folders\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\L067_FD_1_0_B30F_0001\\'
scan_listing = os.listdir(dir1)
scan_listing = sorted (scan_listing)
n_images = len(scan_listing)
filepaths = []
for ii in range(len(scan_listing)): filepaths.append(os.path.join(dir1,scan_listing[ii]))

dcm_info = dcmread(filepaths[0])
RescaleIntercept = dcm_info.RescaleIntercept
RescaleSlope = dcm_info.RescaleSlope

SliceLocation1 = dcm_info.SliceLocation
dcm_info2 = dcmread(filepaths[1])
SliceLocation2 = dcm_info2.SliceLocation
Sliceinterval = abs(SliceLocation1-SliceLocation2)

Img0 = dcm_info.pixel_array
Im_Size = Img0.shape
rFOV = dcm_info.ReconstructionDiameter/10 # FOV in cm


cx0 = Img0.shape[0]
cy0 = Img0.shape[1]
del Img0

dx = rFOV/cx0
dy = rFOV/cy0
dz = 0

# Get the CTDIvol and dW
SliceLocation_end = 0

nx = dcm_info.Rows
ny = dcm_info.Columns
pixel_roi = dx*dy*100 # mm^2


ROI_size_circle = round (0.6/dx) # use 6mm ROI for Global noise index calculation
ROI_size = ROI_size_circle
Half_ROI_size = round(np.floor(ROI_size/2))

#initial setings for soft tissue
Thr1 = 0
Thr2 = 150

sample_interval = round(3/Sliceinterval) # sampling interval---Pick one image from every 3mm for calculating all the metrics
if Sliceinterval>=3: # when sliceinterval is larger than 3mm, use all images
    Step = 1

n_sel_images = (n_images + sample_interval - 1) // sample_interval # number of selected images

CTDI_All =  np.zeros(n_sel_images)
Dw = np.zeros(n_sel_images)  # water-equivalent diameter
STD_all = np.zeros([cx0,cy0,n_sel_images], dtype=np.float32)

idx = 0
for i in range(0, n_images, sample_interval):  # Step by "sample_interval" to read 1 image every "sample_interval"    
    dcm_info = dcmread(filepaths[i])
    im1 = dcm_info.pixel_array
    CTDI_All[idx] = dcm_info.CTDIvol
    im10 = im1*dcm_info.RescaleSlope+dcm_info.RescaleIntercept
    
    # for calculating Dw of each location along z-axis
    Mask_Dw_initial= im10 >= -260  # initial Mask for calculating water equivalent diameter (Dw)
    Mask_Dw = ndimage.binary_fill_holes(Mask_Dw_initial).astype(bool)  # Mask for Dw
    
    labeled_image, num_labels = measure.label(Mask_Dw, return_num=True)
    if num_labels > 0:
        largest_region = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1  # Find label of largest region
        binaryImage = (labeled_image == largest_region)  # Create binary image of largest region

    else:
        binaryImage = np.zeros_like(Mask_Dw, dtype=bool)

    pixel_No = np.sum(binaryImage)
    A_roi = pixel_No * pixel_roi
    im10_pat = im10[binaryImage]
    HU_Mean = np.mean(im10_pat)
    Dw[idx] = 2 * np.sqrt((HU_Mean / 1000 + 1) * A_roi / np.pi)
    # for calculating Dw of each location along z-axis

    ROI_size2 = round_to_nearest_even(ROI_size) # even value is needed for average and std filter
    Mean_Map = average_filter(im10,[ROI_size2,ROI_size2],'replicate')
    SD_Map = stdfilter(im10,[ROI_size2,ROI_size2],True,'replicate')
    Mask_mean = (Mean_Map < Thr1) | (Mean_Map > Thr2) 

    Edges = feature.canny(im10, sigma=5) # the results seems very different to matlab function Edges = edge(im10,'Canny');
    Mask_edge = Edges.copy()
    EdgesD =  np.array(Edges, dtype=np.float32)
    Edge_Impact = average_filter(EdgesD,[ROI_size2,ROI_size2],'replicate')

    Mask_edge[Edge_Impact >= 1 / ROI_size] = 1  # Set to 1 where condition is true
    Mask_edge[Edge_Impact < 1 / ROI_size] = 0   # Set to 0 where condition is true
    Mask = np.logical_or(np.logical_or(Mask_mean, Mask_edge), Edges)


    # Replace specified areas with NaN in SD_Map
    SD_Map[Mask] = np.nan
    SD_Map[0:Half_ROI_size, :] = np.nan  # Remove the border part (top)
    SD_Map[-Half_ROI_size:, :] = np.nan  # Remove the border part (bottom)
    SD_Map[:, 0:Half_ROI_size] = np.nan  # Remove the border part (left)
    SD_Map[:, -Half_ROI_size:] = np.nan  # Remove the border part (right)

    # Store results in 3D arrays
    STD_all[:, :, idx] = SD_Map
    idx += 1

    
dcm_info = dcmread(filepaths[n_images-1])
SliceLocation_end = dcm_info.SliceLocation

max_value = np.nanmax(STD_all) #find the maximum value while ignoring any NaN values.
# Flatten the array and create a histogram
h_Values, edges = np.histogram(STD_all.flatten(), bins=np.arange(0, max_value, 0.2))
# Find the maximum count and its corresponding bin
maxcount = np.max(h_Values)
whichbin_SD = np.argmax(h_Values)
# Get the bin edge corresponding to the maximum count = local noise level
Global_noise_level = edges[whichbin_SD]


# used for patient specific dose and image quality report table
# Mean_CTDI_All,  SSDE,  DLP_CTDIvol_L,    DLP_SSDE,   Mean_Dw,  NPS peakfrequency, MTF_10p? 
Mean_CTDI_All = np.mean(CTDI_All)
Mean_Dw = np.mean(Dw)/10 # in cm
para_a = 3.704369
para_b = 0.03671937
f = para_a * math.exp(-para_b * Mean_Dw)
SSDE = f * Mean_CTDI_All
Scan_len = (SliceLocation_end - SliceLocation1)/10 # in cm
DLP_CTDIvol_L = Scan_len * Mean_CTDI_All
DLP_SSDE = Scan_len * SSDE
MTF_10p = 7.30 # hard coded here, which is about the reconstruction kernel of the loaded patient images


    
end = time.time()
print(end - start)

# time: 20.12741994857788 seconds





