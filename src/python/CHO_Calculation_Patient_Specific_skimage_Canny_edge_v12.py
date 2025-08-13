#!/usr/bin/env python
# coding: utf-8

# # ## Demo Main Program with corresponding guidance is provided at the end of the codes
# In[1]:
import numpy as np
from numpy import random
from numpy.linalg import inv
import numpy.matlib
import os
from pydicom import dcmread
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.filters import threshold_multiotsu
# import cv2
import sys
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import math
from scipy import ndimage
import joblib
from skimage import feature
from skimage import measure

from scipy.interpolate import interp2d
import scipy.io as sio
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter, generic_filter, binary_dilation
from skimage.feature import peak_local_max

import math
import gc

def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

# In[2]:
def poly2fit(x,y,z,n):   
    # POLY2FIT	POLY2FIT(x,y,z,n) finds the coefficients of a two-dimensional
	#	polynomial formed from the data in the matrices x and y
	#	that fits the data in the matrix z in a least-squares sense.
	#	The coefficients are returned in the vector p, in descending
	#	powers of x and y.  For example, the second order polynomial
	#	x^2+xy+2x-3y-1
	#	would be returned as [1 1 0 2 -3 -1]

    # transfer from Matlab codes
	#Written by Jim Rees, Aug.12, 1991
	#Adapted by Jeff Siewerdsen
    if x.shape[0]!=y.shape[0] or x.shape[1]!=y.shape[1] or x.shape[0]!=z.shape[0] or x.shape[1]!=z.shape[1]:
        print('X,Y, and Z matrices must be the same size')
    
    x = np.transpose(x)
    y = np.transpose(y)
    z = np.transpose(z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    n=n+1
    k=0
    A=np.zeros((x.shape[0],6)) # for 2D fitting, totally 6 parameters P[1] to P[6]
    i = n
    while i >= 1:        
        for j in range(1, i+1):
            temp1 = np.power(x, i-j)
            temp2 = np.power(y, j-1)
            A[:,k] = np.multiply(temp1, temp2)
            k=k+1
        i = i - 1

    p = np.linalg.lstsq(A,z)[0]

    return p


def subtractMean2D(im,method,psize):   
    if method: # default (method = 1) -> "polynomial fitting"
        FOV = np.zeros(2)
        im_sizeX = im.shape[0]
        im_sizeY = im.shape[1]
        FOV[0] = psize[0] * im_sizeX
        FOV[1] = psize[1] * im_sizeY 

        x=np.arange(0, im_sizeX, 1)*psize[0] - FOV[0]/2
        y=np.arange(0, im_sizeY, 1)*psize[1] - FOV[1]/2
        X, Y = np.meshgrid(x,y)
        P = poly2fit(X,Y,im,1); #Get the 2D first order fit

        im = im-(P[0]*X + P[1]*Y + P[2])

    else: #(method = 0) -> "subtract the mean"
        im = im - np.mean(im.flatten()) 

    return im

def NPS_statistics(nps1d, unit):
    peakfrequencyIndex = np.where(nps1d == np.max(nps1d))[0][0]
    peakfrequency = peakfrequencyIndex * unit

    #Get the average frequency from 1D radial NPS
    n = len(nps1d)
    fre_idx = list(range(0,n))
    Spatial_freq = [x*unit for x in fre_idx]
    p = nps1d/sum(np.array(nps1d)) #turn 1D nps into probability distribution
    p = np.squeeze(p)
    Spatial_freq = np.array(Spatial_freq)
    Spatial_freq = np.squeeze(Spatial_freq)
    fav = sum(np.multiply(Spatial_freq,p))
              
    minfrequencyIndex = np.where(nps1d < 0.10 * np.max(nps1d))[0]

    freq = np.where(minfrequencyIndex > peakfrequencyIndex)[0][0]
    min10percent_frequency = 0
    if freq >= 0:
        min10percent_frequency = minfrequencyIndex[freq] * unit

    npoint = 10
    k = np.polyfit(np.arange(npoint) * unit, nps1d[:npoint], 1)

    return fav, peakfrequency, k, min10percent_frequency



def ROI_to_NPS_Sum(ROI_size, ROI_All, dx, dy):

    # Pre-calculate padding size
    if ROI_size < 20:
        nnn = 8 * 20

    else:
        nnn = 8 * ROI_size

    # Determine number of radial samples
    cc = int(np.rint(nnn / 2))
    unit = 1 / (dx * nnn)

    # Precompute polar angles vector (360 angles)
    oo = np.linspace(0, 2 * np.pi, 360, endpoint=False)  # shape (360,)

    # Precompute radial indices for all radii (0 to cc-1) and all angles
    # Create a column vector for radii: shape (cc, 1)
    radii = np.arange(cc).reshape((cc, 1))

    # Compute x and y offsets for each radius and angle (broadcasting over angles)
    x_offset = radii * np.cos(oo)  # shape (cc, 360)
    y_offset = radii * np.sin(oo)  # shape (cc, 360)

    # Shift indices to mapping coordinates and round to nearest integer
    xi_all = np.rint(x_offset + cc - 1).astype(np.int32)  # shape (cc, 360)
    yi_all = np.rint(y_offset + cc - 1).astype(np.int32)  # shape (cc, 360)

    # Precompute mask of valid indices (within bounds of nnn x nnn)
    valid_mask = (xi_all >= 0) & (xi_all < nnn) & (yi_all >= 0) & (yi_all < nnn)

    # Initialize accumulators
    NPS_1D_sum = np.zeros((cc, 1), dtype=np.float32)
    noise_level_sum = 0.0

    # Iterate over all slices in ROI_All along the third axis
    num_slices = ROI_All.shape[-1]
    for jj in range(num_slices):
        # Extract ROI slice and subtract the mean using provided subtraction function
        roi = ROI_All[:,:,jj]
        psize = (dx, dy)
        roi = subtractMean2D(roi, 1, psize)  # default method (polynomial fitting)
        noise_level_sum += np.std(roi)

        # Zero padding: insert ROI into the center of an array of shape (nnn, nnn)
        roipad = np.zeros((nnn, nnn), dtype=np.float32)
        half_roi = int(np.ceil(ROI_size / 2))
        start_index = int(np.rint(nnn / 2 - half_roi))
        end_index = int(np.rint(nnn / 2 + half_roi))
        roipad[start_index:end_index-1, start_index:end_index-1] = roi

        # Compute Fourier transform and shift, then scale the squared magnitude
        roipad_fft = np.fft.fftshift(np.abs(np.fft.fft2(roipad)))  # 2D FFT and shift
        roipad_fft = roipad_fft**2 * dx * dy / (ROI_size * ROI_size)
        nps2D = roipad_fft

        # Allocate polar1 array and fill using precomputed indices with valid_mask
        polar1 = np.zeros((cc, 360), dtype=np.float32)
        # Use valid_mask to assign values from nps2D, vectorized over whole (cc,360)
        polar1[valid_mask] = nps2D[xi_all[valid_mask], yi_all[valid_mask]]

        # Compute radial average (mean over the angle axis)
        nps1d = np.mean(polar1, axis=1).reshape((cc, 1))

        # Sum over slices
        NPS_1D_sum += nps1d

    # Compute Spatial frequency axis
    Spatial_freq = np.arange(cc).astype(np.float32) * unit

    return Spatial_freq, NPS_1D_sum, noise_level_sum, unit
# In[3]:
def Laguerre2D(order,a,b,cx,cy,X,Y):
    val1 = np.zeros(X.size)
    ga = 2*np.pi*((X-cx)**2/(a**2)+(Y-cy)**2/(b**2));
    for jp in range(order+1):
        val1 = val1 + (-1)**jp * np.prod(np.linspace(1,order,order))/( np.prod(np.linspace(1,jp,jp)) * np.prod(np.linspace(1,order-jp,order-jp)) ) * (ga**jp)/np.prod(np.linspace(1,jp,jp))
    channel_filter = np.exp(-ga/2)*val1
    
    return channel_filter


# In[4]:
def Gabor2D(fc, wd, theta, beta, cx, cy, X, Y):
    channel_filter = np.exp(-4*np.log(2)*((X-cx)**2+(Y-cy)**2)/(wd**2)) * np.cos(2*np.pi*fc*( (X-cx)*np.cos(theta)+(Y-cy)*np.sin(theta) )+beta);
    
    return channel_filter


# In[5]:
def channel_selection(Chnl, inputArg1, inputArg2, inputArg3='0'):
     
    if Chnl.Chnl_Toggle == 'Laguerre-Gauss':
        if isinstance(inputArg1, int): Chnl.LG_order = inputArg1
        if isinstance(inputArg2, int): Chnl.LG_orien = inputArg2

    if Chnl.Chnl_Toggle == 'Gabor':
        if isinstance(inputArg1, str): # 5 options for channel dropdown
            if inputArg1 == "[[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]]":
                Chnl.Gabor_passband = np.transpose([[1/64,1/32], [1/32,1/16], [1/16,1/8], [1/8,1/4]])
            if inputArg1 == "[[1/64,1/32], [1/32,1/16]]":
                Chnl.Gabor_passband = np.transpose([[1/64,1/32], [1/32,1/16]])
                
        if isinstance(inputArg2, str): # 3 options for orientations           
            if inputArg2 == '[0, pi/3, 2*pi/3]':  Chnl.Gabor_theta = [0, np.pi/3, 2*np.pi/3]
            if inputArg2 == '[0, pi/2]': Chnl.Gabor_theta = [0, np.pi/2]
            if inputArg2 == '0': Chnl.Gabor_theta = [0]
                
        if isinstance(inputArg3, str):
            if inputArg3 == '[0, pi/2]': Chnl.Gabor_beta = [0, np.pi/2]
            if inputArg3 == '0': Chnl.Gabor_beta = [0]
        
        Chnl.Gabor_fc = np.mean(Chnl.Gabor_passband,axis=0)
        Chnl.Gabor_wd = 4*np.log(2)/(np.pi*(Chnl.Gabor_passband[1,:]-Chnl.Gabor_passband[0,:]))
            
    return Chnl


# In[6]:
def ChannelMatrix_Generation(Chnl, roiSize_xy):
    
    x = np.linspace(1, roiSize_xy, roiSize_xy)-(roiSize_xy+1)/2; y = x;
    X, Y = np.meshgrid(x,y); X = X.T.reshape(-1); Y = Y.T.reshape(-1)
    channelMatrix = None
    if Chnl.Chnl_Toggle == 'Laguerre-Gauss': 
        # generate LG CHO channel matrix
        LG_ORIEN,LG_ORDER = np.meshgrid(np.arange(Chnl.LG_orien)+1,np.arange(Chnl.LG_order)+1);
        LG_ORIEN = LG_ORIEN.T.reshape(-1); LG_ORDER = LG_ORDER.T.reshape(-1)
        A = 5*np.ones(LG_ORDER.size); B = 14*np.ones(LG_ORDER.size);
        A[LG_ORIEN==1] = 8; B[LG_ORIEN==1] = 8; A[LG_ORIEN==2] = 14; B[LG_ORIEN==2] = 5;
        channelMatrix = np.zeros((roiSize_xy*roiSize_xy, Chnl.LG_order*Chnl.LG_orien));
        for ii in range(LG_ORDER.size):
            channelMatrix[:,ii] = Laguerre2D(LG_ORDER[ii], A[ii], B[ii], 0,0, X,Y)
    
    elif Chnl.Chnl_Toggle == 'Gabor':
        # Generate Gabor CHO channel matrix
        Gabor_THETA,Gabor_FC,Gabor_BETA = np.meshgrid(Chnl.Gabor_theta,Chnl.Gabor_fc,Chnl.Gabor_beta); 
        Gabor_wd_matrix = np.zeros((Chnl.Gabor_wd.size,1,1)); Gabor_wd_matrix[:,0,0] = Chnl.Gabor_wd
        Gabor_WD = np.tile(Gabor_wd_matrix,(1,Gabor_FC.shape[1],Gabor_FC.shape[2]));
        
        Gabor_FC = Gabor_FC.T.reshape(-1); Gabor_WD = Gabor_WD.T.reshape(-1)
        Gabor_THETA = Gabor_THETA.T.reshape(-1); Gabor_BETA = Gabor_BETA.T.reshape(-1)
        
        channelMatrix = np.zeros((roiSize_xy*roiSize_xy, Gabor_FC.size));
        for ii in range(Gabor_FC.size):
            channelMatrix[:,ii] = Gabor2D(Gabor_FC[ii], Gabor_WD[ii], Gabor_THETA[ii], Gabor_BETA[ii], 0,0, X,Y);
    
    return channelMatrix


# In[7]:
def CHO_patient_with_resampling(sig_true,bkg_ordered,channelMatrix,internalNoise,Resampling_method):
    
    N_total_bkg  = bkg_ordered.shape[1]   
    rand_scanSelect_bkg = 0
    # generate random numbers to select scans for CHO
    if Resampling_method == 'Bootstrap': # Resampling with replacement
        rand_scanSelect_bkg = np.random.randint(N_total_bkg, size = N_total_bkg)
    elif Resampling_method == 'Shuffle': # Resampling without replacement
        rand_scanSelect_bkg = np.random.permutation(N_total_bkg)

    # Reorder data as indicated by rand_scanSelect
    sig = sig_true 
    bkg = bkg_ordered[:,rand_scanSelect_bkg].astype(float)

    if sig.shape[0]!=channelMatrix.shape[0] or bkg.shape[0]!=channelMatrix.shape[0]: 
        print('Numbers of pixels do not match.')
      

    # Compute channel outputs only from the noise bkg
    #vS = channelMatrix.T @ sig; 
    vN = channelMatrix.T @ bkg 
    # directly use the mean signal
    sbar = np.squeeze(sig)  
    # Intra-class scatter matrix
    S = np.cov(vN.T,rowvar=False) 
    # channel template
    wCh = np.linalg.inv(S) @ channelMatrix.T @ sbar
    # apply the channel template to produce outputs
    temp = channelMatrix.T @ sbar
    tsN_Mean = wCh.T @ temp
    tN0 = wCh.T @ vN
    ## add internal noise
    tN = tN0 + np.random.randn(tN0.size) * internalNoise * np.std(tN0)
 
    #Compute d' (SNR) for decision variables tS and tN
    dp = np.sqrt( (tsN_Mean**2) / (np.var(tN)) ) 

    return dp


def interpolate_grid(X0, Y0, L0, XX, YY, method):

    # Convert input arrays to numpy arrays if they are not already
    X0 = np.asarray(X0)
    Y0 = np.asarray(Y0)
    L0 = np.asarray(L0)
    XX = np.asarray(XX)
    YY = np.asarray(YY)

    # Create a meshgrid from the XX and YY coordinates
    grid_points = (X0.flatten(), Y0.flatten())
    values = L0.flatten()

    # Perform grid data interpolation
    Lesion = griddata(grid_points, values, (XX, YY), method=method)

    return Lesion

def max_consecutive_ones_2d(bool_array_2d):

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

def prepare_Lesion_sig(lesion_file, psf_file, lesion_con_target,target_lesion_width,patient_FOV,patient_matrix,ROI_size):
  
    # Load the lesion data from MATLAB file
    patient_data = sio.loadmat(lesion_file)
    lesion = patient_data['Patient']['Lesion'][0][0]['VOI'][0][0]
    mask = patient_data['Patient']['Lesion'][0][0]['LesionMask'][0][0]  


    loc = round(lesion.shape[-1]/2)
    L0 = lesion[:, :, loc]  # Select center slice of the lesion
    M0 = mask[:, :, loc]  # Select same slice
    M0 = M0.astype(bool)
    lesion_pixels_width_input = max_consecutive_ones_2d(M0) #24  # Input lesion pixel width-- determined by the original lesion size  # to replace
    lesion_HU = np.mean(L0[M0]) #39.8786  # Original mean HU of the Lesion  


    Diff_HU = lesion_HU - lesion_con_target

    L0 = L0 - Diff_HU    # change the lesion contrast to be the target contrast [-10,-30,-50];
    L0[~M0.astype(bool)] = 0 # Update background in lesion image

    # Define parameters and scaling
    lesion_pixels_width_target = target_lesion_width / (patient_FOV / patient_matrix)  
    scale = lesion_pixels_width_target / lesion_pixels_width_input  # Upscaling

    
    # Compute output size
    s1, s2 = L0.shape
    os0 = np.floor(np.array(L0.shape) * scale).astype(int)

    # Create grid for interpolation
    x0 = np.linspace(0, s2-1, s2)
    y0 = np.linspace(0, s1-1, s1)
    X0, Y0 = np.meshgrid(x0, y0)

    # Output coordinate space for interpolation
    x_out = np.linspace(0, s2-1, os0[1])
    y_out = np.linspace(0, s1-1, os0[0])
    XX, YY = np.meshgrid(x_out, y_out)

    # Perform interpolation
    Lesion = interpolate_grid(X0, Y0, L0, XX, YY, 'linear')


    # Load the PSF from MATLAB file
    psf_data = sio.loadmat(psf_file)
    PSF = psf_data['PSF']  # Assuming PSF is contained within the loaded dict    

    wire_FOV = 50 
    scaling_factor = wire_FOV / patient_FOV / 4  # downscaling, the original loaded PSF is 4 times oversampled
    os0_psf = np.floor(np.array(PSF.shape) * scaling_factor).astype(int)

    # Interpolation for PSF
    PSF_x0 = np.linspace(0, PSF.shape[1]-1, PSF.shape[1])
    PSF_y0 = np.linspace(0, PSF.shape[0]-1, PSF.shape[0])
    PSF_X0, PSF_Y0 = np.meshgrid(PSF_x0, PSF_y0)

    PSF_x_out = np.linspace(0, PSF.shape[1]-1, os0_psf[1])
    PSF_y_out = np.linspace(0, PSF.shape[0]-1, os0_psf[0])
    PSF_XX, PSF_YY = np.meshgrid(PSF_x_out, PSF_y_out)

    # Interpolate PSF, but there is only 2*2 size?-> upsamling then convolution and then downsampling, similar results?  to be investigated???
    PSF_end = interpolate_grid(PSF_X0, PSF_Y0, PSF, PSF_XX, PSF_YY, 'linear')
    PSF_end /= PSF_end.sum()  # Normalize PSF

    ## Initialize the extended matrix other size of 3mm lesion Matrix is only 16*16
    Lesion_ext = np.zeros((80, 80)) 

    # Calculate the starting index to get the center part
    start_row = (Lesion_ext.shape[0] - Lesion.shape[0]) // 2  # Center row index for ROI_size rows
    start_col = (Lesion_ext.shape[1] - Lesion.shape[1]) // 2  # Center column index for ROI_size columns

    Lesion_ext[start_row:start_row + Lesion.shape[0], start_col:start_col + Lesion.shape[1]] = Lesion

    # Convolve lesion with the PSF
    from scipy.signal import convolve2d
    Lesion_conv = convolve2d(Lesion_ext, PSF_end, mode='full')

    
    # Calculate the starting index to get the center part
    start_row = (Lesion_conv.shape[0] - ROI_size) // 2  # Center row index for ROI_size rows
    start_col = (Lesion_conv.shape[1] - ROI_size) // 2  # Center column index for ROI_size columns


    # Extract the center ROI_sizexROI_size part
    Lesion_conv_end = Lesion_conv[start_row:start_row + ROI_size, start_col:start_col + ROI_size]

    return Lesion_conv_end

# get the intergral image
def Integral_image(image, window_size=[3, 3], padding='constant'):

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

    # Cast the resulting image back to the original type
    return t


def calculate_std_dev(Intel_images, Intel_images_Square, Intel_Edge, ROI_size, Padding_size, Im_Size, N1, N2, N_sub, Thr1, Thr2, Edges, corrected=False):
    """
    Calculates the standard deviation (STD) for a set of images based on integral images and edge impact.
    Args:
        Intel_images (np.ndarray): Integral images.
        Intel_images_Square (np.ndarray): Integral images of the squared images.
        Intel_Edge (np.ndarray): Integral images of the edge information.
        ROI_size (int): Size of the region of interest.
        Padding_size (int): Size of the padding.
        Im_Size (tuple): Size of the original images (height, width).
        N1 (int): Index start.
        N2 (int): Index end.
        N_sub (int): Index sub.
        Thr1 (float): Lower threshold for the mean.
        Thr2 (float): Upper threshold for the mean.
        Edges (np.ndarray): Edge mask.
        corrected (bool, optional): Flag to use a corrected normalization factor. Defaults to False.

    Returns:
        np.ndarray: A 3D numpy array containing the standard deviation maps for each image channel.

    """
    ROI_size2 = int(ROI_size) if ROI_size % 2 == 0 else int(ROI_size + 1)  # Ensure even
    Half_Diff_Size = round((Padding_size - ROI_size2) / 2)

    Intel_Im_Sizex = Intel_Edge.shape[0]
    Intel_Im_Sizey = Intel_Edge.shape[1]

    Intel_Edge = Intel_Edge[Half_Diff_Size:Intel_Im_Sizex-Half_Diff_Size, Half_Diff_Size:Intel_Im_Sizey-Half_Diff_Size, :]
    Intel_images = Intel_images[Half_Diff_Size:Intel_Im_Sizex-Half_Diff_Size, Half_Diff_Size:Intel_Im_Sizey-Half_Diff_Size, :]
    Intel_images_Square = Intel_images_Square[Half_Diff_Size:Intel_Im_Sizex-Half_Diff_Size, Half_Diff_Size:Intel_Im_Sizey-Half_Diff_Size, :]

    m = ROI_size2
    n = ROI_size2
    Half_ROI_size = round(np.floor(ROI_size / 2))

    if corrected:
        normalization_factor = m * n / (m * n - 1)  # denominator is 'n-1'
    else:
        normalization_factor = 1  # denominator is 'n'

    STD_all = np.zeros([Im_Size[0], Im_Size[1], N2 + N_sub * 2 - N1], dtype=np.float32)
    Mask_edge = np.zeros_like(Edges, dtype=np.float32)

    for kk in range(Intel_images.shape[-1]):
        t = Intel_images[:, :, kk]
        mean_map = (t[m:, n:]
                    + t[:-m, :-n]
                    - t[m:, :-n]
                    - t[:-m, n:])

        t2 = Intel_images_Square[:, :, kk]
        mean_square = (t2[m:, n:]
                       + t2[:-m, :-n]
                       - t2[m:, :-n]
                       - t2[:-m, n:])

        t3 = Intel_Edge[:, :, kk]
        edge_impact = (t3[m:, n:]
                       + t3[:-m, :-n]
                       - t3[m:, :-n]
                       - t3[:-m, n:])

        # Normalize the mean/mean_square by the area of the window
        mean_map /= (m * n)
        mean_square /= (m * n)
        edge_impact /= (m * n)

        Mask_mean = (mean_map < Thr1) | (mean_map > Thr2)
        Mask_edge[edge_impact >= 1 / ROI_size] = 1  # Set to 1 where condition is true
        Mask_edge[edge_impact < 1 / ROI_size] = 0  # Set to 0 where condition is true
        Mask = np.logical_or(np.logical_or(Mask_mean, Mask_edge), Edges)

        # Replace specified areas with NaN in SD_Map
        SD_Map = np.sqrt(normalization_factor * (mean_square - mean_map ** 2))
        SD_Map[Mask] = np.nan
        SD_Map[0:Half_ROI_size, :] = np.nan  
        SD_Map[-Half_ROI_size:, :] = np.nan  
        SD_Map[:, 0:Half_ROI_size] = np.nan  
        SD_Map[:, -Half_ROI_size:] = np.nan  

        STD_all[:, :, kk] = SD_Map

    return STD_all

def extract_ROIs(STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size):

    ROIs = []
    # Loop over each slice in the last dimension of STD_map_all
    for ii in range(STD_map_all.shape[-1]):  # every 10cm or 15cm, moving 5cm each time
        im10 = Images_Section[:, :, ii]
        SD_Map = STD_map_all[:, :, ii]

        # Vectorized candidate extraction
        valid_mask = (SD_Map < Thre_SD) & (~np.isnan(SD_Map))
        rows, cols = np.where(valid_mask)

        # Continue to next iteration if no valid points are found
        if rows.size == 0:
            continue

        # Sort candidate points by their SD values
        std_vals = SD_Map[rows, cols]
        sorted_indices = np.argsort(std_vals)
        sorted_rows = rows[sorted_indices]
        sorted_cols = cols[sorted_indices]

        # Create a mask to mark regions already blocked by a selected ROI
        selection_mask = np.zeros_like(SD_Map, dtype=bool)
        selected_centers = []

        # Iterate through sorted candidate center points
        for r, c in zip(sorted_rows, sorted_cols):
            # If the candidate point is already blocked, skip it.
            if selection_mask[r, c]:
                continue
            selected_centers.append((r, c))

            # Mark the ROI region as blocked using binary dilation concept:
            # Clip boundaries if needed.
            r_start = max(r - Half_ROI_size, 0)
            r_end = min(r + Half_ROI_size + 1, selection_mask.shape[0])
            c_start = max(c - Half_ROI_size, 0)
            c_end = min(c + Half_ROI_size + 1, selection_mask.shape[1])
            selection_mask[r_start:r_end, c_start:c_end] = True

        # Extract ROIs for selected centers if the boundaries are valid
        for (row, col) in selected_centers:
            if (row - Half_ROI_size < 0 or 
                row + Half_ROI_size + 1 > Im_Size[0] or 
                col - Half_ROI_size < 0 or 
                col + Half_ROI_size + 1 > Im_Size[1]):
                continue

            roi = im10[row - Half_ROI_size: row + Half_ROI_size + 1,
                       col - Half_ROI_size: col + Half_ROI_size + 1]

            ROIs.append(roi)
    ROIs_array = np.array(ROIs)
    
    Total_NPS_No = 0
    if ROIs_array.size:
        ROIs_array = np.transpose(ROIs_array, (1, 2, 0))
        Total_NPS_No = ROIs_array.shape[-1]

    return ROIs_array, Total_NPS_No



#**************** Demo Main Program*****************************
# In[77]:
import time
start = time.time()

# prepare the lesion signal
lesion_file = "\\\\mfad.mfroot.org\\rchapp\\eb028591\\CT_CIC_Group_Server\\Staff_Folders\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\Liver_lesion_sample\\Patient02-411-920_Lesion1.mat"
psf_file = '\\\\mfad.mfroot.org\\rchapp\\eb028591\\CT_CIC_Group_Server\\Staff_Folders\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\Liver_lesion_sample\\EID_PSF_Br44.mat'
# loop all the lesion conditons
Lesion_Contrasts = [-30,-30,-10,-30,-50] # [-30] #
Lesion_Size = [3,9,6,6,6]
ROI_sizes = [21,29,25,25,25] # pixel size 360/512:  21 for 3mm lesion d' calculation;   25 for 6mm lesion d' calculation;  29 for 9mm lesion d' calculation;--for big pixels, still need enough number of pixels, otherwise not enough backgound part
patient_FOV = 340 # from original lesion dicom file
patient_matrix = 512  # from original lesion dicom file
Lesion_sigs = []  # Initialize a list to store Lesion_sig for each contrast
for i in range(len(Lesion_Contrasts)):
    lesion_con_target = Lesion_Contrasts[i]  # Get the current contrast target
    target_lesion_width = Lesion_Size[i]
    ROI_size = ROI_sizes[i]
    Lesion_sig = prepare_Lesion_sig(lesion_file, psf_file, lesion_con_target, target_lesion_width, patient_FOV, patient_matrix, ROI_size)
    Lesion_sigs.append(Lesion_sig)  # Append the result to the list

# users should specif the location of files for CHO d' calculation in "dir1"
# dir1 = '\\\\mfad.mfroot.org\\rchapp\\eb017185\\eb017185\\Grand_Challenge\\10_Training_Cases_RD\\L067_Mayo\\FD_1_0_B30F_0001\\'
# dir1 ='\\\\mfad.mfroot.org\\rchapp\\eb028591\\CT_CIC_Group_Server\\Staff_Folders\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\L067_FD_1_0_B30F_0001\\'
# dir1 = '\\\\mfad.mfroot.org\\rchapp\\eb028936\\eb028936\\_Projects\\_PCD_Abdomen_Noise_Insertion\\For_review\\Pineda\\Z_Others\\QIR3_Br44_FD\\' #FBP reconstruction has no CTDIvol information?!!
dir1 = "Y:\\Patient_data\\Alpha\\Radiomics\\healthy\\BARTZ_LYDIA_L\\DICOMS\\100%\\3\\Br44f\\1x1\\vmi_67"
# dir1 = "V:\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\L067_FD_1_0_B30F_0001"

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
CTDI_All =  np.zeros(n_images)
Dw = np.zeros(n_images)  # water-equivalent diameter
nx = dcm_info.Rows
ny = dcm_info.Columns
pixel_roi = dx*dy*100 # mm^2


for i in range(n_images): 
        dcm_info = dcmread(filepaths[i])
        CTDI_All[i] = dcm_info.get('CTDIvol', -1)
        im1 = dcm_info.pixel_array
        im10 = im1 * dcm_info.RescaleSlope + dcm_info.RescaleIntercept
        Mask = im10 >= -260  # Combined masking operation ((70))
        Mask22 = ndimage.binary_fill_holes(Mask).astype(bool)        

        labeled_image, num_labels = measure.label(Mask22, return_num=True)
        if num_labels > 0:
            largest_region = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1  # Find label of largest region
            binaryImage = (labeled_image == largest_region)  # Create binary image of largest region

        else:
            binaryImage = np.zeros_like(Mask22, dtype=bool)

        pixel_No = np.sum(binaryImage)
        A_roi = pixel_No * pixel_roi
        im10_pat = im10[binaryImage]
        HU_Mean = np.mean(im10_pat)
        Dw[i] = 2 * np.sqrt((HU_Mean / 1000 + 1) * A_roi / np.pi)
        if i==n_images-1:
            SliceLocation_end = dcm_info.SliceLocation


# used for patient specific dose and image quality report table
# Mean_CTDI_All,  SSDE,  DLP_CTDIvol_L,    DLP_SSDE,   Mean_Dw,  NPS peakfrequency, MTF_10p, Mean_All_dps
Mean_CTDI_All = np.mean(CTDI_All)
Mean_Dw = np.mean(Dw)/10 # in cm
para_a = 3.704369
para_b = 0.03671937
f = para_a * math.exp(-para_b * Mean_Dw)
SSDE = f * Mean_CTDI_All
Scan_len = (SliceLocation_end - SliceLocation1)/10 # in cm
DLP_CTDIvol_L = Scan_len * Mean_CTDI_All
DLP_SSDE = Scan_len * SSDE
# peakfrequency #see Line 924
MTF_10p = 7.30 # hard coded here, which needed to be obtained from Database of CTPro
# Mean_All_dps see Line 903



ROI_size_circle_N = round (0.6/dx) # use 6mm ROI for noise levels calculation
ROI_size_N = ROI_size_circle_N
Half_ROI_size_N = round(np.floor(ROI_size_N/2))

#initial setings for soft tissue
Thr1 = 0
Thr2 = 150

# CHO parameters
numResample = 500;
internalNoise = 2.25;

# Resampling_method = 'Shuffle';
Resampling_method = 'Bootstrap'; 
Validation_method = 'Resubstitution'; 

## CHO channels: 
class Chnl: Chnl_Toggle = 'Gabor'; # Default value    
Chnl_Toggle = 'Gabor'; # or 'Laguerre-Gauss' 
Chnl.Chnl_Toggle = Chnl_Toggle

if Chnl.Chnl_Toggle == 'Gabor': 
    Gabor_passband = '[[1/64,1/32], [1/32,1/16]]'
    Gabor_theta = '[0, pi/2]'
    Gabor_beta = '0'
    Chnl = channel_selection(Chnl, Gabor_passband, Gabor_theta, Gabor_beta)
if Chnl.Chnl_Toggle == 'Laguerre-Gauss':
    LG_order = 6 
    LG_orien = 3
    Chnl = channel_selection(Chnl, LG_order,LG_orien)

N_sub = round(50/Sliceinterval) #every 5 cm
All_dps = np.zeros([n_images // N_sub -2,5])
Mean_All_dps = np.zeros([n_images // N_sub -2,1])
Noise_level_local = np.zeros([n_images // N_sub -2,1])
Total_NPS_Num = 0
NPS_1D_sum  = 0
Spatial_freq = []
noise_level_sum = 0
NPS_Cal = True # calculate NPS only for 1 time

Padding_size = 2* round (1.2/dx) # for obtaining the integral images, twice larger than the ROI size for the biggest lesion. e.g., 12mm leision
Padding_size = int(Padding_size) if Padding_size % 2 == 0 else int(Padding_size + 1)  # ensure even

for mm in range(1, n_images // N_sub -1):  # need to meet the requirement of N2 + N_sub * 2 < n_images # every 10cm or 15cm, moving 5cm each time                   
    N1 =(mm-1)*N_sub # every 5cm
    N2 = mm*N_sub #n_images 

    # 3 integral images will be used for all calculations 
    Intel_images = np.zeros([Im_Size[0]+Padding_size,Im_Size[1]+Padding_size,N2 + N_sub * 2-N1], dtype=np.float32) # integral images of the orignal images  
    Intel_images_Square = np.zeros([Im_Size[0]+Padding_size,Im_Size[1]+Padding_size,N2 + N_sub * 2-N1], dtype=np.float32) # integral images of the images**2
    Intel_Edge = np.zeros([Im_Size[0]+Padding_size,Im_Size[1]+Padding_size,N2 + N_sub * 2-N1], dtype=np.float32) # integral images of the cannon edge images
    Images_Section = np.zeros([Im_Size[0],Im_Size[1],N2 + N_sub * 2-N1], dtype=np.float32) # original images
  
    for nn in range(N1, N2 + N_sub * 2): # every 10cm or 15cm, moving 5cm each time
        dcm_info = dcmread(filepaths[nn])                
        print(f"idx: {nn-N1}")
        print(f'Slice Location: {(dcm_info.SliceLocation - SliceLocation1) / 10}')

        im1 = dcm_info.pixel_array
        im10 = im1 * dcm_info.RescaleSlope + dcm_info.RescaleIntercept
        Edges = feature.canny(im10, sigma=5) # the results seems very different to matlab function Edges = edge(im10,'Canny');
        Images_Section[:,:,nn-N1] = im10
        Intel_Edge[:,:,nn-N1] = Integral_image(Edges, [Padding_size, Padding_size], 'replicate')
        Intel_images[:,:,nn-N1] = Integral_image(im10, [Padding_size, Padding_size], 'replicate')
        Intel_images_Square[:,:,nn-N1] = Integral_image(im10** 2, [Padding_size, Padding_size], 'replicate')


    # ****************************************to get the  2 sigma point on the right half of histogram ****************************************
    STD_all = calculate_std_dev(Intel_images, Intel_images_Square, Intel_Edge, ROI_size_N, Padding_size, Im_Size, N1, N2, N_sub, Thr1, Thr2, Edges, corrected=False)

    max_value = np.nanmax(STD_all) #find the maximum value while ignoring any NaN values.
    # Flatten the array and create a histogram
    h_Values, edges = np.histogram(STD_all.flatten(), bins=np.arange(0, max_value, 0.2))
    # Find the maximum count and its corresponding bin
    maxcount = np.max(h_Values)
    whichbin_SD = np.argmax(h_Values)
    # Get the bin edge corresponding to the maximum count = local noise level
    bin_edge = edges[whichbin_SD]
    Noise_level_local[mm-1] = bin_edge 

   
    if NPS_Cal:
          Half_ROI_size_N = round(np.floor(ROI_size_N/2))
          ROI_All_NPS, Total_NPS_No = extract_ROIs(STD_all, bin_edge, Half_ROI_size_N, Images_Section, Im_Size)  
          Spatial_freq, NPS_1D_sum, noise_level_sum, unit = ROI_to_NPS_Sum(ROI_size_N, ROI_All_NPS[:,:,0:200], dx, dy) # only pick 200 ROIs for NPS calculation????  
          Total_NPS_Num =  200
          NPS_Cal = False  
         
    # Clearing temporary arrays
    del STD_all
    gc.collect()

    # find the 2 sigma point in the left side of histogram, then get the gap between this point and peak value.
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
    # Find "closest" values array wrt. target value.
    closest = Ratios[np.argmin(temp_dis)]
    loc = np.where(Ratios == closest)[0][0]
    gap = whichbin_SD - loc
    #include the range to the 2 sigma point on the right half of histogram by adding gap to peak value location
    Thre_SD = edges[whichbin_SD+gap] 
    # ****************************************to get the  2 sigma point on the right half of histogram ****************************************

    all_dps = []

    for ii in range(len(Lesion_sigs)):
        lesion_sig = Lesion_sigs[ii]
        ROI_size = ROI_sizes[ii]
        lesion_con_target = Lesion_Contrasts[ii]
        Half_ROI_size = round(np.floor(ROI_size/2))
        if ii in [3, 4]:
            print("use same ROI_All")  # Skip to avoid redundant calls
        else:
            STD_map_all = calculate_std_dev(Intel_images, Intel_images_Square, Intel_Edge, ROI_size, Padding_size, Im_Size, N1, N2, N_sub, Thr1, Thr2, Edges, corrected=False)           
            ROI_All, Total_NPS_No = extract_ROIs(STD_map_all, Thre_SD, Half_ROI_size, Images_Section, Im_Size)


        Bkg_HU = 0 # make the background ROI to be 0 mean  #40 + abs(Lesion_Contrasts[ii])  #New_bkg_HU; #90 for -50HU lesion; 70 for -30HU lesion; 50 for -10HU lesion;
        Noise_ROI =  ROI_All[:,:,0:Total_NPS_No]     

        depth = Noise_ROI.shape[2]
        for i in range(depth):
            temp = Noise_ROI[:, :, i]
            Noise_ROI[:, :, i] = temp - np.mean(temp)  # Adjusting the noise ROI  to be 0 mean    


        N_total = Noise_ROI.shape[2]  # Getting the size of the third dimension (similar to MATLAB's size function)
        dp = np.zeros((numResample, 1))  # Creating a zero array of shape (numResample, 1)
        sample_idx = np.random.permutation(N_total)  # Randomly permute the indices (equivalent to randperm in MATLAB)
        channelMatrix = ChannelMatrix_Generation(Chnl, ROI_size) 
        bkg_ordered = np.reshape(Noise_ROI[:, :, sample_idx], (ROI_size**2, len(sample_idx)))  # Reshape the Noise_ROI array

        sig_true = np.reshape(lesion_sig, (ROI_size**2, 1))  # Reshaping Lesion_sig to a 2D array
        dp = CHO_patient_with_resampling(sig_true,bkg_ordered,channelMatrix,internalNoise,Resampling_method)

        all_dps.append(dp)

    All_dps[mm-1,:]=all_dps
    # Clearing temporary arrays
    del Intel_images, Intel_images_Square, Intel_Edge,Images_Section
    gc.collect()

# to show Noise_level_local and Mean_loc_dps in one image
#Noise_level_local
Mean_loc_dps = np.mean(All_dps,axis=-1)
Mean_All_dps = np.mean(Mean_loc_dps)


#To show NPS and (peak frequency, average frequency, 10% peak frequency, average noise level)
NPS_1D = NPS_1D_sum/Total_NPS_Num 
Ave_noise_level = noise_level_sum/Total_NPS_Num # Average noise

end = time.time()
print(end - start)


fig3, ax3 = plt.subplots(1, 1)
# 1D NPS figure
ax3.plot(Spatial_freq, NPS_1D, "b", label="average")
ax3.set_xlabel("Spatial frequency (1/cm)", fontsize=16)
ax3.set_ylabel("NPS (HU^2 cm^2)", fontsize=16)
ax3.grid(True)
ax3.legend()
plt.show()
# Statistics
[fav, peakfrequency, k, min10percent_frequency] = NPS_statistics(NPS_1D, unit)
print('peakfrequency = %.3f\n' % peakfrequency)
print('averagefrequency = %.3f\n' % fav)
print('min10percent_frequency = %.3f\n' % min10percent_frequency)
print('average noise level = %.3f\n' % Ave_noise_level)
#print('sanity check: sqrt(area under nps2D) * unit = %.3f\n\n' % (np.sqrt(np.sum(nps2D))*unit) )
plot = [Spatial_freq, NPS_1D]

#To show Dw and CTDIvol in 3rd image

# to show folloiwng metrics in a table
   


# time: around 5.5 minutes





