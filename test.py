"""
Interactive MTF Plot with Matplotlib Sliders
Pure Python implementation with native GUI window
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Read the MTF data
df = pd.read_csv('mtfc_data.csv')
freq_original = df['Frequency (Hz)'].values
mtf_10hu = df['MTF_10HU'].values
mtf_30hu = df['MTF_30HU'].values
mtf_50hu = df['MTF_50HU'].values

def find_frequency_at_mtf(freq, mtf, target_mtf):
    """
    Find the frequency where MTF equals target_mtf using interpolation.
    Returns None if target not found in range.
    """
    interp_func = interp1d(freq, mtf, kind='cubic', fill_value='extrapolate')
    
    try:
        # Search for zero crossing of (MTF - target_mtf)
        result = brentq(lambda x: interp_func(x) - target_mtf, 
                       freq.min(), freq.max())
        return result
    except:
        return None

def transform_mtf_curve(freq_original, mtf_original, freq_at_05, freq_at_01):
    """
    Transform MTF curve to match specified frequencies at MTF=0.5 and MTF=0.1.
    Uses smooth cubic spline interpolation for natural, continuous transitions.
    """
    # Find original frequencies at MTF=0.5 and MTF=0.1
    orig_freq_05 = find_frequency_at_mtf(freq_original, mtf_original, 0.5)
    orig_freq_01 = find_frequency_at_mtf(freq_original, mtf_original, 0.1)
    
    if orig_freq_05 is None or orig_freq_01 is None:
        return freq_original, mtf_original
    
    # Calculate scaling factors at anchor points
    scale_05 = freq_at_05 / orig_freq_05
    scale_01 = freq_at_01 / orig_freq_01
    
    # Create interpolation function for MTF values
    interp_mtf = interp1d(freq_original, mtf_original, kind='cubic', 
                          fill_value='extrapolate', bounds_error=False)
    
    # Define anchor points for smooth scaling interpolation
    # MTF values where we define the scaling behavior
    mtf_anchors = np.array([0.0, 0.1, 0.5, 1.0])
    
    # Corresponding scale factors at each anchor
    # - At MTF=1.0: no scaling (scale=1.0)
    # - At MTF=0.5: use user-specified scale_05
    # - At MTF=0.1: use user-specified scale_01
    # - At MTF=0.0: extrapolate smoothly from scale_01
    scale_at_zero = scale_01 + (scale_01 - scale_05) * 0.1 / 0.4  # Linear extrapolation
    scale_anchors = np.array([scale_at_zero, scale_01, scale_05, 1.0])
    
    # Create smooth cubic spline for scale as a function of MTF
    # This ensures C1 continuity (continuous first derivative)
    from scipy.interpolate import CubicSpline
    scale_interpolator = CubicSpline(mtf_anchors, scale_anchors, bc_type='natural')
    
    # Apply smooth transformation to each frequency point
    freq_new = []
    mtf_new = []
    
    for f in freq_original:
        mtf_val = interp_mtf(f)
        
        # Clamp MTF to valid range
        mtf_val = np.clip(mtf_val, 0.0, 1.0)
        
        # Get smooth scale factor from spline interpolation
        scale = scale_interpolator(mtf_val)
        
        # Ensure scale is positive and reasonable
        scale = max(0.1, min(scale, 10.0))
        
        freq_new.append(f * scale)
        mtf_new.append(mtf_val)
    
    # Convert to numpy arrays
    freq_new = np.array(freq_new)
    mtf_new = np.array(mtf_new)
    
    # Sort by frequency to ensure monotonic increase
    sort_idx = np.argsort(freq_new)
    freq_new = freq_new[sort_idx]
    mtf_new = mtf_new[sort_idx]
    
    # Remove any duplicate frequencies (keep first occurrence)
    unique_idx = np.concatenate([[True], np.diff(freq_new) > 1e-10])
    freq_new = freq_new[unique_idx]
    mtf_new = mtf_new[unique_idx]
    
    # Filter to valid MTF range
    valid_idx = (mtf_new >= 0) & (mtf_new <= 1)
    
    return freq_new[valid_idx], mtf_new[valid_idx]

# Find original threshold frequencies
orig_10hu_freq_05 = find_frequency_at_mtf(freq_original, mtf_10hu, 0.5)
orig_10hu_freq_01 = find_frequency_at_mtf(freq_original, mtf_10hu, 0.1)
orig_30hu_freq_05 = find_frequency_at_mtf(freq_original, mtf_30hu, 0.5)
orig_30hu_freq_01 = find_frequency_at_mtf(freq_original, mtf_30hu, 0.1)
orig_50hu_freq_05 = find_frequency_at_mtf(freq_original, mtf_50hu, 0.5)
orig_50hu_freq_01 = find_frequency_at_mtf(freq_original, mtf_50hu, 0.1)

print("\nOriginal threshold frequencies:")
print(f"  MTF_10HU: MTF=0.5 at {orig_10hu_freq_05:.4f} Hz, MTF=0.1 at {orig_10hu_freq_01:.4f} Hz")
print(f"  MTF_30HU: MTF=0.5 at {orig_30hu_freq_05:.4f} Hz, MTF=0.1 at {orig_30hu_freq_01:.4f} Hz")
print(f"  MTF_50HU: MTF=0.5 at {orig_50hu_freq_05:.4f} Hz, MTF=0.1 at {orig_50hu_freq_01:.4f} Hz")
print("\nInteractive window opening... Use the sliders to adjust the curves.")
print("Close the window when finished.\n")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.45, right=0.95, top=0.95)

# Plot original curves (dotted)
ax.plot(freq_original, mtf_10hu, 'b:', alpha=0.5, linewidth=1.5, label='MTF_10HU (Original)')
ax.plot(freq_original, mtf_30hu, 'g:', alpha=0.5, linewidth=1.5, label='MTF_30HU (Original)')
ax.plot(freq_original, mtf_50hu, 'r:', alpha=0.5, linewidth=1.5, label='MTF_50HU (Original)')

# Plot simulated curves (solid) - will be updated by sliders
line_10hu, = ax.plot(freq_original, mtf_10hu, 'b-', linewidth=2, label='MTF_10HU (Simulated)')
line_30hu, = ax.plot(freq_original, mtf_30hu, 'g-', linewidth=2, label='MTF_30HU (Simulated)')
line_50hu, = ax.plot(freq_original, mtf_50hu, 'r-', linewidth=2, label='MTF_50HU (Simulated)')

# Add reference lines
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(ax.get_xlim()[1] * 0.95, 0.5, 'MTF = 0.5', va='center', ha='right', 
        fontsize=9, color='gray')
ax.text(ax.get_xlim()[1] * 0.95, 0.1, 'MTF = 0.1', va='center', ha='right', 
        fontsize=9, color='gray')

# Labels and title
ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
ax.set_ylabel('MTF', fontsize=12, fontweight='bold')
ax.set_title('Interactive MTF Curves - Adjust Threshold Frequencies', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(-0.05, 1.05)

# Create slider axes (6 sliders total)
slider_height = 0.03
slider_spacing = 0.05
slider_bottom = 0.05

# Define slider positions (from bottom to top)
ax_slider_10hu_05 = plt.axes([0.15, slider_bottom + 5*slider_spacing, 0.7, slider_height])
ax_slider_10hu_01 = plt.axes([0.15, slider_bottom + 4*slider_spacing, 0.7, slider_height])
ax_slider_30hu_05 = plt.axes([0.15, slider_bottom + 3*slider_spacing, 0.7, slider_height])
ax_slider_30hu_01 = plt.axes([0.15, slider_bottom + 2*slider_spacing, 0.7, slider_height])
ax_slider_50hu_05 = plt.axes([0.15, slider_bottom + 1*slider_spacing, 0.7, slider_height])
ax_slider_50hu_01 = plt.axes([0.15, slider_bottom + 0*slider_spacing, 0.7, slider_height])

# Create sliders with appropriate ranges (30% to 170% of original)
slider_10hu_05 = Slider(
    ax_slider_10hu_05, '10HU @ MTF=0.5', 
    orig_10hu_freq_05 * 0.3, orig_10hu_freq_05 * 1.7,
    valinit=orig_10hu_freq_05, valstep=0.001, color='blue'
)

slider_10hu_01 = Slider(
    ax_slider_10hu_01, '10HU @ MTF=0.1',
    orig_10hu_freq_01 * 0.3, orig_10hu_freq_01 * 1.7,
    valinit=orig_10hu_freq_01, valstep=0.001, color='blue'
)

slider_30hu_05 = Slider(
    ax_slider_30hu_05, '30HU @ MTF=0.5',
    orig_30hu_freq_05 * 0.3, orig_30hu_freq_05 * 1.7,
    valinit=orig_30hu_freq_05, valstep=0.001, color='green'
)

slider_30hu_01 = Slider(
    ax_slider_30hu_01, '30HU @ MTF=0.1',
    orig_30hu_freq_01 * 0.3, orig_30hu_freq_01 * 1.7,
    valinit=orig_30hu_freq_01, valstep=0.001, color='green'
)

slider_50hu_05 = Slider(
    ax_slider_50hu_05, '50HU @ MTF=0.5',
    orig_50hu_freq_05 * 0.3, orig_50hu_freq_05 * 1.7,
    valinit=orig_50hu_freq_05, valstep=0.001, color='red'
)

slider_50hu_01 = Slider(
    ax_slider_50hu_01, '50HU @ MTF=0.1',
    orig_50hu_freq_01 * 0.3, orig_50hu_freq_01 * 1.7,
    valinit=orig_50hu_freq_01, valstep=0.001, color='red'
)

# Update function for sliders
def update(val):
    """Update the plot when any slider changes"""
    # Get current slider values
    freq_10hu_05 = slider_10hu_05.val
    freq_10hu_01 = slider_10hu_01.val
    freq_30hu_05 = slider_30hu_05.val
    freq_30hu_01 = slider_30hu_01.val
    freq_50hu_05 = slider_50hu_05.val
    freq_50hu_01 = slider_50hu_01.val
    
    # Transform curves based on new threshold frequencies
    freq_10hu_new, mtf_10hu_new = transform_mtf_curve(
        freq_original, mtf_10hu, freq_10hu_05, freq_10hu_01
    )
    freq_30hu_new, mtf_30hu_new = transform_mtf_curve(
        freq_original, mtf_30hu, freq_30hu_05, freq_30hu_01
    )
    freq_50hu_new, mtf_50hu_new = transform_mtf_curve(
        freq_original, mtf_50hu, freq_50hu_05, freq_50hu_01
    )
    
    # Update plot data
    line_10hu.set_xdata(freq_10hu_new)
    line_10hu.set_ydata(mtf_10hu_new)
    line_30hu.set_xdata(freq_30hu_new)
    line_30hu.set_ydata(mtf_30hu_new)
    line_50hu.set_xdata(freq_50hu_new)
    line_50hu.set_ydata(mtf_50hu_new)
    
    # Auto-scale x-axis to fit all data
    all_freqs = np.concatenate([freq_10hu_new, freq_30hu_new, freq_50hu_new])
    ax.set_xlim(0, max(all_freqs.max() * 1.05, freq_original.max() * 1.1))
    
    # Redraw the canvas
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_10hu_05.on_changed(update)
slider_10hu_01.on_changed(update)
slider_30hu_05.on_changed(update)
slider_30hu_01.on_changed(update)
slider_50hu_05.on_changed(update)
slider_50hu_01.on_changed(update)

# Show the interactive plot
plt.show()

print("\nPlot window closed.")
print("To save current slider values, modify the script to export data.")
