#!/usr/bin/env python3

"""
Updated wrapper for CHO calculation that supports custom parameters from modal
"""

import io
import threading
import time
import traceback
import pydicom
import orthanc
from progress_tracker import progress_tracker
from patient_specific_calculation import main as unified_cho_main
import json

def run_cho_calculation_with_progress(series_uuid, full_test=False, custom_params=None):
    """
    Run CHO calculation with progress tracking and custom parameters
    
    Parameters:
    - series_uuid: The Orthanc series UUID
    - slices: The DICOM slices
    - full_test: False for global noise, True for full analysis
    - custom_params: Dictionary with custom parameters from modal
    
    Returns:
    - The calculation ID for tracking
    """
    # Default parameters
    default_params = {
        'resamples': 500,
        'internalNoise': 2.25,
        'resamplingMethod': 'Bootstrap',
        'roiSize': 6,
        'thresholdLow': 0,
        'thresholdHigh': 150,
        'windowLength': 15.0,
        'stepSize': 5.0,
        'channelType': 'Gabor',
        'lesionSet': 'standard'
    }
    
    # Merge custom parameters with defaults
    if custom_params:
        default_params.update(custom_params)
    
    # Get metadata about the calculation
    metadata = {
        # 'slice_count': len(slices),
        # 'patient_id': slices[0].PatientID if hasattr(slices[0], 'PatientID') else 'Unknown',
        # 'patient_name': str(slices[0].PatientName) if hasattr(slices[0], 'PatientName') else 'Unknown',
        'full_test': full_test,
        'calculation_type': 'full_analysis' if full_test else 'global_noise',
        'custom_parameters': default_params
    }
    
    # Start tracking the calculation
    progress_tracker.start_calculation(series_uuid, metadata)
    
    # Start the calculation in a separate thread
    thread = threading.Thread(
        target=_run_calculation_thread,
        args=(series_uuid, full_test, default_params)
    )
    thread.daemon = True
    thread.start()
    
    return series_uuid

def _run_calculation_thread(series_uuid, full_test, params):
    """
    Run the calculation in a separate thread and report progress
    """
    try:
        # Update progress to indicate we're starting
        analysis_type = "Full CHO" if full_test else "Global Noise"
        progress_tracker.update_progress(
            series_uuid, 5, 
            f"Starting {analysis_type} calculation with custom parameters...", 
            "initialization"
        )
        
        # Create config for unified CHO module with custom parameters
        config = {
            'series_uuid': series_uuid,
            'full_test': full_test,
            'report_progress': True,
            'custom_parameters': params
        }
        
        # Import and run the unified CHO calculation with progress reporting
        # Load DICOM files
        instances_res = orthanc.RestApiGet(f'/series/{series_uuid}/instances')
        instances_json = json.loads(instances_res)

        print(f"loading: {series_uuid}")
        files = []
        for i in range(len(instances_json)):
            if (i % 10 == 0 or i == len(instances_json) - 1):
                # 5 = progress during analysis
                # 15 = progress after analysis
                progress_percentage = 5 + ((i / len(instances_json)) * (15 - 5))
                progress_tracker.update_progress(   #type:ignore
                    series_uuid, int(progress_percentage), 
                    f"Loading slice {i+1}/{len(instances_json)}", 
                    "loading"
                )
            instance_id = instances_json[i]['ID']
            f = orthanc.GetDicomForInstance(instance_id)
            files.append(pydicom.dcmread(io.BytesIO(f)))
        
        slices = [f for f in files if hasattr(f, "SliceLocation")]
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        results = unified_cho_main(slices, config)
        
        # Add calculation type and parameters to results
        if results:
            results['full_test'] = full_test
            results['calculation_type'] = 'full_analysis' if full_test else 'global_noise'
            results['parameters_used'] = params
        
        # The unified module handles database saving internally
        # Schedule cleanup after some time (5 minutes)
        def delayed_cleanup():
            time.sleep(300)  # 5 minutes
            progress_tracker.cleanup_calculation(series_uuid)
            
        cleanup_thread = threading.Thread(target=delayed_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return results
        
    except Exception as e:
        error_msg = f"Error running CHO calculation: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        progress_tracker.fail_calculation(series_uuid, error_msg)
        raise

def apply_custom_parameters_to_analysis(config, custom_params):
    """
    Apply custom parameters to the analysis configuration
    This function can be called from the unified CHO module to customize behavior
    """
    if not custom_params:
        return config
    
    # Map modal parameters to internal configuration
    parameter_mapping = {
        'resamples': 'num_resample',
        'internalNoise': 'internal_noise', 
        'resamplingMethod': 'resampling_method',
        'roiSize': 'roi_size_mm',
        'thresholdLow': 'threshold_low',
        'thresholdHigh': 'threshold_high',
        'windowLength': 'window_length_cm',
        'stepSize': 'step_size_cm',
        'channelType': 'channel_type',
        'lesionSet': 'lesion_configuration'
    }
    
    # Apply the mappings
    for modal_param, internal_param in parameter_mapping.items():
        if modal_param in custom_params:
            config[internal_param] = custom_params[modal_param]
    
    return config

def get_lesion_configuration(lesion_set):
    """
    Get lesion configuration based on the selected set
    """
    lesion_configs = {
        'standard': {
            'contrasts': [-30, -30, -10, -30, -50],
            'sizes': [3, 9, 6, 6, 6],
            'roi_sizes': [21, 29, 25, 25, 25]
        },
        'low-contrast': {
            'contrasts': [-10, -15, -20],
            'sizes': [6, 6, 6],
            'roi_sizes': [25, 25, 25]
        },
        'high-contrast': {
            'contrasts': [-50, -75, -100],
            'sizes': [6, 6, 6],
            'roi_sizes': [25, 25, 25]
        }
    }
    
    return lesion_configs.get(lesion_set, lesion_configs['standard'])

def get_channel_configuration(channel_type):
    """
    Get CHO channel configuration based on type
    """
    if channel_type == 'Gabor':
        return {
            'type': 'Gabor',
            'passband': '[[1/64,1/32], [1/32,1/16]]',
            'theta': '[0, pi/2]',
            'beta': '0'
        }
    elif channel_type == 'Laguerre-Gauss':
        return {
            'type': 'Laguerre-Gauss',
            'order': 6,
            'orientation': 3
        }
    else:
        # Default to Gabor
        return {
            'type': 'Gabor',
            'passband': '[[1/64,1/32], [1/32,1/16]]',
            'theta': '[0, pi/2]',
            'beta': '0'
        }