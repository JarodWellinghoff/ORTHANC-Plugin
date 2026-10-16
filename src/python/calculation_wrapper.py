#!/usr/bin/env python3
"""Run CHO calculations and protect accepted results while reviewing reruns."""
import io
import json
import threading
import time
import traceback
import orthanc
import pydicom
from progress_tracker import progress_tracker
from patient_specific_calculation import main as unified_cho_main
from analysis_review_api import review_manager


def run_cho_calculation_with_progress(series_uuid, full_test, custom_params):
    series = json.loads(orthanc.RestApiGet(f"/series/{series_uuid}"))
    series_uid = series.get("MainDicomTags", {}).get("SeriesInstanceUID")
    if not series_uid:
        raise ValueError("The series has no SeriesInstanceUID.")
    run_id, needs_review = review_manager.begin(series_uuid, series_uid)
    params = dict(custom_params or {})
    params.setdefault("report_progress", True)
    if needs_review:
        # The scientific pipeline owns saving/deleting; disable both before it
        # starts, including when called from the batch or automation endpoints.
        params["saveResults"] = False
        params["deleteAfterCompletion"] = False
    metadata = {"full_test": full_test,
                "calculation_type": "full_analysis" if full_test else "global_noise",
                "custom_parameters": params, "run_id": run_id, "review_required": needs_review}
    try:
        if params["report_progress"]:
            progress_tracker.start_calculation(series_uuid, metadata)
        threading.Thread(target=_run_calculation_thread,
                         args=(series_uuid, full_test, params, run_id), daemon=True).start()
    except Exception:
        review_manager.fail(run_id)
        raise
    return series_uuid


def _run_calculation_thread(series_uuid, full_test, params, run_id):
    try:
        config = {"series_uuid": series_uuid, "full_test": full_test,
                  "report_progress": params["report_progress"], "custom_parameters": params}
        if config["report_progress"]:
            progress_tracker.update_progress(series_uuid, 5, "Starting analysis...", "initialization")
        instances = json.loads(orthanc.RestApiGet(f"/series/{series_uuid}/instances"))
        files = []
        for index, instance in enumerate(instances):
            if config["report_progress"] and (index % 10 == 0 or index == len(instances) - 1):
                progress_tracker.update_progress(series_uuid, int(5 + 10 * index / len(instances)),
                                                 f"Loading slice {index + 1}/{len(instances)}", "loading")
            files.append(pydicom.dcmread(io.BytesIO(orthanc.GetDicomForInstance(instance["ID"]))))
        slices = sorted((item for item in files if hasattr(item, "SliceLocation")), key=lambda item: item.SliceLocation)
        if not slices:
            raise ValueError("No slices with SliceLocation are available for analysis.")
        results = unified_cho_main(slices, config)
        if not results:
            raise ValueError("The analysis returned no results.")
        results.update({"full_test": full_test,
                        "calculation_type": "full_analysis" if full_test else "global_noise",
                        "parameters_used": params})
        review_manager.complete(run_id, results)

        def delayed_cleanup():
            time.sleep(300)
            current = progress_tracker.get_calculation_status(series_uuid)
            # An old run's timer must not delete the new run's progress.
            if current and current.get("metadata", {}).get("run_id") == run_id:
                progress_tracker.cleanup_calculation(series_uuid)
        threading.Thread(target=delayed_cleanup, daemon=True).start()
        return results
    except Exception as exc:
        review_manager.fail(run_id)
        error = f"Error running CHO calculation: {exc}"
        print(error)
        print(traceback.format_exc())
        progress_tracker.fail_calculation(series_uuid, error)
        return None


def apply_custom_parameters_to_analysis(config, custom_params):
    if not custom_params:
        return config
    mapping = {"resamples": "num_resample", "internalNoise": "internal_noise",
               "resamplingMethod": "resampling_method", "roiSize": "roi_size_mm",
               "thresholdLow": "threshold_low", "thresholdHigh": "threshold_high",
               "windowLength": "window_length_cm", "stepSize": "step_size_cm",
               "channelType": "channel_type", "lesionSet": "lesion_configuration",
               "spatialResolution": "spatial_resolution", "mtf50": "mtf50", "mtf10": "mtf10"}
    for source, target in mapping.items():
        if source in custom_params:
            config[target] = custom_params[source]
    return config


def get_channel_configuration(channel_type):
    if channel_type == "Laguerre-Gauss":
        return {"type": "Laguerre-Gauss", "order": 6, "orientation": 3}
    return {"type": "Gabor", "passband": "[[1/64,1/32], [1/32,1/16]]", "theta": "[0, pi/2]", "beta": "0"}
