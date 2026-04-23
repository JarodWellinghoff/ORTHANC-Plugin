#!/usr/bin/env python
"""
run_test.py  -  Test runner for CHO_Calculation_Patient_Specific_skimage_Canny_edge_v22_copy.py
=====================================================

Usage
-----
# Minimum (simulated PSF, no MTF10 → pure Gaussian):
    python run_test.py --dicom_dir /path/to/dicoms --lesion_file lesion.mat --mtf50 0.434

# Full simulated PSF with MTF10:
    python run_test.py --dicom_dir /path/to/dicoms --lesion_file lesion.mat --mtf50 0.434 --mtf10 0.730

# Measured PSF:
    python run_test.py --dicom_dir /path/to/dicoms --lesion_file lesion.mat --psf_file psf.mat

# Custom output directory:
    python run_test.py --dicom_dir /path/to/dicoms --lesion_file lesion.mat --mtf50 0.434 --output_dir ./results

Outputs (written to --output_dir, default: ./test_output_<timestamp>/)
-------
  results.json          - Full results dictionary
  summary.txt           - Human-readable key metrics
  *.png                 - All matplotlib figures
"""
import CHO_Calculation_Patient_Specific_skimage_Canny_edge_v22_copy as pipeline
import argparse
import json
import os
import pathlib
import sys
import textwrap
import traceback
from datetime import datetime
from pydicom import dcmread


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test runner for CHO_Calculation_Patient_Specific_skimage_Canny_edge_v22_copy.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )

    # --- Required ---
    parser.add_argument(
        "--dicom_dir",
        required=True,
        help="[REQUIRED] Directory containing DICOM image files.",
    )
    parser.add_argument(
        "--lesion_file",
        required=True,
        help="[REQUIRED] Path to the lesion .mat file.",
    )

    # --- PSF / MTF (at least one required) ---
    psf_group = parser.add_argument_group(
        "PSF / MTF",
        "Provide either --psf_file OR --mtf50 (and optionally --mtf10).",
    )
    psf_group.add_argument(
        "--psf_file",
        default=None,
        help="Path to measured PSF .mat file.  Mutually exclusive with --mtf50/--mtf10.",
    )
    psf_group.add_argument(
        "--mtf50",
        type=float,
        default=None,
        help="Spatial frequency (cycles/mm) at MTF=0.50.  Required when --psf_file is omitted.",
    )
    psf_group.add_argument(
        "--mtf10",
        type=float,
        default=None,
        help="Spatial frequency (cycles/mm) at MTF=0.10.  "
        "Optional; omit to use a pure Gaussian PSF (only --mtf50 used).",
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save figures and results.  "
        "Defaults to ./test_output_<YYYYMMDD_HHMMSS>/",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_inputs(args) -> list[str]:
    """
    Check all supplied paths exist and that the PSF/MTF combination is valid.
    Returns a list of error strings (empty → all OK).
    """
    errors = []

    # DICOM directory
    if not os.path.isdir(args.dicom_dir):
        errors.append(
            f"--dicom_dir does not exist or is not a directory: {args.dicom_dir!r}"
        )
    else:
        dicoms = [f for f in os.listdir(args.dicom_dir) if not f.startswith(".")]
        if len(dicoms) < 2:
            errors.append(
                f"--dicom_dir contains fewer than 2 files ({len(dicoms)} found): {args.dicom_dir!r}"
            )

    # Lesion file
    if not os.path.isfile(args.lesion_file):
        errors.append(f"--lesion_file not found: {args.lesion_file!r}")

    # PSF / MTF mutual-exclusivity
    if args.psf_file is not None and args.mtf50 is not None:
        errors.append("Provide --psf_file OR --mtf50/--mtf10, not both.")
    elif args.psf_file is None and args.mtf50 is None:
        errors.append(
            "Must supply either --psf_file or --mtf50 (required when PSF is simulated)."
        )

    if args.psf_file is not None and not os.path.isfile(args.psf_file):
        errors.append(f"--psf_file not found: {args.psf_file!r}")

    # MTF range sanity
    if args.mtf50 is not None and not (0 < args.mtf50 < 20):
        errors.append(
            f"--mtf50 value looks out of range: {args.mtf50} (expected ~ 0.1-5 cycles/mm)"
        )
    if args.mtf10 is not None and not (0 < args.mtf10 < 20):
        errors.append(
            f"--mtf10 value looks out of range: {args.mtf10} (expected ~ 0.1-5 cycles/mm)"
        )
    if args.mtf50 is not None and args.mtf10 is not None and args.mtf10 <= args.mtf50:
        errors.append(
            f"--mtf10 ({args.mtf10}) should be greater than --mtf50 ({args.mtf50}) "
            "(MTF declines with frequency)."
        )

    return errors


def validate_inputs_dict(args_dict: dict) -> list[str]:
    """
    Check all supplied paths exist and that the PSF/MTF combination is valid.
    Returns a list of error strings (empty → all OK).
    """
    errors = []

    # DICOM directory
    if not os.path.isdir(args_dict.get("dicom_dir", "")):
        errors.append(
            f"--dicom_dir does not exist or is not a directory: {args_dict.get('dicom_dir', '')!r}"
        )
    else:
        dicoms = [
            f
            for f in os.listdir(args_dict.get("dicom_dir", ""))
            if not f.startswith(".")
        ]
        if len(dicoms) < 2:
            errors.append(
                f"--dicom_dir contains fewer than 2 files ({len(dicoms)} found): {args_dict.get('dicom_dir', '')!r}"
            )

    # Lesion file
    if not os.path.isfile(args_dict.get("lesion_file", "")):
        errors.append(f"--lesion_file not found: {args_dict.get('lesion_file', '')!r}")

    # PSF / MTF mutual-exclusivity
    if args_dict.get("psf_file") is not None and args_dict.get("mtf50") is not None:
        errors.append("Provide --psf_file OR --mtf50/--mtf10, not both.")
    elif args_dict.get("psf_file") is None and args_dict.get("mtf50") is None:
        errors.append(
            "Must supply either --psf_file or --mtf50 (required when PSF is simulated)."
        )

    if args_dict.get("psf_file") is not None and not os.path.isfile(
        args_dict.get("psf_file")
    ):
        errors.append(f"--psf_file not found: {args_dict.get('psf_file')!r}")

    # MTF range sanity
    if args_dict.get("mtf50") is not None and not (0 < args_dict.get("mtf50") < 20):
        errors.append(
            f"--mtf50 value looks out of range: {args_dict.get('mtf50')} (expected ~ 0.1-5 cycles/mm)"
        )
    if args_dict.get("mtf10") is not None and not (0 < args_dict.get("mtf10") < 20):
        errors.append(
            f"--mtf10 value looks out of range: {args_dict.get('mtf10')} (expected ~ 0.1-5 cycles/mm)"
        )
    if (
        args_dict.get("mtf50") is not None
        and args_dict.get("mtf10") is not None
        and args_dict.get("mtf10") <= args_dict.get("mtf50")
    ):
        errors.append(
            f"--mtf10 ({args_dict.get('mtf10')}) should be greater than --mtf50 ({args_dict.get('mtf50')}) "
            "(MTF declines with frequency)."
        )

    return errors


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def write_summary(results: dict, output_dir: pathlib.Path, args):
    """Write a human-readable summary of key metrics to summary.txt."""
    lines = [
        "=" * 60,
        "  CT Image Quality Pipeline  -  Test Summary",
        "=" * 60,
        f"  Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  DICOM dir     : {args.dicom_dir}",
        f"  Lesion file   : {args.lesion_file}",
        f"  PSF source    : {'Measured - ' + str(args.psf_file) if args.psf_file else 'Simulated'}",
    ]
    if not args.psf_file:
        lines.append(f"  MTF50         : {args.mtf50} cycles/mm")
        if args.mtf10:
            lines.append(f"  MTF10         : {args.mtf10} cycles/mm")
        else:
            lines.append("  MTF10         : (not set - pure Gaussian)")
    lines += [
        "",
        "  -- Dose ----------------------------------------------",
        f"  CTDIvol avg   : {results.get('ctdivol_avg', 'N/A'):.3f} mGy",
        f"  SSDE          : {results.get('ssde', 'N/A'):.3f} mGy",
        f"  DLP           : {results.get('dlp', 'N/A'):.1f} mGy*cm",
        f"  DLP (SSDE)    : {results.get('dlp_ssde', 'N/A'):.1f} mGy*cm",
        f"  Dw avg        : {results.get('dw_avg', 'N/A'):.1f} cm",
        "",
        "  -- Noise / NPS ---------------------------------------",
        f"  Avg noise lvl : {results.get('average_noise_level', 'N/A'):.3f} HU",
        f"  Peak freq     : {results.get('peak_frequency', 'N/A'):.3f} cycles/mm",
        f"  Avg freq      : {results.get('average_frequency', 'N/A'):.3f} cycles/mm",
        f"  10% freq      : {results.get('percent_10_frequency', 'N/A'):.3f} cycles/mm",
        "",
        "  -- CHO Detectability (d') ----------------------------",
        f"  Mean d' (all) : {results.get('average_index_of_detectability', 'N/A'):.4f}",
    ]

    cho = results.get("cho_detectability", [])
    contrasts = [-30, -30, -10, -30, -50]
    sizes = [3, 9, 6, 6, 6]
    for i, dp in enumerate(cho):
        lines.append(
            f"    Lesion {i+1}  ({contrasts[i]:+d} HU, {sizes[i]} mm) : d' = {dp:.4f}"
        )

    lines += [
        "",
        f"  Elapsed time  : {results.get('elapsed_seconds', 'N/A'):.1f} s",
        "=" * 60,
    ]

    summary_path = output_dir / "summary.txt"
    summary_text = "\n".join(lines)
    with open(summary_path, "w") as fh:
        fh.write(summary_text)
    print(summary_text)
    print(f"\n  Summary saved: {summary_path}")


def write_summary_dict(results: dict, output_dir: pathlib.Path, input_dict: dict):
    """Write a human-readable summary of key metrics to summary.txt."""
    lines = [
        "=" * 60,
        "  CT Image Quality Pipeline  -  Test Summary",
        "=" * 60,
        f"  Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  DICOM dir     : {input_dict.get('dicom_dir', 'N/A')}",
        f"  Lesion file   : {input_dict.get('lesion_file', 'N/A')}",
        f"  PSF source    : {'Measured - ' + str(input_dict.get('psf_file')) if input_dict.get('psf_file') else 'Simulated'}",
    ]
    if not input_dict.get("psf_file"):
        lines.append(f"  MTF50         : {input_dict.get('mtf50', 'N/A')} cycles/mm")
        if input_dict.get("mtf10"):
            lines.append(
                f"  MTF10         : {input_dict.get('mtf10', 'N/A')} cycles/mm"
            )
        else:
            lines.append("  MTF10         : (not set - pure Gaussian)")
    lines += [
        "",
        "  -- Dose ----------------------------------------------",
        f"  CTDIvol avg   : {results.get('ctdivol_avg', 'N/A'):.3f} mGy",
        f"  SSDE          : {results.get('ssde', 'N/A'):.3f} mGy",
        f"  DLP           : {results.get('dlp', 'N/A'):.1f} mGy*cm",
        f"  DLP (SSDE)    : {results.get('dlp_ssde', 'N/A'):.1f} mGy*cm",
        f"  Dw avg        : {results.get('dw_avg', 'N/A'):.1f} cm",
        "",
        "  -- Noise / NPS ----------------------------------------",
        f"  Avg noise lvl : {results.get('average_noise_level', 'N/A'):.3f} HU",
        f"  Peak freq     : {results.get('peak_frequency', 'N/A'):.3f} cycles/mm",
        f"  Avg freq      : {results.get('average_frequency', 'N/A'):.3f} cycles/mm",
        f"  10% freq      : {results.get('percent_10_frequency', 'N/A'):.3f} cycles/mm",
        "",
        "  -- CHO Detectability (d') -----------------------------",
        f"  Mean d' (all) : {results.get('average_index_of_detectability', 'N/A'):.4f}",
    ]

    cho = results.get("cho_detectability", [])
    contrasts = [-30, -30, -10, -30, -50]
    sizes = [3, 9, 6, 6, 6]
    for i, (contrast, size) in enumerate(zip(contrasts, sizes)):
        lines.append(f"    Lesion {i+1}  ({contrast:+d} HU, {size} mm)")

    for i, dp in enumerate(cho):
        lines.append(f"    Window {i+1} : d' = {dp:.4f}")

    lines += [
        "",
        f"  Elapsed time  : {results.get('elapsed_seconds', 'N/A'):.1f} s",
        "=" * 60,
    ]

    summary_path = output_dir / "summary.txt"
    summary_text = "\n".join(lines)
    with open(summary_path, "w") as fh:
        fh.write(summary_text)
    print(summary_text)
    print(f"\n  Summary saved: {summary_path}")


# ---------------------------------------------------------------------------
# Main test entry point
# ---------------------------------------------------------------------------


def run_test():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Validate inputs before touching the pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Validating inputs…")
    errors = validate_inputs(args)
    if errors:
        print("\n  FAILED - input validation errors:\n")
        for err in errors:
            print(f"    ✗  {err}")
        print()
    print("  ✓  All inputs look valid.")

    # ------------------------------------------------------------------
    # 2. Resolve output directory
    # ------------------------------------------------------------------
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = pathlib.Path(f"test_output_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] Output directory: {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # 3. Import and run the pipeline
    # ------------------------------------------------------------------
    print("\n[3/4] Running pipeline…\n")

    # Import here so import errors are caught and reported cleanly
    results = {}
    try:
        results = pipeline.main(
            dicom_dir=args.dicom_dir,
            lesion_file=args.lesion_file,
            psf_file=args.psf_file,
            mtf50=args.mtf50,
            mtf10=args.mtf10,
            output_dir=str(output_dir),
        )
    except FileNotFoundError as exc:
        _fail(
            "A required file was not found during pipeline execution.", exc, output_dir
        )
    except ValueError as exc:
        _fail("A value error occurred (bad input data or parameter).", exc, output_dir)
    except MemoryError as exc:
        _fail(
            "Out of memory during pipeline execution.",
            exc,
            output_dir,
            hint="Try reducing the number of DICOM slices or ROI count.",
        )
    except Exception as exc:
        _fail("Unexpected error during pipeline execution.", exc, output_dir)

    # ------------------------------------------------------------------
    # 4. Write human-readable summary
    # ------------------------------------------------------------------
    print("\n[4/4] Writing summary…\n")
    write_summary(results, output_dir, args)
    print("\n  ✓  Test completed successfully.\n")


def run_test_dict(input_dict):
    # ------------------------------------------------------------------
    # 1. Validate inputs before touching the pipeline
    # ------------------------------------------------------------------
    print("\n[1/4] Validating inputs…")
    errors = validate_inputs_dict(input_dict)
    if errors:
        print("\n  FAILED - input validation errors:\n")
        for err in errors:
            print(f"    ✗  {err}")
        print()
    print("  ✓  All inputs look valid.")

    # ------------------------------------------------------------------
    # 2. Resolve output directory
    # ------------------------------------------------------------------
    if input_dict.get("output_dir"):
        output_dir = pathlib.Path(input_dict["output_dir"])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = pathlib.Path(f"test_output_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/4] Output directory: {output_dir.resolve()}")

    # ------------------------------------------------------------------
    # 3. Import and run the pipeline
    # ------------------------------------------------------------------
    print("\n[3/4] Running pipeline…\n")
    results = {}
    try:
        results = pipeline.main(
            dicom_dir=input_dict["dicom_dir"],
            lesion_file=input_dict["lesion_file"],
            psf_file=input_dict.get("psf_file"),
            mtf50=input_dict.get("mtf50"),
            mtf10=input_dict.get("mtf10"),
            output_dir=str(output_dir),
        )
    except FileNotFoundError as exc:
        _fail(
            "A required file was not found during pipeline execution.", exc, output_dir
        )
    except ValueError as exc:
        _fail("A value error occurred (bad input data or parameter).", exc, output_dir)
    except MemoryError as exc:
        _fail(
            "Out of memory during pipeline execution.",
            exc,
            output_dir,
            hint="Try reducing the number of DICOM slices or ROI count.",
        )
    except Exception as exc:  # noqa: BLE001
        _fail("Unexpected error during pipeline execution.", exc, output_dir)

    # ------------------------------------------------------------------
    # 4. Write human-readable summary
    # ------------------------------------------------------------------
    print("\n[4/4] Writing summary…\n")
    write_summary_dict(results, output_dir, input_dict)
    print("\n  ✓  Test completed successfully.\n")


# ---------------------------------------------------------------------------
# Failure handler
# ---------------------------------------------------------------------------


def _fail(message: str, exc: Exception, output_dir: pathlib.Path, hint: str = ""):
    """Print a structured failure report, write error.txt, and exit non-zero."""
    tb = traceback.format_exc()

    report_lines = [
        "",
        "  ╔══════════════════════════════════════════════════════╗",
        "  ║                    TEST  FAILED                      ║",
        "  ╚══════════════════════════════════════════════════════╝",
        "",
        f"  Reason  : {message}",
        f"  Error   : {type(exc).__name__}: {exc}",
    ]
    if hint:
        report_lines.append(f"  Hint    : {hint}")
    report_lines += [
        "",
        "  -- Traceback -----------------------------------------",
        tb,
    ]
    report = "\n".join(report_lines)
    print(report)

    # Also persist the error to disk so it isn't lost
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / "error.txt"
        with open(error_path, "w") as fh:
            fh.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            fh.write(report)
        print(f"  Error details saved: {error_path}\n")
    except Exception:  # noqa: BLE001
        pass  # Don't let the error handler itself crash


if __name__ == "__main__":
    import glob

    psf_file_options = [
        r"C:\Users\M297802\Desktop\ORTHANC Plugin\src\data\EID_PSF_Br44.mat",
        None,
    ]
    lesion_file = r"C:\Users\M297802\Desktop\ORTHANC Plugin\src\data\\Patient02-411-920_Lesion1.mat"
    mtf50_options = [None, 0.434]
    mtf10_options = [None, 0.730]
    output_dir = r"D:\results"
    m = r"\\mfad\researchMN\EB036541\YU\PUBLIC\PatientCT_monitoring\DICOMs\**\IMAGES"
    dicom_dir_options = glob.glob(m, recursive=True)
    tests = []
    for dcm_dir in dicom_dir_options:
        name = os.path.basename(os.path.dirname(dcm_dir))
        scanner = os.path.basename(os.path.dirname(os.path.dirname(dcm_dir)))
        dcm_files = os.listdir(dcm_dir)
        dcm_file = dcmread(os.path.join(dcm_dir, dcm_files[0]))
        body_part = dcm_file.get("BodyPartExamined", "Unknown").upper()
        base_output = os.path.join(output_dir, f"{scanner}-{name}-{body_part}")
        base_test = {
            "dicom_dir": dcm_dir,
            "lesion_file": lesion_file,
            "body_part": body_part,
        }
        default_test = base_test.copy()

        tests.append(
            {
                **default_test,
                "psf_file": psf_file_options[0],
                "mtf50": mtf50_options[0],
                "mtf10": mtf10_options[0],
                "output_dir": os.path.join(base_output, "measured_psf"),
            }
        )
        # for mtf10 in mtf10_options:
        #     m10 = f"_mtf10_{mtf10}" if mtf10 else "_mtf10_None"
        #     tests.append(
        #         {
        #             **default_test,
        #             "psf_file": psf_file_options[1],
        #             "mtf50": mtf50_options[1],
        #             "mtf10": mtf10,
        #             "output_dir": os.path.join(
        #                 base_output,
        #                 f"mtf50_{mtf50_options[1]}{m10}",
        #             ),
        #         }
        #     )
    tests = sorted(tests, key=lambda d: d["dicom_dir"])
    tests = [d for d in tests if not "HEAD" in d.get("body_part", "")]
    # tests = [d for d in tests if "ROLAND, THOMAS" in d.get("dicom_dir", "")]
    # tests = [d for d in tests if "_mtf10_None" in d.get("output_dir", "")]
    for test_input in tests:
        try:
            print(f"\n\n=== Running test with DICOM dir: {test_input['dicom_dir']} ===")
            run_test_dict(test_input)
        except Exception as exc:
            print(f"\nTest failed for DICOM dir: {test_input['dicom_dir']}")
            print(f"Error: {exc}")
