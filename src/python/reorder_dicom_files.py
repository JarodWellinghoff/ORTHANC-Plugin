import os
import pydicom
import shutil
from pathlib import Path

dd = "C:\\Users\\M297802\\Documents\\2026_05_27_11_10_41"
files = list(Path(dd).iterdir())
for f in files:
    # Translate filename from hexidecimal to decimal
    try:
        num = int(f.stem, 16)
        print(f"{f.name} -> {num}")
    except ValueError:
        print(f"Skipping non-hex file: {f.name}")


def reorder_dicom_files(input_dir, output_dir=None, prefix="IMG", dry_run=False):
    """
    Rename DICOM files so filenames reflect correct order by InstanceNumber.

    Args:
        input_dir:  Directory containing the DICOM files
        output_dir: Where to write renamed files (None = rename in-place)
        prefix:     Filename prefix (default: "IMG")
        dry_run:    If True, print plan without touching files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if output_dir != input_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read all DICOM files and extract InstanceNumber ---
    entries = []
    skipped = []
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            instance_num = int(ds.InstanceNumber)
            entries.append((instance_num, f))
        except Exception:
            skipped.append(f.name)

    if not entries:
        print("No valid DICOM files found.")
        return

    if skipped:
        print(f"Skipped (not DICOM or missing InstanceNumber): {skipped}")

    # --- 2. Sort by InstanceNumber ---
    entries.sort(key=lambda x: x[0])

    # --- 3. Determine zero-padded width ---
    max_instance = entries[-1][0]
    pad_width = len(str(max_instance))

    # --- 4. Rename / copy files ---
    print(f"\n{'DRY RUN — ' if dry_run else ''}Renaming {len(entries)} files...\n")
    for instance_num, src_path in entries:
        suffix = src_path.suffix or ".dcm"
        new_name = f"{prefix}_{str(instance_num).zfill(pad_width)}{suffix}"
        dst_path = output_dir / new_name

        print(f"  {src_path.name:40s} -> {new_name}")

        if not dry_run:
            if output_dir == input_dir:
                src_path.rename(dst_path)
            else:
                shutil.copy2(str(src_path), str(dst_path))

    print("\nDone." if not dry_run else "\nDry run complete — no files were modified.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorder DICOM files by InstanceNumber"
    )
    parser.add_argument("input_dir", help="Directory containing DICOM files")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory (default: rename in-place)"
    )
    parser.add_argument(
        "--prefix", default="IMG", help="Filename prefix (default: IMG)"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Preview changes without renaming"
    )
    args = parser.parse_args()

    reorder_dicom_files(args.input_dir, args.output_dir, args.prefix, args.dry_run)
