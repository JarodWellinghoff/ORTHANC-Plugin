# dicom_parser.py

import os
import re
from datetime import datetime
from collections.abc import Sequence

from pydicom import dcmread, config
from pydicom.multival import MultiValue
from pydicom.errors import InvalidDicomError

# Relax validation so we can handle non-conformant stuff ourselves
config.enforce_valid_values = False


class DicomParser:
    """
    Minimal CT-oriented DICOM parser with robust handling of 'NA' / bad numeric
    values and a safe JSON export helper.
    """

    def __init__(self, dicom_data=None) -> None:
        self.dicom_data = dicom_data
        # canonical NA markers (string check is always done case-insensitively)
        self.NA_SET = {"", "NA", "N/A", None}
        self._uid_re = re.compile(r"^[0-9.]{1,64}$")

    # -------------------------------------------------------------------------
    # Helpers for NA / bad values
    # -------------------------------------------------------------------------

    def _is_na(self, v) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip().upper() in self.NA_SET
        return False

    def sanitize_for_json(self, ds):
        """
        Clean numeric VRs so that ds.to_json_dict()/to_json() won't crash on
        bad values like 'NA', 'N/A', '' etc.
        """

        numeric_vrs = {
            "DS",
            "IS",
            "FL",
            "FD",
            "SL",
            "SS",
            "UL",
            "US",
            "OF",
            "OD",
        }
        int_vrs = {"IS", "SL", "SS", "UL", "US"}

        for elem in ds.iterall():
            if elem.VR not in numeric_vrs:
                continue

            v = elem.value
            if v is None:
                continue

            # Multi-valued: any non-string Sequence (list, tuple, MultiValue, etc.)
            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                cleaned = []
                for x in v:
                    if self._is_na(x):
                        cleaned.append(None)
                        continue
                    try:
                        if elem.VR in int_vrs:
                            cleaned.append(int(x))
                        else:
                            cleaned.append(float(x))
                    except (TypeError, ValueError):
                        cleaned.append(None)
                elem.value = cleaned

            # Single-valued scalar
            else:
                if self._is_na(v):
                    elem.value = None
                else:
                    try:
                        if elem.VR in int_vrs:
                            elem.value = int(v)
                        else:
                            elem.value = float(v)
                    except (TypeError, ValueError):
                        elem.value = None

    # -------------------------------------------------------------------------
    # Core metadata extraction
    # -------------------------------------------------------------------------

    def extract_core(self):
        if not self.dicom_data:
            raise ValueError("No DICOM data available")

        ds_first = self.dicom_data[0]
        ds_second = self.dicom_data[1] if len(self.dicom_data) > 1 else None
        ds_last = self.dicom_data[-1] if len(self.dicom_data) > 1 else None

        slice_interval_mm = None
        number_of_frames = None
        series_length_mm = None

        if (
            ds_second
            and hasattr(ds_first, "SliceLocation")
            and hasattr(ds_second, "SliceLocation")
        ):
            slice_interval_mm = round(
                abs(ds_second.SliceLocation - ds_first.SliceLocation), 3
            )

        if (
            ds_last
            and hasattr(ds_first, "InstanceNumber")
            and hasattr(ds_last, "InstanceNumber")
        ):
            number_of_frames = abs(ds_last.InstanceNumber - ds_first.InstanceNumber) + 1

        if (
            ds_last
            and hasattr(ds_first, "SliceThickness")
            and hasattr(ds_first, "SliceLocation")
            and hasattr(ds_last, "SliceLocation")
        ):
            series_length_mm = (
                abs(ds_last.SliceLocation - ds_first.SliceLocation)
                + ds_first.SliceThickness
            )

        # Patient
        patient = {
            "patient_id": self._norm_text(ds_first.get("PatientID")),
            "name": self._norm_text(ds_first.get("PatientName")),
            "birth_date": self._parse_date(ds_first.get("PatientBirthDate")),
            "sex": self._norm_text(ds_first.get("PatientSex")),
            "weight_kg": self._norm_text(ds_first.get("PatientWeight")),
        }

        # Study
        study = {
            "study_instance_uid": self._validate_uid(ds_first.StudyInstanceUID),
            "study_id": self._norm_text(ds_first.get("StudyID")),
            "accession_number": self._norm_text(ds_first.get("AccessionNumber")),
            "study_date": self._parse_date(ds_first.get("StudyDate")),
            "study_time": self._parse_time(ds_first.get("StudyTime")),
            "description": self._norm_text(ds_first.get("StudyDescription")),
            "referring_physician": self._norm_text(
                ds_first.get("ReferringPhysicianName")
            ),
            "institution_name": self._norm_text(ds_first.get("InstitutionName")),
            "institution_address": self._norm_text(ds_first.get("InstitutionAddress")),
        }

        # Scanner
        scanner = {
            "manufacturer": self._norm_text(ds_first.get("Manufacturer")),
            "model_name": self._norm_text(ds_first.get("ManufacturerModelName")),
            "device_serial_number": self._norm_text(ds_first.get("DeviceSerialNumber")),
            "software_versions": self._norm_text(ds_first.get("SoftwareVersions")),
            "station_name": self._norm_text(ds_first.get("StationName")),
            "institution_name": self._norm_text(ds_first.get("InstitutionName")),
        }

        # Series
        series = {
            "series_instance_uid": self._validate_uid(
                ds_first.get("SeriesInstanceUID")
            ),
            "series_number": self._norm_text(ds_first.get("SeriesNumber")),
            "description": self._norm_text(ds_first.get("SeriesDescription")),
            "modality": self._norm_text(ds_first.get("Modality")),
            "body_part_examined": self._norm_text(ds_first.get("BodyPartExamined")),
            "protocol_name": self._norm_text(ds_first.get("ProtocolName")),
            "convolution_kernel": self._norm_text(ds_first.get("ConvolutionKernel")),
            "patient_position": self._norm_text(ds_first.get("PatientPosition")),
            "series_date": self._parse_date(ds_first.get("SeriesDate")),
            "series_time": self._parse_time(ds_first.get("SeriesTime")),
            "frame_of_reference_uid": self._validate_uid(
                ds_first.get("FrameOfReferenceUID")
            ),
            "image_type": [
                self._norm_text(x) for x in (ds_first.get("ImageType") or [])
            ],
            "slice_thickness_mm": self._norm_text(ds_first.get("SliceThickness")),
            "slice_interval_mm": slice_interval_mm,
            "pixel_spacing_mm": self._check_len2(
                "PixelSpacing", self._as_list(ds_first.get("PixelSpacing"), float)
            ),
            "rows": self._norm_text(ds_first.get("Rows")),
            "columns": self._norm_text(ds_first.get("Columns")),
            "series_length_mm": series_length_mm,
            "number_of_frames": number_of_frames,
        }

        # CT technique (per-series)
        ct = {
            "kvp": self._norm_text(ds_first.get("KVP")),
            "exposure_time_ms": self._norm_text(ds_first.get("ExposureTime")),
            "generator_power_kW": self._norm_text(ds_first.get("GeneratorPower")),
            "focal_spots_mm": self._as_list(ds_first.get("FocalSpots"), float),
            "filter_type": self._norm_text(ds_first.get("FilterType")),
            "data_collection_diam_mm": self._norm_text(
                ds_first.get("DataCollectionDiameter")
            ),
            "recon_diameter_mm": self._norm_text(
                ds_first.get("ReconstructionDiameter")
            ),
            "dist_src_detector_mm": self._norm_text(
                ds_first.get("DistanceSourceToDetector")
            ),
            "dist_src_patient_mm": self._norm_text(
                ds_first.get("DistanceSourceToPatient")
            ),
            "gantry_detector_tilt_deg": self._norm_text(
                ds_first.get("GantryDetectorTilt")
            ),
            "single_collimation_width_mm": self._norm_text(
                ds_first.get("SingleCollimationWidth")
            ),
            "total_collimation_width_mm": self._norm_text(
                ds_first.get("TotalCollimationWidth")
            ),
            "table_speed_mm_s": self._norm_text(ds_first.get("TableSpeed")),
            "table_feed_per_rot_mm": self._norm_text(
                ds_first.get("TableFeedPerRotation")
            ),
            "spiral_pitch_factor": self._norm_text(ds_first.get("SpiralPitchFactor")),
            "exposure_modulation_type": self._norm_text(
                ds_first.get("ExposureModulationType")
            ),
        }

        # Basic sanity checks
        if series["modality"] and series["modality"] != "CT":
            raise ValueError(f"Not CT modality: {series['modality']}")

        return patient, study, scanner, series, ct

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_dicoms(self, dicom_path: str) -> None:
        """Load all DICOMs with SliceLocation from a folder, sorted by SliceLocation."""
        dicom_data = []
        try:
            for fname in os.listdir(dicom_path):
                fpath = os.path.join(dicom_path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    dcm = dcmread(fpath)
                except InvalidDicomError:
                    continue
                if hasattr(dcm, "SliceLocation"):
                    # sanitize here so all downstream JSON ops are safe
                    self.sanitize_for_json(dcm)
                    dicom_data.append(dcm)
            self.dicom_data = sorted(dicom_data, key=lambda s: s.SliceLocation)
        except InvalidDicomError:
            self.dicom_data = None

    # -------------------------------------------------------------------------
    # Low-level normalization helpers
    # -------------------------------------------------------------------------

    def _norm_text(self, v):
        if isinstance(v, MultiValue):
            v = "_".join(str(x) for x in v)
        if v in self.NA_SET:
            return None
        s = str(v).strip()
        s = s.replace("\r\n", ", ")
        return None if self._is_na(s) else s

    def _parse_date(self, v):
        v = self._norm_text(v)
        if not v:
            return None
        return datetime.strptime(v, "%Y%m%d").date()

    def _parse_time(self, v):
        v = self._norm_text(v)
        if not v:
            return None
        fmt = "%H%M" if len(v) <= 4 else ("%H%M%S.%f" if "." in v else "%H%M%S")
        return datetime.strptime(v, fmt).time()

    def _as_list(self, v, cast=float):
        if v is None or self._is_na(v):
            return None

        # Multi-valued attribute
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            cleaned = []
            for x in v:
                if self._is_na(x):
                    continue
                try:
                    cleaned.append(cast(x))
                except (TypeError, ValueError):
                    continue
            return cleaned or None

        # Single-valued attribute
        try:
            return [cast(v)]
        except (TypeError, ValueError):
            return None

    def _validate_uid(self, u):
        u = self._norm_text(u)
        if not u or not self._uid_re.fullmatch(u):
            raise ValueError(f"Invalid UID: {u!r}")
        return u

    @staticmethod
    def _check_len2(name, arr):
        if arr is None:
            return None
        if len(arr) != 2:
            raise ValueError(f"{name} must have length 2, got {arr}")
        return arr

    # -------------------------------------------------------------------------
    # Safe JSON wrapper
    # -------------------------------------------------------------------------

    def to_json_safe(self, index: int = 0) -> str:
        """
        Return JSON for dicom_data[index], after sanitizing numeric fields so
        pydicom's to_json() doesn't raise ValueError.
        """
        if not self.dicom_data:
            raise ValueError("No DICOM data available")

        ds = self.dicom_data[index]
        # Extra safety: sanitize again in case the dataset came from elsewhere
        self.sanitize_for_json(ds)
        return ds.to_json()


# -------------------------------------------------------------------------
# Example usage (optional)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = DicomParser()
    parser.load_dicoms("V:\\Zhou_Zhongxing\\Zhou_ZX\\For_Jarod\\L067_FD_1_0_B30F_0001")

    patient, study, scanner, series, ct = parser.extract_core()
    print("Patient:", patient)
    print("Series:", series)

    # safe JSON
    json_str = parser.to_json_safe(0)
    print("First slice JSON length:", len(json_str))
