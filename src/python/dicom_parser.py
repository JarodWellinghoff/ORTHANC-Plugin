import re
import os
from datetime import datetime
from pydicom import dcmread, config
from pydicom.multival import MultiValue
from pydicom.errors import InvalidDicomError

# Validation settings
config.enforce_valid_values = False  # raises on invalid VR values


class DicomParser():
    def __init__(self, dicom_data=None) -> None:
        self.dicom_data = dicom_data
        self.NA_SET = {"", "NA", "N/A", None}
        self._uid_re = re.compile(r"^[0-9.]{1,64}$")

    def extract_core(self):
        if not self.dicom_data:
            raise ValueError("No DICOM data available")
        
        ds = self.dicom_data[0]
        # Patient
        patient = {
            "patient_id":   self._norm_text(ds.get("PatientID")),
            "name":         self._norm_text(ds.get("PatientName")),
            "birth_date":   self._parse_date(ds.get("PatientBirthDate")),
            "sex":          self._norm_text(ds.get("PatientSex")),
            "weight_kg":    self._norm_text(ds.get("PatientWeight")),
        }

        # Study
        study = {
            "study_instance_uid": self._validate_uid(ds.StudyInstanceUID),
            "study_id":           self._norm_text(ds.get("StudyID")),
            "accession_number":   self._norm_text(ds.get("AccessionNumber")),
            "study_date":         self._parse_date(ds.get("StudyDate")),
            "study_time":         self._parse_time(ds.get("StudyTime")),
            "description":        self._norm_text(ds.get("StudyDescription")),
            "referring_physician": self._norm_text(ds.get("ReferringPhysicianName")),
            "institution_name":   self._norm_text(ds.get("InstitutionName")),
            "institution_address": self._norm_text(ds.get("InstitutionAddress")),
        }

        # Scanner (dimension)
        scanner = {
            "manufacturer":          self._norm_text(ds.get("Manufacturer")),
            "model_name":            self._norm_text(ds.get("ManufacturerModelName")),
            "device_serial_number":  self._norm_text(ds.get("DeviceSerialNumber")),
            "software_versions":     self._norm_text(ds.get("SoftwareVersions")),
            "station_name":          self._norm_text(ds.get("StationName")),
            "institution_name":      self._norm_text(ds.get("InstitutionName")),
        }

        # Series
        series = {
            "series_instance_uid":    self._validate_uid(ds.get("SeriesInstanceUID")),
            "series_number":          self._norm_text(ds.get("SeriesNumber")),
            "description":            self._norm_text(ds.get("SeriesDescription")),
            "modality":               self._norm_text(ds.get("Modality")),
            "body_part_examined":     self._norm_text(ds.get("BodyPartExamined")),
            "protocol_name":          self._norm_text(ds.get("ProtocolName")),
            "convolution_kernel":     self._norm_text(ds.get("ConvolutionKernel")),
            "patient_position":       self._norm_text(ds.get("PatientPosition")),
            "series_date":            self._parse_date(ds.get("SeriesDate")),
            "series_time":            self._parse_time(ds.get("SeriesTime")),
            "frame_of_reference_uid": self._validate_uid(ds.get("FrameOfReferenceUID")),
            "image_type":             [self._norm_text(x) for x in (ds.get("ImageType") or [])],
            "slice_thickness_mm":     self._norm_text(ds.get("SliceThickness")),
            "pixel_spacing_mm":       self._check_len2("PixelSpacing", self._as_list(ds.get("PixelSpacing"), float)),
            "rows":                   self._norm_text(ds.get("Rows")),
            "columns":                self._norm_text(ds.get("Columns")),
        }

        # CT technique (per-series)
        ct = {
            "kvp":                         self._norm_text(ds.get("KVP")),
            "exposure_time_ms":            self._norm_text(ds.get("ExposureTime")),
            "generator_power_kW":          self._norm_text(ds.get("GeneratorPower")),
            "focal_spots_mm":              self._as_list(ds.get("FocalSpots"), float),
            "filter_type":                 self._norm_text(ds.get("FilterType")),
            "data_collection_diam_mm":     self._norm_text(ds.get("DataCollectionDiameter")),
            "recon_diameter_mm":           self._norm_text(ds.get("ReconstructionDiameter")),
            "dist_src_detector_mm":        self._norm_text(ds.get("DistanceSourceToDetector")),
            "dist_src_patient_mm":         self._norm_text(ds.get("DistanceSourceToPatient")),
            "gantry_detector_tilt_deg":    self._norm_text(ds.get("GantryDetectorTilt")),
            "single_collimation_width_mm": self._norm_text(ds.get("SingleCollimationWidth")),
            "total_collimation_width_mm":  self._norm_text(ds.get("TotalCollimationWidth")),
            "table_speed_mm_s":            self._norm_text(ds.get("TableSpeed")),
            "table_feed_per_rot_mm":       self._norm_text(ds.get("TableFeedPerRotation")),
            "spiral_pitch_factor":         self._norm_text(ds.get("SpiralPitchFactor")),
            "exposure_modulation_type":    self._norm_text(ds.get("ExposureModulationType")),
        }

        # Basic sanity checks
        if series["modality"] and series["modality"] != "CT":
            raise ValueError(f"Not CT modality: {series['modality']}")

        return patient, study, scanner, series, ct

    def load_dicoms(self, dicom_path: str) -> None:
        dicom_data = []
        try:
            for f in os.listdir(dicom_path):
                dcm = dcmread(os.path.join(dicom_path, f))
                if hasattr(dcm, "SliceLocation"):
                    dicom_data.append(dcm)
            self.dicom_data = sorted(dicom_data, key=lambda s: s.SliceLocation)
        except InvalidDicomError:
            self.dicom_data = None

    def _norm_text(self, v):
        if isinstance(v, MultiValue):
            v = "_".join(v)
        if v in self.NA_SET: return None
        s = str(v).strip()
        s = s.replace("\r\n", ", ")
        return None if s in self.NA_SET else s

    def _parse_date(self, v):
        v = self._norm_text(v)
        if not v: return None
        # DICOM DA: YYYYMMDD
        return datetime.strptime(v, "%Y%m%d").date()

    def _parse_time(self, v):
        v = self._norm_text(v)
        if not v: return None
        # DICOM TM: HHMMSS(.ffffff)  ; seconds may be missing
        fmt = "%H%M" if len(v) <= 4 else ("%H%M%S.%f" if "." in v else "%H%M%S")
        return datetime.strptime(v, fmt).time()

    def _as_list(self, v, cast=float):
        if isinstance(v, MultiValue): return [cast(x) for x in v]
        if v in self.NA_SET: return None
        # single value -> one-element list
        return [cast(v)]

    def _validate_uid(self, u):
        u = self._norm_text(u)
        if not u or not self._uid_re.fullmatch(u):
            raise ValueError(f"Invalid UID: {u!r}")
        return u

    @staticmethod
    def _check_len2(name, arr):
        if arr is None: return None
        if len(arr) != 2:
            raise ValueError(f"{name} must have length 2, got {arr}")
        return arr
