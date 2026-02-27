import glob
from pathlib import Path
import pydicom

DIRECTORY = "Y:\\Patient_data\\Force\\HiRes_Routine_Dose_Recons"
NAME_PART = 5


dcm_files = sum(
    (
        glob.glob(f"{DIRECTORY}/**/{pat}", recursive=True)
        for pat in ("*.dcm", "IM00001", "*.ima")
    ),
    [],
)
dcm_files = [Path(dcm_file) for dcm_file in dcm_files]
dcm_files.sort()
patients = list(set([dcm_file.parts[NAME_PART] for dcm_file in dcm_files]))
patients.sort()
patients_dict = [
    {
        "patient": patient,
        "dcm_file": dcm_files[
            [dcm_file.parts[NAME_PART] for dcm_file in dcm_files].index(patient)
        ],
    }
    for patient in patients
]

for patient in patients_dict:
    name = patient["patient"]
    file = patient["dcm_file"]
    ds = pydicom.dcmread(file)
    try:
        kvp = int(ds["KVP"].value)
        print(name)
        print(f"\t{kvp}")
    except (KeyError, ValueError):
        continue
