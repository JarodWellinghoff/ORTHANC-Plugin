import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import glob
import json

HEADERS = [
    "manufacturer",
    "model",
    "kernel",
    "eta_peak",
    "f_peak",
    "f_50",
    "f_10",
    "f_2",
]
MANUFACTURER_MAP = {
    "siemens healthineers": "Siemens",
    "siemens": "Siemens",
    "toshiba": "Canon",
}
MODEL_MAP = {
    "naeotom alpha": "Alpha",
    "somatom force": "Force",
    "aquilion prime sp": "Aquilion Prime",
}

FILE_TYPE = "json"  # "csv" or "json"


def main():
    if FILE_TYPE == "csv":
        mtf_dir = r"C:\Users\M297802\Desktop\MTF Curves"
        mtf_files = os.listdir(mtf_dir)
        parse_xls_files(mtf_dir, mtf_files)
    elif FILE_TYPE == "json":
        mtf_dir = r"W:\mtf_reports"
        mtf_files = glob.glob(os.path.join(mtf_dir, "**", "*.json"), recursive=True)
        parse_json_files(mtf_files)
    else:
        raise ValueError(f"Unsupported FILE_TYPE: {FILE_TYPE}")


def parse_json_files(mtf_files):
    rows = []
    for json_file in mtf_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            manufacturer = data["dcm_info"]["Manufacturer"].lower()
            model = data["dcm_info"]["ModelName"].lower()
            kernel = "_".join(data["dcm_info"]["ConvolutionKernel"])
            rois = data["rois"]
            optimal_roi = max(rois, key=lambda r: r["is_optimal"])
            metrics = optimal_roi["metrics"]

            f_peak = metrics["f_peak"]["radial"]
            eta_peak = metrics["eta_peak"]["radial"]
            f_50 = metrics["f_50"]["radial"]
            f_10 = metrics["f_10"]["radial"]
            f_2 = metrics["f_2"]["radial"]

            row = [
                MANUFACTURER_MAP.get(manufacturer, manufacturer),
                MODEL_MAP.get(model, model),
                kernel,
                eta_peak,
                f_peak,
                f_50,
                f_10,
                f_2,
            ]
            rows.append(row)
    save_results(rows)


def parse_xls_files(mtf_dir, mtf_files):
    rows = []
    for csv_file in mtf_files:
        with open(os.path.join(mtf_dir, csv_file), "r") as f:
            lines = [line.strip().split(",") for line in f.readlines() if line.strip()]
            manufacturer = lines[1][0].lower()
            model = lines[1][1].lower()
            kernel = lines[1][7]

            f = [
                float(lines[i][3]) / 10
                for i in range(5, len(lines))
                if lines[i][3] != ""
            ]
            eta = [
                float(lines[i][4]) for i in range(5, len(lines)) if lines[i][4] != ""
            ]
            t = np.linspace(0, 1, len(f))

            t_new = np.linspace(0, 1, 100001)
            f_new = np.interp(t_new, t, f)
            eta_new = np.interp(t_new, t, eta)

            eta_50 = 0.5
            eta_10 = 0.1
            eta_2 = 0.02

            idx_peak = eta_new.argmax()
            idx_50 = np.abs(eta_new - eta_50).argmin()
            idx_10 = np.abs(eta_new - eta_10).argmin()
            idx_2 = np.abs(eta_new - eta_2).argmin()

            f_peak = f_new[idx_peak]
            f_50 = f_new[idx_50]
            f_10 = f_new[idx_10]
            f_2 = f_new[idx_2]

            if f_peak == 0:
                eta_peak = 1
            else:
                eta_peak = max(eta_new)

            row = [
                MANUFACTURER_MAP.get(manufacturer, manufacturer),
                MODEL_MAP.get(model, model),
                kernel,
                eta_peak,
                f_peak,
                f_50,
                f_10,
                f_2,
            ]
            rows.append(row)
    save_results(rows)


def save_results(rows):
    df = pd.DataFrame(rows, columns=HEADERS)
    print(df)
    df.to_csv(f"mtf_summary_{FILE_TYPE}.csv", index=False)


if __name__ == "__main__":
    main()
