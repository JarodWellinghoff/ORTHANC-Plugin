import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

mtf_dir = r"C:\Users\M297802\Desktop\MTF Curves"
csv_files = os.listdir(mtf_dir)

headers = [
    "manufacturer",
    "model",
    "kernel",
    "eta_peak",
    "f_peak",
    "f_50",
    "f_10",
    "f_2",
]
rows = []
for csv_file in csv_files:
    with open(os.path.join(mtf_dir, csv_file), "r") as f:
        lines = [line.strip().split(",") for line in f.readlines() if line.strip()]
        manufacturer = lines[1][0]
        model = lines[1][1]
        kernel = lines[1][7]

        f = [float(lines[i][3]) / 10 for i in range(5, len(lines)) if lines[i][3] != ""]
        eta = [float(lines[i][4]) for i in range(5, len(lines)) if lines[i][4] != ""]
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
            manufacturer,
            model,
            kernel,
            eta_peak,
            f_peak,
            f_50,
            f_10,
            f_2,
        ]
        rows.append(row)

df = pd.DataFrame(rows, columns=headers)
print(df)
df.to_csv("mtf_summary.csv", index=False)
