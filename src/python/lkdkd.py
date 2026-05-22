import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mtf_dir = r"C:\Users\M297802\Desktop\MTF Curves"
csv_files = [f for f in os.listdir(mtf_dir) if f.endswith(".csv")]
headers = [
    "Manufacturer",
    "Model",
    "Kernel",
    "SFINIT",
    "MTFINIT",
    "SF100",
    "MTF100",
    "MTF50",
    "MTF10",
    "MTF2",
]
rows = []
for csv_file in csv_files:
    with open(os.path.join(mtf_dir, csv_file), "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines if line.strip()]
        manufacturer = lines[1][0]
        model = lines[1][1]
        kernel = lines[1][7]
        spatial_freq = [
            float(lines[i][3]) / 10 for i in range(5, len(lines)) if lines[i][3] != ""
        ]
        mtf = [float(lines[i][4]) for i in range(5, len(lines)) if lines[i][4] != ""]
        mtf_sp_interp = interp1d(
            spatial_freq, mtf, kind="cubic", fill_value="extrapolate"
        )
        sp_mtf_interp = interp1d(
            mtf, spatial_freq, kind="cubic", fill_value="extrapolate"
        )
        mtf_init = mtf[0]
        mtf_100p = max(mtf)
        mtf_50p = mtf_100p * 0.5
        mtf_10p = mtf_100p * 0.1
        mtf_2p = mtf_100p * 0.02
        mtf_0p = mtf[-1]

        sp_freq_init = spatial_freq[0]
        sp_freq_100p = sp_mtf_interp(mtf_100p)
        sp_freq_50p = sp_mtf_interp(mtf_50p)
        sp_freq_10p = sp_mtf_interp(mtf_10p)
        sp_freq_2p = sp_mtf_interp(mtf_2p)
        sp_freq_0p = spatial_freq[-1]

        if not (sp_freq_100p < sp_freq_50p < sp_freq_10p < sp_freq_2p):
            pass

        row = [
            manufacturer,
            model,
            kernel,
            mtf_init,
            sp_freq_init,
            mtf_100p,
            sp_freq_100p,
            sp_freq_50p,
            sp_freq_10p,
            sp_freq_2p,
        ]
        rows.append(row)

df = pd.DataFrame(rows, columns=headers)
print(df)
# df.to_csv("mtf_summary.csv", index=False)
