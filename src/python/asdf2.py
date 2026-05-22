import pandas as pd
import json

force_json = "./asdf.json"

with open(force_json, "r") as file:
    force_data = json.load(file)

imgArrs = [f.get("imgArr")[0] for f in force_data]
mtfDatas = [f.get("mtfData") for f in force_data]
psfDatas = [f.get("psfData") for f in force_data]
mtfDatasets = [m.get("datasets")[0] for m in mtfDatas]
mtfDatasets_x = [d.get("x") for d in mtfDatasets]
mtfDatasets_y = [d.get("y") for d in mtfDatasets]

col_names = [
    "Manufacturer",
    "Model",
    "Kernel",
    "MTF100",
    "SF100",
    "MTF50",
    "MTF10",
    "MTF2",
    "PSFFWHM",
]
manufacturers = [i.split("/")[6] for i in imgArrs]
models = [i.split("/")[7] for i in imgArrs]
kernels = [
    i.split("/")[10].split("_")[0].replace("T3D-", "").replace("-", "_")
    for i in imgArrs
]
mtf100s = [max(m) for m in mtfDatasets_y]
sf100s = [x[y.index(m)] for x, y, m in zip(mtfDatasets_x, mtfDatasets_y, mtf100s)]
mtf50s = [m.get("mtf50") for m in mtfDatas]
mtf10s = [m.get("mtf10") for m in mtfDatas]
mtf2s = [m.get("mtf2") for m in mtfDatas]
psffwhms = [p.get("psffwhm") for p in psfDatas]
df = pd.DataFrame(
    {
        "Manufacturer": manufacturers,
        "Model": models,
        "Kernel": kernels,
        "MTF100": mtf100s,
        "SF100": sf100s,
        "MTF50": mtf50s,
        "MTF10": mtf10s,
        "MTF2": mtf2s,
        "PSFFWHM": psffwhms,
    },
    columns=col_names,
)
df.to_csv("patient_iq_data.csv", index=False)
