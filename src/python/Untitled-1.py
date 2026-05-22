"""
Universal MTF template fitter.

Goal
----
Produce ONE (v_bps, bumps) pair so that, for *any* anchor tuple
    (mtf100, sf100, mtf050, mtf010, mtf002),
calling
    simulate_mtf(fr, mtf100, sf100, mtf050, mtf010, mtf002, V_BPS, BUMPS)
yields a sensible smooth MTF curve --- no per-curve refitting needed.

How the universality works
--------------------------
The model is curve-specific ONLY through:
  * the breakpoint locations (which are the anchor frequencies), and
  * the z_k component shapes (which are functions of the anchors).
The alpha weights (v_bps + bumps) live in *normalized region coordinates*
t = (f - r_l) / (r_h - r_l) per region. They are therefore parameter-free
shape templates: one set of (v_bps, bumps) describes weight behaviour for
all parameter tuples.

z100 redesign
-------------
The previous z100 was
    z100 = (1 - sf100) * cos(pi*f/mtf100) / 2
which gives (1-sf100)/2 at f=0 and (sf100-1)/2 at f=mtf100 -- neither
matches the anchor values 1.0 and sf100. The previous per-curve fit
worked because the alphas absorbed that mismatch, but a universal
template cannot rely on per-curve absorption. The new z100 is a clean
cosine rise that exactly hits both anchors:
    z100(0)       = mtf_dc  (= 1 by default)
    z100(mtf100)  = sf100
    z100(f > mtf100) = sf100  (plateau)
With this, alpha100 = 1 in region 0 produces the correct rising shape
exactly, leaving the optimizer free to focus on subtler details elsewhere.

Workflow
--------
1. Build a list of MeasuredCurve objects from your real CT MTF data.
2. fit_universal_template(curves) returns (V_BPS, BUMPS).
3. np.savez to persist; load and use simulate_mtf(...) anywhere.
"""

from dataclasses import dataclass, field
import os
from typing import List, Optional
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
N_COMPONENTS = 5
N_BPS = 6
N_REGIONS = 5
EPS_DEGENERATE = 1e-4
DEFAULT_PERCENTILES = (0.5, 0.1, 0.02)


# ============================================================
# Model components (curve-specific; redesigned z100)
# ============================================================
def compute_z_components(fr, mtf100, sf100, mtf050, mtf010, mtf002, mtf_dc=1.0):
    """Five MTF shape components.

    z100 (REDESIGNED): cosine rise from mtf_dc at f=0 to sf100 at f=mtf100,
                       then plateau at sf100. Matches anchors exactly.
    z050: constant sf100 (plateau component, useful as a blending neutral).
    z010: power-Gaussian through (mtf050, sf050) and (mtf010, sf010).
    z002: power-Gaussian through (mtf050, sf050) and (mtf002, sf002).
    z000: zero (tail component for decay toward 0).
    """
    sf050 = 0.5 * sf100
    sf010 = 0.1 * sf100
    sf002 = 0.02 * sf100

    L050 = np.log(sf100 / sf050)
    L010 = np.log(sf100 / sf010)
    L002 = np.log(sf100 / sf002)

    d050 = max(mtf050 - mtf100, EPS_DEGENERATE)
    d010 = max(mtf010 - mtf100, EPS_DEGENERATE)
    d002 = max(mtf002 - mtf100, EPS_DEGENERATE)

    n010 = np.log(L010 / L050) / np.log(d010 / d050)
    n002 = np.log(L002 / L050) / np.log(d002 / d050)
    c010 = d050 / (L050 ** (1.0 / n010))
    c002 = d050 / (L050 ** (1.0 / n002))

    fr = np.asarray(fr, dtype=float)

    # --- z100: cosine rise + plateau ---
    if mtf100 > EPS_DEGENERATE:
        t = np.clip(fr / mtf100, 0.0, 1.0)
        rise = mtf_dc + (sf100 - mtf_dc) * (1.0 - np.cos(np.pi * t)) / 2.0
        z100 = np.where(fr <= mtf100, rise, sf100)
    else:
        z100 = np.full_like(fr, sf100)

    z050 = np.full_like(fr, sf100)

    z010 = sf100 * np.exp(-(np.maximum((fr - mtf100) / c010, 0.0) ** n010))
    z010 = np.where(fr < mtf100, sf100, z010)

    z002 = sf100 * np.exp(-(np.maximum((fr - mtf100) / c002, 0.0) ** n002))
    z002 = np.where(fr < mtf100, sf100, z002)

    z000 = np.zeros_like(fr)

    return np.stack([z100, z050, z010, z002, z000])


def cosine_ramp_bump(f, r_l, r_h, v_L, v_R, b):
    """C1-smooth interpolant within a single region (normalized coords)."""
    width = max(r_h - r_l, EPS_DEGENERATE)
    t = np.clip((f - r_l) / width, 0.0, 1.0)
    s = (1.0 - np.cos(np.pi * t)) / 2.0
    bump = b * np.sin(np.pi * t) ** 2
    return v_L + (v_R - v_L) * s + bump


def alpha_component(fr, v_bps_row, bumps_row, r_l_arr, r_h_arr):
    out = np.zeros_like(fr, dtype=float)
    n = len(r_l_arr)
    for i in range(n):
        upper = fr <= r_h_arr[i] if i == n - 1 else fr < r_h_arr[i]
        mask = (fr >= r_l_arr[i]) & upper
        out[mask] = cosine_ramp_bump(
            fr[mask],
            r_l_arr[i],
            r_h_arr[i],
            v_bps_row[i],
            v_bps_row[i + 1],
            bumps_row[i],
        )
    return out


# ============================================================
# The simulator -- the function you'll actually call after fitting
# ============================================================
def simulate_mtf(
    fr, mtf100, sf100, mtf050, mtf010, mtf002, v_bps, bumps, mtf_dc=1.0, f_max=None
):
    """Simulate an MTF curve given anchor parameters + universal template.

    Parameters
    ----------
    fr : array_like
        Frequencies at which to evaluate.
    mtf100, sf100, mtf050, mtf010, mtf002 : float
        Anchor parameters. sf100 is the peak MTF value;
        mtf100 is the frequency at which the peak occurs (0 for non-rising MTFs).
    v_bps : array (N_COMPONENTS, N_BPS)
    bumps : array (N_COMPONENTS, N_REGIONS)
    mtf_dc : float, optional
        MTF value at f=0. Defaults to 1.0 (standard normalization).
    f_max : float, optional
        Frequency at which the model's tail region ends. Defaults to
        max(fr.max(), 1.5*mtf002).
    """
    fr = np.atleast_1d(np.asarray(fr, dtype=float))
    if f_max is None:
        f_max = max(float(fr.max()), mtf002 * 1.5)

    bps = np.array([0.0, mtf100, mtf050, mtf010, mtf002, f_max], dtype=float)
    for i in range(1, len(bps)):
        if bps[i] <= bps[i - 1]:
            bps[i] = bps[i - 1] + EPS_DEGENERATE

    z = compute_z_components(fr, mtf100, sf100, mtf050, mtf010, mtf002, mtf_dc)

    r_l, r_h = bps[:-1], bps[1:]
    alphas = np.stack(
        [alpha_component(fr, v_bps[k], bumps[k], r_l, r_h) for k in range(N_COMPONENTS)]
    )
    alphap = np.maximum(alphas.sum(axis=0), 1e-12)
    return ((alphas / alphap) * z).sum(axis=0)


# ============================================================
# Measured-curve container with automatic anchor detection
# ============================================================
@dataclass
class MeasuredCurve:
    """A measured MTF curve plus its automatically detected anchors.

    Construct from (fr, mtf) arrays; anchors are detected on init.
    Override `anchors` after construction if you want to set them manually.
    """

    fr: np.ndarray
    mtf: np.ndarray
    label: str = ""
    mtf_dc: float = 1.0
    anchors: dict = field(default_factory=dict)

    def __post_init__(self):
        self.fr = np.asarray(self.fr, dtype=float)
        self.mtf = np.asarray(self.mtf, dtype=float)
        if not self.anchors:
            self.anchors = self._detect_anchors()

    def _detect_anchors(self):
        f, m = self.fr, self.mtf
        peak_idx = int(np.argmax(m))
        sf100 = float(m[peak_idx])
        mtf100 = float(f[peak_idx])

        decay_f = f[peak_idx:]
        decay_m = m[peak_idx:]

        crossings = []
        for pct in DEFAULT_PERCENTILES:
            target = pct * sf100
            below = decay_m <= target
            if not below.any():
                # Log-linear extrapolation past data range
                tail_n = min(5, len(decay_m))
                tail_m = np.maximum(decay_m[-tail_n:], 1e-12)
                slope, intercept = np.polyfit(decay_f[-tail_n:], np.log(tail_m), 1)
                if slope < -1e-6 and target > 0:
                    f_target = (np.log(target) - intercept) / slope
                    f_target = float(
                        np.clip(
                            f_target,
                            decay_f[-1] + EPS_DEGENERATE,
                            decay_f[-1] + 3 * (decay_f[-1] - decay_f[0] + 1e-9),
                        )
                    )
                    crossings.append(f_target)
                else:
                    crossings.append(float(decay_f[-1] + EPS_DEGENERATE))
                continue
            j = int(np.argmax(below))
            if j == 0:
                crossings.append(float(decay_f[0]))
            else:
                y0, y1 = decay_m[j - 1], decay_m[j]
                x0, x1 = decay_f[j - 1], decay_f[j]
                if y0 == y1:
                    crossings.append(float(x0))
                else:
                    crossings.append(float(x0 + (target - y0) / (y1 - y0) * (x1 - x0)))
        return {
            "mtf100": mtf100,
            "sf100": sf100,
            "mtf050": crossings[0],
            "mtf010": crossings[1],
            "mtf002": crossings[2],
        }


# ============================================================
# Universal-template fit
# ============================================================
# Initial template: each component dominates exactly where its z_k passes
# through the anchor values, with smooth handoffs in between.
#
# Breakpoint index ->  0     mtf100 mtf050 mtf010 mtf002 f_max
V_BPS_INIT = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # alpha100: dominates [0, mtf100]
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # alpha050: passive
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # alpha010: dominates [mtf100, mtf010]
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # alpha002: dominates [mtf010, mtf002]
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # alpha000: dominates [mtf002, end]
    ]
)

# Logit init: large positive for dominant component, negative for others.
# Softmax of these recovers a near-one-hot column pattern but stays
# differentiable.
LOGIT_SCALE = 6.0
LOGIT_INIT = LOGIT_SCALE * V_BPS_INIT - LOGIT_SCALE / 2.0


def softmax_columns(x):
    """Numerically-stable softmax along axis 0."""
    x_max = np.max(x, axis=0, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=0, keepdims=True)


def logits_to_v_bps(logits):
    """Convert (N_COMPONENTS, N_BPS) logits to columns on the simplex."""
    return softmax_columns(logits)


def fit_universal_template(
    curves: List[MeasuredCurve],
    bump_bound: float = 0.3,
    bump_reg_weight: float = 0.5,
    logit_init: Optional[np.ndarray] = None,
    logit_reg_weight: float = 0.0,
    normalize_per_curve: bool = True,
    max_nfev: int = 300,
    ftol: float = 1e-7,
    verbose: int = 1,
):
    """Jointly fit ONE (v_bps, bumps) across all curves.

    The v_bps are reparameterized through a column-wise softmax of logits
    (no scale ambiguity, simplex-valued by construction). Bumps remain
    unconstrained perturbations within bounds.

    Parameters
    ----------
    bump_bound : float
        |bump| <= bump_bound. Smaller => more universal.
    bump_reg_weight : float
        L2 weight on bumps in the residual vector. Higher => bumps closer to 0.
    logit_reg_weight : float
        Optional L2 weight on (logits - logit_init). Keeps fit near the
        canonical template structure when many curves stress it differently.
    """
    if logit_init is None:
        logit_init = LOGIT_INIT.copy()

    cases = []
    for c in curves:
        a = c.anchors
        params = (a["mtf100"], a["sf100"], a["mtf050"], a["mtf010"], a["mtf002"])
        w = 1.0 / len(c.fr) if normalize_per_curve else 1.0
        cases.append((params, c.fr, c.mtf, c.mtf_dc, w))

    n_l = N_COMPONENTS * N_BPS
    n_b = N_COMPONENTS * N_REGIONS
    logit_init_flat = logit_init.ravel()

    def residuals(x):
        logits = x[:n_l].reshape(N_COMPONENTS, N_BPS)
        v_bps = logits_to_v_bps(logits)
        bumps = x[n_l:].reshape(N_COMPONENTS, N_REGIONS)
        all_r = []
        for params, fr, mtf_meas, mtf_dc, w in cases:
            pred = simulate_mtf(fr, *params, v_bps, bumps, mtf_dc=mtf_dc)
            all_r.append(np.sqrt(w) * (pred - mtf_meas))
        all_r.append(bump_reg_weight * bumps.ravel())
        if logit_reg_weight > 0:
            all_r.append(logit_reg_weight * (x[:n_l] - logit_init_flat))
        return np.concatenate(all_r)

    x0 = np.concatenate([logit_init_flat, np.zeros(n_b)])
    lb = np.concatenate([-15.0 * np.ones(n_l), -bump_bound * np.ones(n_b)])
    ub = np.concatenate([15.0 * np.ones(n_l), bump_bound * np.ones(n_b)])

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=ftol,
        verbose=verbose,
    )
    logits_opt = res.x[:n_l].reshape(N_COMPONENTS, N_BPS)
    bumps_opt = res.x[n_l:].reshape(N_COMPONENTS, N_REGIONS)
    v_bps_opt = logits_to_v_bps(logits_opt)
    return v_bps_opt, bumps_opt, res


# ============================================================
# Per-curve diagnostics
# ============================================================
def curve_stats(curve: MeasuredCurve, v_bps, bumps):
    pred = simulate_mtf(
        curve.fr,
        *(curve.anchors[k] for k in ("mtf100", "sf100", "mtf050", "mtf010", "mtf002")),
        v_bps=v_bps,
        bumps=bumps,
        mtf_dc=curve.mtf_dc,
    )
    err = pred - curve.mtf
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
        "pred": pred,
    }


# ============================================================
# === PLACEHOLDER CORPUS ===
# Replace this whole section with your real measured curves:
#
#   curves = [
#       MeasuredCurve(fr=fr_array_1, mtf=mtf_array_1, label="Br40 routine"),
#       MeasuredCurve(fr=fr_array_2, mtf=mtf_array_2, label="Br44 high-res"),
#       ...
#   ]
#
# The framework auto-detects (mtf100, sf100, mtf050, mtf010, mtf002) per curve.
# You can override by passing anchors=dict(...) to MeasuredCurve.
# ============================================================
ORIGINAL_X = (
    np.array(
        [
            0,
            0.214225941,
            0.428451883,
            0.642677824,
            0.856903766,
            1.071129707,
            1.285355649,
            1.49958159,
            1.713807531,
            1.928033473,
            2.142259414,
            2.356485356,
            2.570711297,
            2.784937238,
            2.99916318,
            3.213389121,
            3.427615063,
            3.641841004,
            3.856066946,
            4.070292887,
            4.284518828,
            4.49874477,
            4.712970711,
            4.927196653,
            5.141422594,
            5.355648536,
            5.569874477,
            5.784100418,
            5.99832636,
            6.212552301,
            6.426778243,
            6.641004184,
            6.855230126,
            7.069456067,
            7.283682008,
            7.49790795,
            7.712133891,
            7.926359833,
            8.140585774,
            8.354811715,
            8.569037657,
            8.783263598,
            8.99748954,
            9.211715481,
            9.425941423,
            9.640167364,
            9.854393305,
            10.06861925,
            10.28284519,
            10.49707113,
            10.71129707,
            10.92552301,
            11.13974895,
            11.3539749,
            11.56820084,
            11.78242678,
            11.99665272,
            12.21087866,
            12.4251046,
            12.63933054,
            12.85355649,
            13.06778243,
            13.28200837,
            13.49623431,
            13.71046025,
            13.92468619,
            14.13891213,
            14.35313808,
            14.56736402,
            14.78158996,
            14.9958159,
            15.21004184,
            15.42426778,
            15.63849372,
            15.85271967,
            16.06694561,
            16.28117155,
            16.49539749,
            16.70962343,
            16.92384937,
            17.13807531,
            17.35230126,
            17.5665272,
            17.78075314,
            17.99497908,
            18.20920502,
            18.42343096,
            18.6376569,
            18.85188285,
            19.06610879,
            19.28033473,
            19.49456067,
            19.70878661,
            19.92301255,
            20.13723849,
            20.35146444,
            20.56569038,
            20.77991632,
            20.99414226,
            21.2083682,
            21.42259414,
            21.63682008,
            21.85104603,
            22.06527197,
            22.27949791,
            22.49372385,
            22.70794979,
            22.92217573,
            23.13640167,
            23.35062762,
            23.56485356,
            23.7790795,
        ]
    )
    / 10.0
)
ORIGINAL_Y = np.array(
    [
        1.003595888,
        1.00575259,
        1.010885573,
        1.01904904,
        1.030193138,
        1.044277676,
        1.061237749,
        1.080110038,
        1.10141589,
        1.124424547,
        1.148782427,
        1.174240251,
        1.199491317,
        1.226088663,
        1.252272209,
        1.277539716,
        1.303242317,
        1.326448752,
        1.348935129,
        1.36847065,
        1.386435409,
        1.402331154,
        1.415514182,
        1.425216462,
        1.432318904,
        1.436166616,
        1.436780988,
        1.434111966,
        1.42844795,
        1.418631042,
        1.405856661,
        1.389491074,
        1.369885043,
        1.348304151,
        1.322101111,
        1.295021123,
        1.263651061,
        1.229773645,
        1.194712313,
        1.157491459,
        1.11608372,
        1.07766997,
        1.035091709,
        0.992611221,
        0.945836773,
        0.903390897,
        0.856302026,
        0.809837148,
        0.762950615,
        0.718742622,
        0.674511456,
        0.628945415,
        0.585478379,
        0.544992339,
        0.500316701,
        0.457992155,
        0.420120382,
        0.380156758,
        0.343024717,
        0.310785915,
        0.276878562,
        0.243549628,
        0.214741701,
        0.185345753,
        0.16008791,
        0.137733964,
        0.1147742,
        0.095311378,
        0.077282246,
        0.061971318,
        0.048788149,
        0.036326938,
        0.027267812,
        0.01936426,
        0.014901,
        0.012204498,
        0.011159317,
        0.011176014,
        0.01151025,
        0.011722516,
        0.011554747,
        0.011064431,
        0.010262285,
        0.00930856,
        0.008461522,
        0.007525365,
        0.006894803,
        0.00661363,
        0.006553003,
        0.006683568,
        0.006874844,
        0.00702946,
        0.007108444,
        0.007105263,
        0.006959542,
        0.006673534,
        0.006270182,
        0.005839408,
        0.005375063,
        0.004955238,
        0.004630795,
        0.004471274,
        0.004393916,
        0.004393456,
        0.004452945,
        0.004527151,
        0.00459226,
        0.004614334,
        0.004588333,
        0.004521455,
        0.004378992,
        0.004188483,
    ]
)


def _placeholder_synthetic(anchors, n_f=120, mtf_dc=1.0):
    """Generate a plausible synthetic CT MTF via PCHIP through anchors.

    Used only as a stand-in until real measured curves are provided.
    """
    f_max = anchors["mtf002"] * 1.4
    fr = np.linspace(0.0, f_max, n_f)
    anchor_f = [
        0.0,
        anchors["mtf100"],
        anchors["mtf050"],
        anchors["mtf010"],
        anchors["mtf002"],
        f_max,
    ]
    anchor_y = [
        mtf_dc,
        anchors["sf100"],
        0.5 * anchors["sf100"],
        0.1 * anchors["sf100"],
        0.02 * anchors["sf100"],
        0.0,
    ]
    for i in range(1, len(anchor_f)):
        if anchor_f[i] <= anchor_f[i - 1]:
            anchor_f[i] = anchor_f[i - 1] + EPS_DEGENERATE
    mtf = PchipInterpolator(anchor_f, anchor_y)(fr)
    return fr, mtf


def _placeholder_corpus():
    """A 6-curve placeholder spanning realistic CT MTF shapes.

    REPLACE THIS with your actual measured curves.
    """
    mtf_dir = r"C:\Users\M297802\Desktop\MTF Curves"
    csv_files = [f for f in os.listdir(mtf_dir) if f.endswith(".csv")]
    real_data = []
    for csv_file in csv_files:
        with open(os.path.join(mtf_dir, csv_file), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split(",") for line in lines if line.strip()]
            make = lines[1][0]
            model = lines[1][1]
            kernel = lines[1][7]
            spatial_freq = np.array(
                [
                    float(lines[i][3]) / 10
                    for i in range(5, len(lines))
                    if lines[i][3] != ""
                ]
            )
            mtf = np.array(
                [float(lines[i][4]) for i in range(5, len(lines)) if lines[i][4] != ""]
            )
            real_data.append(
                MeasuredCurve(
                    spatial_freq, mtf, label="{} {} {}".format(make, model, kernel)
                )
            )

    # synthetic_anchor_sets = [
    #     dict(
    #         mtf100=0.0, sf100=1.0, mtf050=0.55, mtf010=0.95, mtf002=1.15
    #     ),  # sharp standard
    #     dict(
    #         mtf100=0.0, sf100=1.0, mtf050=0.80, mtf010=1.35, mtf002=1.60
    #     ),  # softer standard
    #     dict(
    #         mtf100=0.1, sf100=1.05, mtf050=0.70, mtf010=1.20, mtf002=1.45
    #     ),  # mild rise
    #     dict(
    #         mtf100=0.4, sf100=1.30, mtf050=1.00, mtf010=1.40, mtf002=1.60
    #     ),  # strong rise
    #     dict(
    #         mtf100=0.0, sf100=1.0, mtf050=0.45, mtf010=0.70, mtf002=0.85
    #     ),  # very sharp
    # ]
    # syns = []
    # for i, a in enumerate(synthetic_anchor_sets):
    #     fr, mtf = _placeholder_synthetic(a)
    #     syns.append(MeasuredCurve(fr=fr, mtf=mtf, label=f"Synth #{i + 1}", anchors=a))
    return real_data


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("Building corpus (REPLACE _placeholder_corpus with real curves)...")
    curves = _placeholder_corpus()
    for c in curves:
        a = c.anchors
        print(
            f"  {c.label:35s}  mtf100={a['mtf100']:.3f} sf100={a['sf100']:.3f}  "
            f"mtf050={a['mtf050']:.3f} mtf010={a['mtf010']:.3f} "
            f"mtf002={a['mtf002']:.3f}  n_pts={len(c.fr)}"
        )

    print(f"\nFitting universal template across {len(curves)} curves...")
    t0 = time.time()
    V_BPS, BUMPS, res = fit_universal_template(
        curves, verbose=2, bump_bound=0.5, bump_reg_weight=0.3
    )
    elapsed = time.time() - t0
    print(f"\nFit completed in {elapsed:.1f}s, final cost = {res.cost:.4e}")

    print("\n=== Universal template ===")
    print(
        "V_BPS (rows: alpha100, alpha050, alpha010, alpha002, alpha000;\n"
        "       cols: f=0, mtf100, mtf050, mtf010, mtf002, f_max):"
    )
    with np.printoptions(precision=4, suppress=True):
        print(V_BPS)
    print("\nBUMPS (rows: components; cols: regions):")
    with np.printoptions(precision=4, suppress=True):
        print(BUMPS)

    print(
        "V_BPS (rows: f=0, mtf100, mtf050, mtf010, mtf002, f_max;\n"
        "       cols: alpha100, alpha050, alpha010, alpha002, alpha000):"
    )
    with np.printoptions(precision=4, suppress=True):
        print(V_BPS.T)
    print("\nBUMPS (rows: regions; cols: components):")
    with np.printoptions(precision=4, suppress=True):
        print(BUMPS.T)

    np.savez("universal_template.npz", v_bps=V_BPS, bumps=BUMPS)
    print("\nSaved to universal_template.npz")

    print("\nPer-curve diagnostics:")
    print(f"  {'Curve':<35s} {'RMSE':>10s} {'max|err|':>10s}")
    print("  " + "-" * 60)
    for c in curves:
        s = curve_stats(c, V_BPS, BUMPS)
        print(f"  {c.label:<35s} {s['rmse']:>10.3e} {s['max_abs']:>10.3e}")

    # --- Plot training curves with predictions ---
    n = len(curves)
    # fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n))
    for row, c in enumerate(curves):
        fig, axes = plt.subplots(1, 2, figsize=(13, 2.6))
        s = curve_stats(c, V_BPS, BUMPS)

        # Dense prediction across the curve
        fr_dense = np.linspace(c.fr[0], c.fr[-1], 1500)
        pred_dense = simulate_mtf(
            fr_dense,
            c.anchors["mtf100"],
            c.anchors["sf100"],
            c.anchors["mtf050"],
            c.anchors["mtf010"],
            c.anchors["mtf002"],
            V_BPS,
            BUMPS,
            mtf_dc=c.mtf_dc,
        )
        anchor_f = [
            0,
            c.anchors["mtf100"],
            c.anchors["mtf050"],
            c.anchors["mtf010"],
            c.anchors["mtf002"],
        ]
        anchor_y = [
            c.mtf_dc,
            c.anchors["sf100"],
            0.5 * c.anchors["sf100"],
            0.1 * c.anchors["sf100"],
            0.02 * c.anchors["sf100"],
        ]

        # ax = axes[row, 0] if n > 1 else axes[0]
        ax = axes[0]
        ax.plot(c.fr, c.mtf, "k.", markersize=3, label="measured")
        ax.plot(fr_dense, pred_dense, "b-", linewidth=1.5, label="universal template")
        ax.plot(anchor_f, anchor_y, "ro", markersize=6, label="anchors")
        for af in anchor_f[1:]:
            ax.axvline(af, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(f"{c.label}  RMSE={s['rmse']:.2e}  max|err|={s['max_abs']:.2e}")
        ax.set_xlabel("frequency")
        ax.set_ylabel("MTF")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # alpha curves
        bps_curve = np.array(
            [
                0.0,
                c.anchors["mtf100"],
                c.anchors["mtf050"],
                c.anchors["mtf010"],
                c.anchors["mtf002"],
                max(c.fr[-1], c.anchors["mtf002"] * 1.5),
            ]
        )
        for i in range(1, len(bps_curve)):
            if bps_curve[i] <= bps_curve[i - 1]:
                bps_curve[i] = bps_curve[i - 1] + EPS_DEGENERATE
        r_l, r_h = bps_curve[:-1], bps_curve[1:]
        alphas = np.stack(
            [
                alpha_component(fr_dense, V_BPS[k], BUMPS[k], r_l, r_h)
                for k in range(N_COMPONENTS)
            ]
        )
        alphas /= np.maximum(alphas.sum(axis=0), 1e-12)
        # ax = axes[row, 1] if n > 1 else axes[1]
        ax = axes[1]
        for k, lab in enumerate(
            ["alpha100", "alpha050", "alpha010", "alpha002", "alpha000"]
        ):
            ax.plot(fr_dense, alphas[k], linewidth=1.3, label=lab)
        for af in anchor_f[1:]:
            ax.axvline(af, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(
            "Normalized component weights (same template, " "different anchors)"
        )
        ax.set_xlabel("frequency")
        ax.set_ylabel("weight")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="center right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/{c.label} fit.png", dpi=110)

    # plt.tight_layout()
    # plt.savefig("universal_template_fit.png", dpi=110)
    print("Saved training plot to universal_template_fit.png")

    # --- Held-out generalization test: anchors NOT in training corpus ---
    print("\nGeneralization test (anchors NOT in training):")
    holdout_anchor_sets = [
        dict(mtf100=0.0, sf100=1.0, mtf050=0.65, mtf010=1.10, mtf002=1.35),
        dict(mtf100=0.2, sf100=1.15, mtf050=0.85, mtf010=1.30, mtf002=1.55),
        dict(mtf100=0.0, sf100=1.0, mtf050=0.35, mtf010=0.55, mtf002=0.68),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print(f"  {'Anchor set':<70s} {'RMSE vs PCHIP-truth':>20s}")
    for i, a in enumerate(holdout_anchor_sets):
        fr, truth = _placeholder_synthetic(a)
        pred = simulate_mtf(
            fr,
            a["mtf100"],
            a["sf100"],
            a["mtf050"],
            a["mtf010"],
            a["mtf002"],
            V_BPS,
            BUMPS,
        )
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        desc = (
            f"mtf100={a['mtf100']:.2f} sf100={a['sf100']:.2f} "
            f"050/010/002={a['mtf050']:.2f}/{a['mtf010']:.2f}/{a['mtf002']:.2f}"
        )
        print(f"  {desc:<70s} {rmse:>20.3e}")

        ax = axes[i]
        ax.plot(fr, truth, "k--", label="PCHIP ground truth", linewidth=1.5)
        ax.plot(fr, pred, "b-", label="universal template", linewidth=1.5)
        anchor_f = [0, a["mtf100"], a["mtf050"], a["mtf010"], a["mtf002"]]
        anchor_y = [
            1.0,
            a["sf100"],
            0.5 * a["sf100"],
            0.1 * a["sf100"],
            0.02 * a["sf100"],
        ]
        ax.plot(anchor_f, anchor_y, "ro", markersize=6, label="anchors")
        ax.set_title(f"Holdout #{i + 1}  RMSE={rmse:.2e}")
        ax.set_xlabel("frequency")
        ax.set_ylabel("MTF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("universal_template_holdout.png", dpi=110)
    print("Saved holdout plot to universal_template_holdout.png")
