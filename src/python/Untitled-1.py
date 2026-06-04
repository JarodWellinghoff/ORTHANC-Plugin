"""
Universal MTF template fitter.

Goal
----
Produce ONE (v_bps, bumps) pair so that, for *any* anchor tuple
    (f_peak, eta_peak, f_50, f_10, f_2),
calling
    simulate_mtf(fr, f_peak, eta_peak, f_50, f_10, f_2, V_BPS, BUMPS)
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
    z100 = (1 - eta_peak) * cos(pi*f/f_peak) / 2
which gives (1-eta_peak)/2 at f=0 and (eta_peak-1)/2 at f=f_peak -- neither
matches the anchor values 1.0 and eta_peak. The previous per-curve fit
worked because the alphas absorbed that mismatch, but a universal
template cannot rely on per-curve absorption. The new z100 is a clean
cosine rise that exactly hits both anchors:
    z100(0)       = mtf_dc  (= 1 by default)
    z100(f_peak)  = eta_peak
    z100(f > f_peak) = eta_peak  (plateau)
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
def compute_z_components(fr, f_peak, eta_peak, f_50, f_10, f_2, mtf_dc=1.0):
    """Five MTF shape components.

    z100 (REDESIGNED): cosine rise from mtf_dc at f=0 to eta_peak at f=f_peak,
                       then plateau at eta_peak. Matches anchors exactly.
    z050: constant eta_peak (plateau component, useful as a blending neutral).
    z010: power-Gaussian through (f_50, eta_50) and (f_10, eta_10).
    z002: power-Gaussian through (f_50, eta_50) and (f_2, eta_2).
    z000: zero (tail component for decay toward 0).
    """
    eta_50 = 0.5
    eta_10 = 0.1
    eta_2 = 0.02

    l_50 = np.log(eta_peak / eta_50)
    l_10 = np.log(eta_peak / eta_10)
    l_2 = np.log(eta_peak / eta_2)

    d_50 = max(f_50 - f_peak, EPS_DEGENERATE)
    d_10 = max(f_10 - f_peak, EPS_DEGENERATE)
    d_2 = max(f_2 - f_peak, EPS_DEGENERATE)

    n_10 = np.log(l_10 / l_50) / np.log(d_10 / d_50)
    n_2 = np.log(l_2 / l_50) / np.log(d_2 / d_50)
    c_10 = d_50 / (l_50 ** (1.0 / n_10))
    c_2 = d_50 / (l_50 ** (1.0 / n_2))

    fr = np.asarray(fr, dtype=float)

    # --- z100: cosine rise + plateau ---
    if f_peak > EPS_DEGENERATE:
        t = np.clip(fr / f_peak, 0.0, 1.0)
        rise = mtf_dc + (eta_peak - mtf_dc) * (1.0 - np.cos(np.pi * t)) / 2.0
        z_0 = np.where(fr <= f_peak, rise, eta_peak)
    else:
        z_0 = np.full_like(fr, eta_peak)

    z_1 = np.full_like(fr, eta_peak)

    z_2 = eta_peak * np.exp(-(np.maximum((fr - f_peak) / c_10, 0.0) ** n_10))
    z_2 = np.where(fr < f_peak, eta_peak, z_2)

    z_3 = eta_peak * np.exp(-(np.maximum((fr - f_peak) / c_2, 0.0) ** n_2))
    z_3 = np.where(fr < f_peak, eta_peak, z_3)

    z_4 = np.zeros_like(fr)

    z = np.stack([z_0, z_1, z_2, z_3, z_4])

    return z


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
    fr, f_peak, eta_peak, f_50, f_10, f_2, v_bps, bumps, mtf_dc=1.0, f_max=None
):
    """Simulate an MTF curve given anchor parameters + universal template.

    Parameters
    ----------
    fr : array_like
        Frequencies at which to evaluate.
    f_peak, eta_peak, f_50, f_10, f_2 : float
        Anchor parameters. eta_peak is the peak MTF value;
        f_peak is the frequency at which the peak occurs (0 for non-rising MTFs).
    v_bps : array (N_COMPONENTS, N_BPS)
    bumps : array (N_COMPONENTS, N_REGIONS)
    mtf_dc : float, optional
        MTF value at f=0. Defaults to 1.0 (standard normalization).
    f_max : float, optional
        Frequency at which the model's tail region ends. Defaults to
        max(fr.max(), 1.5*f_2).
    """
    fr = np.atleast_1d(np.asarray(fr, dtype=float))
    if f_max is None:
        f_max = max(float(fr.max()), f_2 * 1.5)

    bps = np.array([0.0, f_peak, f_50, f_10, f_2, f_max], dtype=float)
    for i in range(1, len(bps)):
        if bps[i] <= bps[i - 1]:
            bps[i] = bps[i - 1] + EPS_DEGENERATE

    z = compute_z_components(fr, f_peak, eta_peak, f_50, f_10, f_2, mtf_dc)

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

    Construct from (fr, eta) arrays; anchors are detected on init.
    Override `anchors` after construction if you want to set them manually.
    """

    f: np.ndarray
    eta: np.ndarray
    label: str = ""
    mtf_dc: float = 1.0
    anchors: dict = field(default_factory=dict)

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        self.eta = np.asarray(self.eta, dtype=float)
        if not self.anchors:
            self.anchors = self._detect_anchors()

    def _detect_anchors(self):
        f, eta = self.f, self.eta
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

        return {
            "f_peak": f_peak,
            "eta_peak": eta_peak,
            "f_50": f_50,
            "f_10": f_10,
            "f_2": f_2,
        }


# ============================================================
# Universal-template fit
# ============================================================
# Initial template: each component dominates exactly where its z_k passes
# through the anchor values, with smooth handoffs in between.
#
# Breakpoint index ->  0     f_peak f_50 f_10 f_2 f_max
V_BPS_INIT = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # alpha100: dominates [0, f_peak]
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # alpha050: passive
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # alpha010: dominates [f_peak, f_10]
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # alpha002: dominates [f_10, f_2]
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # alpha000: dominates [f_2, end]
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
        params = (a["f_peak"], a["eta_peak"], a["f_50"], a["f_10"], a["f_2"])
        w = 1.0 / len(c.f) if normalize_per_curve else 1.0
        cases.append((params, c.f, c.eta, c.mtf_dc, w))

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
        curve.f,
        *(curve.anchors[k] for k in ("f_peak", "eta_peak", "f_50", "f_10", "f_2")),
        v_bps=v_bps,
        bumps=bumps,
        mtf_dc=curve.mtf_dc,
    )
    err = pred - curve.eta
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
        "pred": pred,
    }


def _placeholder_synthetic(anchors, n_f=120, mtf_dc=1.0):
    """Generate a plausible synthetic CT MTF via PCHIP through anchors.

    Used only as a stand-in until real measured curves are provided.
    """
    f_max = anchors["f_2"] * 1.4
    fr = np.linspace(0.0, f_max, n_f)
    anchor_f = [
        0.0,
        anchors["f_peak"],
        anchors["f_50"],
        anchors["f_10"],
        anchors["f_2"],
        f_max,
    ]
    anchor_y = [
        mtf_dc,
        anchors["eta_peak"],
        0.5,
        0.1,
        0.02,
        0.0,
    ]
    for i in range(1, len(anchor_f)):
        if anchor_f[i] <= anchor_f[i - 1]:
            anchor_f[i] = anchor_f[i - 1] + EPS_DEGENERATE
    mtf = PchipInterpolator(anchor_f, anchor_y)(fr)
    return fr, mtf


def _corpus():
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
            f = np.array(
                [
                    float(lines[i][3]) / 10
                    for i in range(5, len(lines))
                    if lines[i][3] != ""
                ]
            )
            eta = np.array(
                [float(lines[i][4]) for i in range(5, len(lines)) if lines[i][4] != ""]
            )
            real_data.append(
                MeasuredCurve(f, eta, label="{} {} {}".format(make, model, kernel))
            )
    return real_data


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("Building corpus...")
    curves = _corpus()
    for c in curves:
        a = c.anchors
        print(
            f"  {c.label:35s}  f_peak={a['f_peak']:.3f} eta_peak={a['eta_peak']:.3f}  "
            f"f_50={a['f_50']:.3f} f_10={a['f_10']:.3f} "
            f"f_2={a['f_2']:.3f}  n_pts={len(c.f)}"
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
        "       cols: f=0, f_peak, f_50, f_10, f_2, f_max):"
    )
    with np.printoptions(precision=4, suppress=True):
        print(V_BPS)
    print("\nBUMPS (rows: components; cols: regions):")
    with np.printoptions(precision=4, suppress=True):
        print(BUMPS)

    print(
        "V_BPS (rows: f=0, f_peak, f_50, f_10, f_2, f_max;\n"
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
        fr_dense = np.linspace(c.f[0], c.f[-1], 1500)
        pred_dense = simulate_mtf(
            fr_dense,
            c.anchors["f_peak"],
            c.anchors["eta_peak"],
            c.anchors["f_50"],
            c.anchors["f_10"],
            c.anchors["f_2"],
            V_BPS,
            BUMPS,
            mtf_dc=c.mtf_dc,
        )
        anchor_f = [
            0,
            c.anchors["f_peak"],
            c.anchors["f_50"],
            c.anchors["f_10"],
            c.anchors["f_2"],
        ]
        anchor_y = [
            c.mtf_dc,
            c.anchors["eta_peak"],
            0.5,
            0.1,
            0.02,
        ]

        # ax = axes[row, 0] if n > 1 else axes[0]
        ax = axes[0]
        ax.plot(c.f, c.eta, "k.", markersize=3, label="measured")
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
                c.anchors["f_peak"],
                c.anchors["f_50"],
                c.anchors["f_10"],
                c.anchors["f_2"],
                max(c.f[-1], c.anchors["f_2"] * 1.5),
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
        plt.close(fig)
        print(f"Saved training plot for {c.label} to figures/{c.label} fit.png")

    # plt.tight_layout()
    # plt.savefig("universal_template_fit.png", dpi=110)
    print("Saved training plot to universal_template_fit.png")

    # --- Held-out generalization test: anchors NOT in training corpus ---
    print("\nGeneralization test (anchors NOT in training):")
    holdout_anchor_sets = [
        dict(f_peak=0.0, eta_peak=1.0, f_50=0.65, f_10=1.10, f_2=1.35),
        dict(f_peak=0.2, eta_peak=1.15, f_50=0.85, f_10=1.30, f_2=1.55),
        dict(f_peak=0.0, eta_peak=1.0, f_50=0.35, f_10=0.55, f_2=0.68),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    print(f"  {'Anchor set':<70s} {'RMSE vs PCHIP-truth':>20s}")
    for i, a in enumerate(holdout_anchor_sets):
        fr, truth = _placeholder_synthetic(a)
        pred = simulate_mtf(
            fr,
            a["f_peak"],
            a["eta_peak"],
            a["f_50"],
            a["f_10"],
            a["f_2"],
            V_BPS,
            BUMPS,
        )
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        desc = (
            f"f_peak={a['f_peak']:.2f} eta_peak={a['eta_peak']:.2f} "
            f"050/010/002={a['f_50']:.2f}/{a['f_10']:.2f}/{a['f_2']:.2f}"
        )
        print(f"  {desc:<70s} {rmse:>20.3e}")

        ax = axes[i]
        ax.plot(fr, truth, "k--", label="PCHIP ground truth", linewidth=1.5)
        ax.plot(fr, pred, "b-", label="universal template", linewidth=1.5)
        anchor_f = [0, a["f_peak"], a["f_50"], a["f_10"], a["f_2"]]
        anchor_y = [
            1.0,
            a["eta_peak"],
            0.5,
            0.1,
            0.02,
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
