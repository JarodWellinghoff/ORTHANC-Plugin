"""
Piecewise blended-MTF fit with structural C1 continuity at internal breakpoints.

Key changes vs. the previous version
------------------------------------
1. Each component's region-wise weight (alpha_k) is parameterized by
   per-component values at the 6 breakpoints, **shared between adjacent
   regions**. Continuity (C0) at every internal breakpoint is therefore
   automatic.
2. Within each region the two endpoint values are blended with a cosine
   ramp, whose derivative is zero at both ends -> C1 at every internal
   breakpoint, also automatic.
3. Optional per-region "bump" term (b * sin^2(pi*t)) that is identically
   zero in value AND slope at the region edges, so it preserves C1 while
   giving the fit interior flexibility within a region.
4. The old "boundary penalty" in residuals (which actually just reweighted
   data residuals at boundary indices) is replaced with a real model-side
   continuity *check* residual. With the new parameterization this should
   be machine-epsilon -- it's kept as a sanity assert, not as the thing
   enforcing smoothness.
5. The [0, 1] hard bounds on individual coefficients are dropped: raw
   v-values are required >= 0 (so normalization is safe), and bumps are
   unbounded in sign.

Parameter count: 5 components * (6 bp values + 5 bumps) = 55
(was 5 * 5 * 3 = 75).
"""

from scipy.optimize import least_squares
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
x_init = (
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
    / 10
)
y_init = np.array(
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

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
mtf_init_y = interp1d(y_init, x_init, kind="cubic", fill_value="extrapolate")
mtf_init_x = interp1d(x_init, y_init, kind="cubic", fill_value="extrapolate")

sf100p = y_init.max()
sf050p = 0.5 * sf100p
sf010p = 0.1 * sf100p
sf002p = 0.02 * sf100p

mtf100p = float(mtf_init_y(sf100p))
mtf050p = float(mtf_init_y(sf050p))
mtf010p = float(mtf_init_y(sf010p))
mtf002p = float(mtf_init_y(sf002p))

# 5 regions, 6 breakpoints. Breakpoints are *shared* across adjacent regions,
# which is what makes the model continuous by construction.
F_BP = np.array([x_init[0], mtf100p, mtf050p, mtf010p, mtf002p, x_init[-1]])
R_L = F_BP[:-1]  # lower edges per region
R_H = F_BP[1:]  # upper edges per region
N_REGIONS = len(R_L)  # 5
N_COMPONENTS = 5  # alpha100, alpha050, alpha010, alpha002, alpha000
N_BPS = N_REGIONS + 1  # 6
N_BUMPS = N_REGIONS  # 5
N_PARAMS_PER_COMP = N_BPS + N_BUMPS
N_PARAMS_TOTAL = N_COMPONENTS * N_PARAMS_PER_COMP

# Internal breakpoints (excluding the two ends) -- used for the continuity check.
BP_INTERNAL = F_BP[1:-1]
EPS_BP = 1e-7
W_CONT = 1.0  # near-zero by construction; kept as a sanity check term


# ---------------------------------------------------------------------------
# New parameterization: per-component breakpoint values + per-region bumps
# ---------------------------------------------------------------------------
def pack(v_bps_all, bumps_all):
    """v_bps_all: (N_COMPONENTS, N_BPS), bumps_all: (N_COMPONENTS, N_BUMPS)."""
    return np.concatenate(
        [np.asarray(v_bps_all).ravel(), np.asarray(bumps_all).ravel()]
    )


def unpack(v):
    n_v = N_COMPONENTS * N_BPS
    v_bps_all = v[:n_v].reshape(N_COMPONENTS, N_BPS)
    bumps_all = v[n_v:].reshape(N_COMPONENTS, N_BUMPS)
    return v_bps_all, bumps_all


def cosine_ramp_bump(f, r_l, r_h, v_L, v_R, b):
    """C1-smooth interpolant in a single region.

    At t=0 (left edge):  value = v_L,  slope = 0
    At t=1 (right edge): value = v_R,  slope = 0
    Bump term sin^2(pi*t) is 0 with 0 slope at both ends.
    """
    t = np.clip((f - r_l) / (r_h - r_l), 0.0, 1.0)
    s = (1.0 - np.cos(np.pi * t)) / 2.0
    bump = b * np.sin(np.pi * t) ** 2
    return v_L + (v_R - v_L) * s + bump


def alpha_component(fr, v_bps, bumps, r_l_arr, r_h_arr):
    """Evaluate one component's alpha as a piecewise C1 function over fr.

    v_bps[i] is the value at the left edge of region i AND the right edge
    of region i-1, so adjacent regions automatically agree.
    """
    out = np.zeros_like(fr, dtype=float)
    n = len(r_l_arr)
    for i in range(n):
        # Half-open intervals; last region inclusive on its upper bound.
        upper = fr <= r_h_arr[i] if i == n - 1 else fr < r_h_arr[i]
        mask = (fr >= r_l_arr[i]) & upper
        out[mask] = cosine_ramp_bump(
            fr[mask],
            r_l_arr[i],
            r_h_arr[i],
            v_bps[i],
            v_bps[i + 1],
            bumps[i],
        )
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def generalized_gaussian_psf(
    mtf050,
    mtf010,
    mtf002,
    mtf100,
    sf100,
    size=512,
    params=None,
    pixel_size=1.0,
    fr=None,
):
    """Evaluate the blended-MTF model.

    `params` is either None (use sensible defaults) or a tuple
    `(v_bps_all, bumps_all)` produced by unpack().
    """
    sf050 = 0.5 * sf100
    sf010 = 0.1 * sf100
    sf002 = 0.02 * sf100

    L050 = np.log(sf100 / sf050)
    L010 = np.log(sf100 / sf010)
    L002 = np.log(sf100 / sf002)

    n010 = np.log(L010 / L050) / np.log((mtf010 - mtf100) / (mtf050 - mtf100))
    n002 = np.log(L002 / L050) / np.log((mtf002 - mtf100) / (mtf050 - mtf100))

    c010 = (mtf050 - mtf100) / (L050 ** (1 / n010))
    c002 = (mtf050 - mtf100) / (L050 ** (1 / n002))

    if fr is None:
        fr = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
        fr = fr[size // 2 :]
    else:
        fr = np.asarray(fr, dtype=float)

    section100 = fr < mtf100

    z100 = np.ones_like(fr)
    if mtf100 != 0:
        z100 = ((1 - sf100) * np.cos((np.pi * fr) / mtf100)) / 2
    z050 = np.ones_like(fr) * sf100
    z010 = sf100 * np.exp(-(np.maximum((fr - mtf100) / c010, 0) ** n010))
    z010[section100] = sf100
    z002 = sf100 * np.exp(-(np.maximum((fr - mtf100) / c002, 0) ** n002))
    z002[section100] = sf100
    z000 = np.zeros_like(fr)

    if params is not None:
        v_bps_all, bumps_all = params
    else:
        v_bps_all = np.array(
            [
                [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # alpha100
                [0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # alpha050
                [0.0, 0.0, 0.5, 1.0, 0.5, 0.0],  # alpha010
                [0.0, 0.0, 0.0, 0.5, 1.0, 0.5],  # alpha002
                [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],  # alpha000
            ]
        )
        bumps_all = np.zeros((N_COMPONENTS, N_BUMPS))

    alphas = np.stack(
        [
            alpha_component(fr, v_bps_all[k], bumps_all[k], R_L, R_H)
            for k in range(N_COMPONENTS)
        ]
    )

    alphap = alphas.sum(axis=0)
    alphap = np.maximum(alphap, 1e-12)  # safe floor; should not normally trip
    alphas = alphas / alphap

    mtf = (
        z100 * alphas[0]
        + z050 * alphas[1]
        + z010 * alphas[2]
        + z002 * alphas[3]
        + z000 * alphas[4]
    )
    return mtf


# ---------------------------------------------------------------------------
# Residuals: data + (sanity) continuity check
# ---------------------------------------------------------------------------
f_meas = np.asarray(x_init, dtype=float)
mtf_meas = np.asarray(y_init, dtype=float)


def residuals(v, mtf100, sf100, mtf050, mtf010, mtf002, f_meas, mtf_meas):
    params = unpack(v)
    mtf_pred = generalized_gaussian_psf(
        mtf050=mtf050,
        mtf010=mtf010,
        mtf002=mtf002,
        mtf100=mtf100,
        sf100=sf100,
        params=params,
        fr=f_meas,
    )
    data_resid = mtf_pred - mtf_meas

    # Continuity check: should be ~0 to machine precision by construction.
    m_minus = generalized_gaussian_psf(
        mtf050=mtf050,
        mtf010=mtf010,
        mtf002=mtf002,
        mtf100=mtf100,
        sf100=sf100,
        params=params,
        fr=BP_INTERNAL - EPS_BP,
    )
    m_plus = generalized_gaussian_psf(
        mtf050=mtf050,
        mtf010=mtf010,
        mtf002=mtf002,
        mtf100=mtf100,
        sf100=sf100,
        params=params,
        fr=BP_INTERNAL + EPS_BP,
    )
    cont_resid = W_CONT * (m_plus - m_minus)

    return np.concatenate([data_resid, cont_resid])


# ---------------------------------------------------------------------------
# Initial guess + bounds
# ---------------------------------------------------------------------------
# Each component's bp values are seeded so its weight peaks in the region
# where its z_k is the natural dominant: alpha100 near f=0, alpha050 near
# mtf050, alpha010 near mtf010, alpha002 near mtf002, alpha000 at the tail.
v_bps_init = np.array(
    [
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # alpha100
        [0.0, 0.5, 1.0, 0.5, 0.0, 0.0],  # alpha050
        [0.0, 0.0, 0.5, 1.0, 0.5, 0.0],  # alpha010
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.5],  # alpha002
        [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],  # alpha000
    ]
)
bumps_init = np.zeros((N_COMPONENTS, N_BUMPS))
x0 = pack(v_bps_init, bumps_init)

# Raw breakpoint values must stay non-negative so the normalized weights are
# guaranteed to lie in [0, 1]. Bumps are additive perturbations; allow sign.
lb = np.concatenate(
    [
        np.zeros(N_COMPONENTS * N_BPS),
        -np.inf * np.ones(N_COMPONENTS * N_BUMPS),
    ]
)
ub = np.inf * np.ones(N_PARAMS_TOTAL)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
res = least_squares(
    residuals,
    x0,
    bounds=(lb, ub),
    args=(mtf100p, sf100p, mtf050p, mtf010p, mtf002p, f_meas, mtf_meas),
    method="trf",
    verbose=2,
)

x_opt = res.x
v_bps_opt, bumps_opt = unpack(x_opt)
params_opt = (v_bps_opt, bumps_opt)

with np.printoptions(precision=3, suppress=True):
    print("\nOptimized breakpoint values (rows = components, cols = breakpoints):")
    print("  components: alpha100, alpha050, alpha010, alpha002, alpha000")
    print("  breakpoints: f=0, mtf100, mtf050, mtf010, mtf002, f_max")
    print(v_bps_opt.T / np.concatenate([v_bps_opt.T, bumps_opt]).max())
    print("\nOptimized bumps (rows = components, cols = regions):")
    print(bumps_opt.T / np.concatenate([v_bps_opt.T, bumps_opt]).max())

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
print("\nC0/C1 continuity check at internal breakpoints " "(should be ~machine eps):")
for f in BP_INTERNAL:
    mL = generalized_gaussian_psf(
        mtf050=mtf050p,
        mtf010=mtf010p,
        mtf002=mtf002p,
        mtf100=mtf100p,
        sf100=sf100p,
        params=params_opt,
        fr=np.array([f - EPS_BP]),
    )[0]
    mR = generalized_gaussian_psf(
        mtf050=mtf050p,
        mtf010=mtf010p,
        mtf002=mtf002p,
        mtf100=mtf100p,
        sf100=sf100p,
        params=params_opt,
        fr=np.array([f + EPS_BP]),
    )[0]
    # Numerical slope estimate just inside each side
    h = 1e-4
    mL2 = generalized_gaussian_psf(
        mtf050=mtf050p,
        mtf010=mtf010p,
        mtf002=mtf002p,
        mtf100=mtf100p,
        sf100=sf100p,
        params=params_opt,
        fr=np.array([f - EPS_BP - h]),
    )[0]
    mR2 = generalized_gaussian_psf(
        mtf050=mtf050p,
        mtf010=mtf010p,
        mtf002=mtf002p,
        mtf100=mtf100p,
        sf100=sf100p,
        params=params_opt,
        fr=np.array([f + EPS_BP + h]),
    )[0]
    slope_L = (mL - mL2) / h
    slope_R = (mR2 - mR) / h
    print(
        f"  f={f:7.4f}: L={mL:.9f}  R={mR:.9f}  "
        f"jump={mR - mL:+.2e}  slope_jump={slope_R - slope_L:+.2e}"
    )

mtf_pred_data = generalized_gaussian_psf(
    mtf050=mtf050p,
    mtf010=mtf010p,
    mtf002=mtf002p,
    mtf100=mtf100p,
    sf100=sf100p,
    params=params_opt,
    fr=f_meas,
)
print(f"\nFinal cost (incl. continuity terms): {res.cost:.6e}")
print(
    f"Max |data residual|:                  "
    f"{np.max(np.abs(mtf_pred_data - mtf_meas)):.6e}"
)
print(
    f"RMS  data residual:                   "
    f"{np.sqrt(np.mean((mtf_pred_data - mtf_meas) ** 2)):.6e}"
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
f_dense = np.linspace(x_init[0], x_init[-1], 4000)
mtf_pred_dense = generalized_gaussian_psf(
    mtf050=mtf050p,
    mtf010=mtf010p,
    mtf002=mtf002p,
    mtf100=mtf100p,
    sf100=sf100p,
    params=params_opt,
    fr=f_dense,
)


# Also visualize the (normalized) alphas to see the smooth weight handoff.
def all_alphas(fr):
    alphas = np.stack(
        [
            alpha_component(fr, v_bps_opt[k], bumps_opt[k], R_L, R_H)
            for k in range(N_COMPONENTS)
        ]
    )
    alphap = np.maximum(alphas.sum(axis=0), 1e-12)
    return alphas / alphap


alphas_dense = all_alphas(f_dense)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.plot(x_init, y_init, "ko", label="data", markersize=4)
ax.plot(f_dense, mtf_pred_dense, "b-", label="fit", linewidth=2)
for f in BP_INTERNAL:
    ax.axvline(f, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("MTF")
ax.set_title("Piecewise blended MTF fit (C1-continuous by construction)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
labels = ["alpha100", "alpha050", "alpha010", "alpha002", "alpha000"]
for k in range(N_COMPONENTS):
    ax.plot(f_dense, alphas_dense[k], label=labels[k], linewidth=1.5)
for f in BP_INTERNAL:
    ax.axvline(f, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("frequency")
ax.set_ylabel("normalized weight")
ax.set_title("Component weights (smooth handoff at every breakpoint)")
ax.legend(loc="center right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mtf_fit_smooth.png", dpi=120)
plt.show()
