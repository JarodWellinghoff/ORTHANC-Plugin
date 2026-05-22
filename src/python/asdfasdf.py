from scipy.optimize import least_squares
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

mtf100p = mtf_init_y(sf100p)
mtf050p = mtf_init_y(sf050p)
mtf010p = mtf_init_y(sf010p)
mtf002p = mtf_init_y(sf002p)

idx100p = (x_init > mtf100p).argmax()
idx050p = (x_init > mtf050p).argmax()
idx010p = (x_init > mtf010p).argmax()
idx002p = (x_init > mtf002p).argmax()
idxp = np.array([0, idx100p, idx050p, idx010p, idx002p, len(x_init) - 1])
idx_eps_upper = idxp + 1
idx_eps_lower = idxp - 1
idx_eps = np.concatenate([idx_eps_lower, idxp, idx_eps_upper]).clip(0, len(x_init) - 1)
idx_eps = np.unique(idx_eps)

# Region partition used by the piecewise alpha model. Hoisted to the module
# scope so we can subset measurements per region for independent fits.
mtf_lower_bounds = np.array([x_init[0], mtf100p, mtf050p, mtf010p, mtf002p])
sf_lower_bounds = np.array([y_init[0], sf100p, sf050p, sf010p, sf002p])
mtf_upper_bounds = np.array([mtf100p, mtf050p, mtf010p, mtf002p, x_init[-1]])
sf_upper_bounds = np.array([sf100p, sf050p, sf010p, sf002p, y_init[-1]])


# Boundary-continuity penalty.
# - W_BOUNDARY scales the boundary residuals relative to the data residuals
#   (squared in the cost). 100 makes a 0.01 boundary mismatch cost roughly
#   the same as a 1.0 data mismatch; raise to 1e3+ to enforce more strictly.
W_BOUNDARY = 1e0
BOUNDARY_EPS = 1e-9  # step inward from upper bound to probe boundary line-up

mtf_bounds = np.array([mtf100p, mtf050p, mtf010p, mtf002p])
mtf_bounds_eps_upper = (mtf_bounds + BOUNDARY_EPS).clip(mtf_bounds[0], mtf_bounds[-1])
mtf_bounds_eps_lower = (mtf_bounds - BOUNDARY_EPS).clip(mtf_bounds[0], mtf_bounds[-1])
mtf_bounds_eps = np.concatenate(
    [mtf_bounds_eps_lower, mtf_bounds, mtf_bounds_eps_upper]
)
mtf_init_eps = mtf_init_x(mtf_bounds_eps)

param100 = np.random.rand(5, 3)
param050 = np.random.rand(5, 3)
param010 = np.random.rand(5, 3)
param002 = np.random.rand(5, 3)
param000 = np.random.rand(5, 3)


def pack(p100, p050, p010, p002, p000):
    return np.concatenate(
        [np.asarray(x).ravel() for x in (p100, p050, p010, p002, p000)]
    )


x0 = pack(param100, param050, param010, param002, param000)
zz = np.zeros_like(x0)
f_meas = np.asarray(x_init, dtype=float)
mtf_meas = np.asarray(y_init, dtype=float)
min_idx = 0


def unpack(v):
    return v.reshape(5, 5, 3)


def residuals(v, mtf100, sf100, mtf050, mtf010, mtf002, f_meas, mtf_meas):
    """Data residuals plus boundary line-up residuals for region i.

    Returned vector has length len(f_meas) + (1 or 2). The boundary entries
    are weighted by W_BOUNDARY so a tight line-up is encouraged when this is
    minimized in least-squares sense.
    """
    p = unpack(v)

    mtf_pred = generalized_gaussian_psf(
        mtf050=mtf050,
        mtf010=mtf010,
        mtf002=mtf002,
        mtf100=mtf100,
        sf100=sf100,
        params=p,
        fr=f_meas,
    )
    # mtf_pred_interp = interp1d(f_meas, mtf_pred, kind="cubic", fill_value="extrapolate")
    data_resid = mtf_pred - mtf_meas
    data_resid[idx_eps] = W_BOUNDARY * (mtf_pred[idx_eps] - mtf_meas[idx_eps])
    return data_resid


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

    If `fr` is supplied, the model is evaluated only at those frequencies and
    no FFT-based grid is built. Otherwise a non-negative FFT frequency axis
    of length `size//2` with spacing `pixel_size` is constructed.
    """

    def _t(r, r_l, r_h):
        return np.clip((r - r_l) / (r_h - r_l), 0.0, 1.0)

    def e_sine_in(r, a, r_l, r_h):
        return a * (1 - np.cos(np.pi * _t(r, r_l, r_h))) / 2

    def e_sine_out(r, a, r_l, r_h):
        return a * (1 + np.cos(np.pi * _t(r, r_l, r_h))) / 2

    def e_all(r, a, r_l, r_h):
        return a[0] + e_sine_in(r, a[1], r_l, r_h) + e_sine_out(r, a[2], r_l, r_h)

    def piecewise(fr, a, r_l, r_h):
        """Apply e_all per region. Coef/bound args are length-N lists."""
        out = np.zeros_like(fr, dtype=float)
        n = len(r_l)
        for i in range(n):
            # half-open intervals; last region inclusive on upper bound
            upper = fr <= r_h[i] if i == n - 1 else fr < r_h[i]
            mask = (fr >= r_l[i]) & upper
            out[mask] = e_all(fr[mask], a[i], r_l[i], r_h[i])
        return out

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
        fr = fr[size // 2 :]  # non-negative half
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
    lower_bounds = [0, mtf100, mtf050, mtf010, mtf002]
    upper_bounds = [mtf100, mtf050, mtf010, mtf002, 10]

    if params is not None:
        param100, param050, param010, param002, param000 = params
    else:
        param100 = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        param050 = [[0, 0, 0, 0, 0], [0, 0.15, 0, 0, 0], [0, 0, 0.15, 0, 0]]
        param010 = [[0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]
        param002 = [[0, 1, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]
        param000 = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    alpha100 = piecewise(fr, param100, lower_bounds, upper_bounds)
    alpha050 = piecewise(fr, param050, lower_bounds, upper_bounds)
    alpha010 = piecewise(fr, param010, lower_bounds, upper_bounds)
    alpha002 = piecewise(fr, param002, lower_bounds, upper_bounds)
    alpha000 = piecewise(fr, param000, lower_bounds, upper_bounds)

    alphap = alpha100 + alpha050 + alpha010 + alpha002 + alpha000

    alpha100 /= alphap
    alpha050 /= alphap
    alpha010 /= alphap
    alpha002 /= alphap
    alpha000 /= alphap

    mtf = (
        z100 * alpha100
        + z050 * alpha050
        + z010 * alpha010
        + z002 * alpha002
        + z000 * alpha000
    )
    return mtf


res = least_squares(
    residuals,
    x0,
    bounds=(np.zeros_like(x0), np.ones_like(x0)),
    args=(mtf100p, sf100p, mtf050p, mtf010p, mtf002p, f_meas, mtf_meas),
    method="trf",
    verbose=2,
)

x_opt = res.x

p100_opt, p050_opt, p010_opt, p002_opt, p000_opt = unpack(x_opt)
with np.printoptions(precision=3, suppress=True):
    print("Optimized parameters:")
    print(f"p100: \n{p100_opt}")
    print(f"p050: \n{p050_opt}")
    print(f"p010: \n{p010_opt}")
    print(f"p002: \n{p002_opt}")
    print(f"p000: \n{p000_opt}")
# ---------------------------------------------------------------------------
# Diagnostic: how well do region edges line up after fitting?
# ---------------------------------------------------------------------------
print("\nBoundary line-up check (model value vs. canonical target):")
p_opt = unpack(x_opt)
for i in range(5):
    edges = [(mtf_lower_bounds[i], sf_lower_bounds[i], "lower")]
    if i < len(mtf_upper_bounds) - 1:
        edges.append((mtf_upper_bounds[i], sf_upper_bounds[i], "upper"))
    for f, target, side in edges:
        m_val = generalized_gaussian_psf(
            mtf050=mtf050p,
            mtf010=mtf010p,
            mtf002=mtf002p,
            mtf100=mtf100p,
            sf100=sf100p,
            params=p_opt,
            fr=np.array([f]),
        )[0]
        print(
            f"  region {i} {side:>5} @ f={f:7.4f}: "
            f"model={m_val:.6f}  target={target:.6f}  "
            f"err={m_val - target:+.2e}"
        )
