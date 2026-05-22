"""
Synthesize 2D PSFs from a target MTF, with interactive sliders and a kernel
selector preloaded with 92 Siemens recon kernels (EID Force + Alpha).

Models
------
1. Generalized Gaussian (2-pt fit on f50, f10):
       MTF(f) = exp(-(f/fc)^n).  Monotonic; cannot model overshoot.

2. Flexible 5-point model (f_peak, mtf_peak, f50, f10, f2):
   Radial MTF passes through (0, 1), (f_peak, mtf_peak), (f50, 0.5),
   (f10, 0.1), (f2, 0.02), then exponential decay beyond f2.

UI
--
- Six sliders for FOV, f_peak, mtf_peak, f50, f10, f2.
- TextBox + Prev/Next buttons select a kernel from the embedded list. The
  kernel name accepts case-insensitive prefix match (e.g. "Br69" or "br6").
- Loading a kernel auto-adjusts FOV so that Nyquist > 1.2 * f2.

Conventions
-----------
- Frequencies in cycles/mm; lengths in mm. Nyquist = 1 / (2 * pixel_size).
- PSFs are normalized so sum(psf) == 1. Overshoot kernels produce PSFs with
  negative ringing lobes (physical for edge-enhancing kernels).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

x = [
    0.0,
    0.16050156739811913,
    0.32100313479623827,
    0.48150470219435737,
    0.6420062695924765,
    0.8025078369905957,
    0.9630094043887147,
    1.123510971786834,
    1.284012539184953,
    1.4445141065830722,
    1.6050156739811914,
    1.7655172413793105,
    1.9260188087774295,
    2.0865203761755486,
    2.247021943573668,
    2.407523510971787,
    2.568025078369906,
    2.7285266457680253,
    2.8890282131661444,
    3.0495297805642636,
    3.2100313479623828,
    3.370532915360502,
    3.531034482758621,
    3.6915360501567402,
    3.852037617554859,
    4.012539184952979,
    4.173040752351097,
    4.333542319749217,
    4.494043887147336,
    4.654545454545455,
    4.815047021943574,
    4.9755485893416935,
    5.136050156739812,
    5.296551724137931,
    5.457053291536051,
    5.617554858934169,
    5.778056426332289,
    5.938557993730408,
    6.099059561128527,
    6.259561128526646,
    6.4200626959247655,
    6.580564263322884,
    6.741065830721004,
    6.901567398119123,
    7.062068965517242,
    7.222570532915361,
    7.3830721003134805,
    7.543573667711599,
    7.704075235109718,
    7.8645768025078375,
    8.025078369905957,
    8.185579937304075,
    8.346081504702195,
    8.506583072100314,
    8.667084639498434,
    8.827586206896552,
    8.988087774294671,
    9.14858934169279,
    9.30909090909091,
    9.469592476489028,
    9.630094043887148,
    9.790595611285267,
    9.951097178683387,
    10.111598746081505,
    10.272100313479624,
    10.432601880877744,
    10.593103448275862,
    10.753605015673982,
    10.914106583072101,
    11.07460815047022,
    11.235109717868339,
    11.395611285266458,
    11.556112852664578,
    11.716614420062697,
]
y = [
    1.0006281123382121,
    1.001145272443073,
    1.0023473135635859,
    1.004206798341899,
    1.0066162568962345,
    1.0094655909105326,
    1.0125666596234695,
    1.015587089031737,
    1.0183932012902155,
    1.020649358604562,
    1.0220798340553674,
    1.0223922701340409,
    1.0213475499424405,
    1.0185640883805303,
    1.0138520315057737,
    1.0071247207504745,
    0.997464231793704,
    0.9857967307428296,
    0.9708981515712565,
    0.9540064193285164,
    0.9338990078785968,
    0.9104926798422627,
    0.8841351159808163,
    0.8565608969281878,
    0.825248302245515,
    0.7928909840437154,
    0.7558058437809084,
    0.7189898233143243,
    0.6824254025050127,
    0.6400098959960622,
    0.59921956502497,
    0.556967869149348,
    0.5150664788567328,
    0.475358264615425,
    0.43298257353378866,
    0.39439147095789656,
    0.3547551593866396,
    0.3166827343584175,
    0.2814530671470184,
    0.24784234476328323,
    0.21453728275672232,
    0.18692949481879448,
    0.1594427186785696,
    0.13509821342047315,
    0.1114747179297892,
    0.09260705064923665,
    0.0743024758124695,
    0.05877438080274436,
    0.04535578494689576,
    0.034701019063778184,
    0.025808478932966842,
    0.01836314596526228,
    0.012935469484210774,
    0.009231952511641937,
    0.006509771357144991,
    0.005188505089176815,
    0.00472381852815578,
    0.0045382550280815386,
    0.004432893454076172,
    0.004335273836445066,
    0.004234904671536448,
    0.0041198603939323715,
    0.004075474118250989,
    0.004033911234959115,
    0.00405051092020151,
    0.0040824393528117635,
    0.004089880705164319,
    0.004053135485618643,
    0.0039466471953513455,
    0.003777508723867241,
    0.0035600818310587297,
    0.003275501953857981,
    0.0030464271938984468,
    0.002800610238067811,  # 74
]
from scipy import interpolate

fy = interpolate.interp1d(x, y)
fx = interpolate.interp1d(y, x)
s_100 = max(y)
s_all = np.array([1.0, 0.5, 0.1, 0.02]) * s_100
s_100, s_050, s_010, s_002 = s_all
f_all = fx(s_all)
f_100, f_050, f_010, f_002 = f_all
# ===========================================================================
# Kernel library (Siemens EID Force + Alpha, sorted by Model then Kernel)
# Columns: Manufacturer, Model, Kernel, MTF100, SF100, MTF50, MTF10, MTF2, PSFFWHM
# ===========================================================================
KERNELS = (
    ("Siemens", "Alpha", "Br36f", 1.0000, 0.0000, 3.3961, 5.4381, 6.7063, 0.1579),
    ("Siemens", "Alpha", "Br36u", 1.0000, 0.0000, 3.4821, 5.4381, 6.7493, 0.1579),
    ("Siemens", "Alpha", "Br40f", 1.0000, 0.0000, 3.8475, 6.3194, 7.8455, 0.1339),
    ("Siemens", "Alpha", "Br40f_1", 1.0000, 0.0000, 3.8260, 6.3194, 7.8240, 0.1339),
    ("Siemens", "Alpha", "Br40f_3", 1.0000, 0.0000, 3.7615, 6.3194, 7.7595, 0.1339),
    ("Siemens", "Alpha", "Br40f_4", 1.0000, 0.0000, 3.8475, 6.3194, 7.7810, 0.1339),
    ("Siemens", "Alpha", "Br40u", 1.0000, 0.0000, 3.9120, 6.3624, 7.8670, 0.1343),
    ("Siemens", "Alpha", "Br40u_1", 1.0000, 0.0000, 3.8905, 6.3624, 7.8025, 0.1343),
    ("Siemens", "Alpha", "Br40u_3", 1.0000, 0.0000, 3.8260, 6.3194, 7.7810, 0.1343),
    ("Siemens", "Alpha", "Br40u_4", 1.0000, 0.0000, 3.8260, 6.3194, 7.7380, 0.1343),
    ("Siemens", "Alpha", "Br44f", 1.0000, 0.0000, 4.3609, 7.3758, 8.9640, 0.1152),
    ("Siemens", "Alpha", "Br44u", 1.0000, 0.0000, 4.4147, 7.3220, 8.9640, 0.1172),
    ("Siemens", "Alpha", "Br48f", 1.0502, 2.0997, 5.4646, 7.4027, 8.3718, 0.1221),
    ("Siemens", "Alpha", "Br48u", 1.0655, 2.0997, 5.4646, 7.3758, 8.3987, 0.1221),
    ("Siemens", "Alpha", "Br56f", 1.1576, 3.0688, 7.3489, 9.3947, 10.4177, 0.0952),
    ("Siemens", "Alpha", "Br56u", 1.1136, 3.0245, 7.2731, 9.5055, 10.3696, 0.0952),
    ("Siemens", "Alpha", "Br68f", 1.3478, 5.6169, 11.6658, 14.4743, 15.8065, 0.0610),
    ("Siemens", "Alpha", "Br68u", 1.3323, 5.4008, 11.3778, 14.2942, 15.6264, 0.0635),
    ("Siemens", "Alpha", "Br72f", 1.3491, 6.4810, 13.4661, 16.0585, 17.2107, 0.0549),
    ("Siemens", "Alpha", "Br72u", 1.3154, 6.1962, 13.0989, 16.1970, 17.6102, 0.0562),
    ("Siemens", "Alpha", "Br76f", 1.2149, 7.3451, 15.7345, 20.4512, 23.2956, 0.0439),
    (
        "Siemens",
        "Alpha",
        "Br76u",
        1.2904,
        7.5006,
        16.3710,
        21.1975,
        23.2194,
        0.0415,
    ),
    (
        "Siemens",
        "Alpha",
        "Br80u",
        1.2310,
        7.8268,
        19.0451,
        24.6543,
        27.4589,
        0.0342,
    ),
    (
        "Siemens",
        "Alpha",
        "Br84u",
        1.2327,
        8.4790,
        22.0454,
        28.0459,
        30.4591,
        0.0317,
    ),
    (
        "Siemens",
        "Alpha",
        "Br92u",
        1.0401,
        11.3039,
        28.5922,
        35.1086,
        37.9013,
        0.0244,
    ),
    (
        "Siemens",
        "Alpha",
        "Br98u_3",
        1.0615,
        11.3039,
        31.9169,
        42.1569,
        46.5455,
        0.0195,
    ),
    ("Siemens", "Alpha", "Bv40f", 1.0000, 0.0000, 4.0109, 6.6759, 8.1026, 0.1270),
    ("Siemens", "Alpha", "Bv40u", 1.0000, 0.0000, 4.0625, 6.7063, 8.0605, 0.1270),
    ("Siemens", "Alpha", "Hc40f", 1.0241, 1.1607, 3.9980, 6.6031, 8.4345, 0.1299),
    ("Siemens", "Alpha", "Hc40u", 1.0000, 0.0000, 3.9409, 6.4606, 8.6572, 0.1294),
    ("Siemens", "Alpha", "Hr36f", 1.0028, 0.6448, 3.5853, 5.5456, 7.4801, 0.1519),
    ("Siemens", "Alpha", "Hr36u", 1.0005, 0.3869, 3.5595, 5.5972, 7.4543, 0.1514),
    ("Siemens", "Alpha", "Hr40f", 1.0000, 0.0000, 3.8763, 6.3637, 8.5279, 0.1294),
    ("Siemens", "Alpha", "Hr40f_1", 1.0000, 0.0000, 3.8440, 6.3314, 8.4956, 0.1294),
    ("Siemens", "Alpha", "Hr40f_3", 1.0000, 0.0000, 3.7794, 6.2991, 8.5279, 0.1294),
    ("Siemens", "Alpha", "Hr40f_4", 1.0000, 0.0000, 3.7794, 6.2991, 8.4956, 0.1294),
    ("Siemens", "Alpha", "Hr40u", 1.0000, 0.0000, 3.9409, 6.4606, 8.5926, 0.1294),
    ("Siemens", "Alpha", "Hr40u_1", 1.0000, 0.0000, 3.8763, 6.4606, 8.5603, 0.1294),
    ("Siemens", "Alpha", "Hr40u_3", 1.0000, 0.0000, 3.8763, 6.4606, 8.5603, 0.1294),
    ("Siemens", "Alpha", "Hr40u_4", 1.0000, 0.0000, 3.8763, 6.4606, 8.5279, 0.1294),
    ("Siemens", "Alpha", "Hr44f", 1.0000, 0.0000, 4.3609, 7.1066, 8.9802, 0.1187),
    ("Siemens", "Alpha", "Hr44u", 1.0000, 0.0000, 4.4578, 7.2358, 9.0771, 0.1172),
    ("Siemens", "Alpha", "Hr48f", 1.0000, 0.0000, 5.0552, 7.6908, 9.0302, 0.1133),
    ("Siemens", "Alpha", "Hr48u", 1.0601, 1.9382, 5.3300, 7.7527, 9.5293, 0.1138),
    ("Siemens", "Alpha", "Hr60f", 1.2143, 4.0379, 8.5603, 10.9184, 12.1459, 0.0825),
    ("Siemens", "Alpha", "Hr60u", 1.2890, 4.0379, 8.7218, 10.9830, 12.1459, 0.0811),
    ("Siemens", "Alpha", "Hr68f", 1.2746, 5.4915, 11.5321, 14.3425, 15.6346, 0.0610),
    ("Siemens", "Alpha", "Hr68u", 1.3430, 5.6169, 11.7954, 14.5175, 15.8137, 0.0610),
    ("Siemens", "Alpha", "Hr76f", 1.2110, 6.8484, 15.9144, 20.4148, 24.1325, 0.0439),
    (
        "Siemens",
        "Alpha",
        "Hr76u",
        1.2934,
        7.5006,
        16.4362,
        21.1975,
        23.2194,
        0.0415,
    ),
    (
        "Siemens",
        "Alpha",
        "Hr84u",
        1.2183,
        8.4790,
        21.9801,
        28.0459,
        30.4591,
        0.0317,
    ),
    (
        "Siemens",
        "Alpha",
        "Hr92u",
        1.1351,
        10.5026,
        28.9696,
        34.1333,
        36.4964,
        0.0244,
    ),
    (
        "Siemens",
        "Alpha",
        "Hr98u",
        1.0383,
        10.3884,
        35.7658,
        44.3733,
        56.0974,
        0.0171,
    ),
    ("Siemens", "Alpha", "Qr40f", 1.0000, 0.0000, 4.0109, 6.6490, 8.0757, 0.1274),
    ("Siemens", "Alpha", "Qr40u", 1.0000, 0.0000, 4.0625, 6.6848, 7.9960, 0.1282),
    ("Siemens", "Alpha", "Qr48f", 1.0000, 0.0000, 5.4107, 9.0986, 10.9830, 0.0928),
    ("Siemens", "Alpha", "Qr48u", 1.0000, 0.0000, 5.3569, 9.0179, 10.9022, 0.0928),
    ("Siemens", "Alpha", "Qr56f_3", 1.0000, 0.0000, 6.9182, 11.1445, 12.5981, 0.0781),
    ("Siemens", "Alpha", "Qr56u_3", 1.0000, 0.0000, 6.8771, 11.1977, 12.7460, 0.0781),
    ("Siemens", "Alpha", "Qr76f", 1.0000, 0.0000, 14.8926, 18.9146, 20.6539, 0.0464),
    (
        "Siemens",
        "Alpha",
        "Qr76u",
        1.0062,
        3.9385,
        16.1915,
        22.3179,
        25.1186,
        0.0391,
    ),
    (
        "Siemens",
        "Alpha",
        "Qr80u",
        1.0031,
        3.5009,
        18.4670,
        25.8188,
        29.2321,
        0.0332,
    ),
    (
        "Siemens",
        "Alpha",
        "Qr84u",
        1.0021,
        3.0632,
        21.0926,
        28.7945,
        32.4704,
        0.0293,
    ),
    (
        "Siemens",
        "Alpha",
        "Qr89u",
        1.0000,
        0.0000,
        23.8933,
        34.6839,
        39.6387,
        0.0244,
    ),
    ("Siemens", "EID Force", "Bf40", 1.0000, 0.0000, 3.7425, 7.1212, 8.2648, 0.1182),
    ("Siemens", "EID Force", "Br32", 1.0000, 0.0000, 3.0029, 4.7527, 5.8329, 0.1826),
    ("Siemens", "EID Force", "Br40", 1.0000, 0.0000, 3.8465, 6.2636, 7.7450, 0.1328),
    ("Siemens", "EID Force", "Br40_1", 1.0000, 0.0000, 3.7018, 6.2498, 7.6920, 0.1318),
    ("Siemens", "EID Force", "Br40_3", 1.0000, 0.0000, 3.7425, 6.2636, 7.6930, 0.1318),
    ("Siemens", "EID Force", "Br40_5", 1.0000, 0.0000, 3.7685, 6.2636, 7.6670, 0.1318),
    ("Siemens", "EID Force", "Br44", 1.0000, 0.0000, 4.3373, 7.3050, 8.9029, 0.1152),
    ("Siemens", "EID Force", "Br49", 1.0145, 1.8193, 5.5878, 7.5890, 8.6286, 0.1152),
    ("Siemens", "EID Force", "Br59", 1.2681, 3.9134, 8.2833, 10.4357, 11.5445, 0.0840),
    ("Siemens", "EID Force", "Br69", 1.4519, 5.9915, 12.2372, 15.5779, 23.0582, 0.0586),
    ("Siemens", "EID Force", "Hc40", 1.0070, 0.8641, 3.7158, 6.3946, 8.2525, 0.1270),
    ("Siemens", "EID Force", "Hf40", 1.0085, 0.8641, 3.6942, 6.8483, 8.5117, 0.1240),
    ("Siemens", "EID Force", "Hf40_1", 1.0000, 0.0000, 3.6646, 6.8873, 8.9665, 0.1143),
    ("Siemens", "EID Force", "Hf40_3", 1.0000, 0.0000, 3.6906, 6.9133, 8.9665, 0.1133),
    ("Siemens", "EID Force", "Hf40_5", 1.0000, 0.0000, 3.7685, 6.9133, 8.9665, 0.1143),
    ("Siemens", "EID Force", "Hr32", 1.0000, 0.0000, 3.2747, 4.8861, 6.5235, 0.1729),
    ("Siemens", "EID Force", "Hr40", 1.0066, 0.7561, 3.7158, 6.3946, 8.2093, 0.1279),
    ("Siemens", "EID Force", "Hr44", 1.0000, 0.0000, 4.2104, 7.0952, 8.7586, 0.1172),
    ("Siemens", "EID Force", "Hr49", 1.0427, 2.2836, 5.5767, 7.4757, 8.5333, 0.1172),
    ("Siemens", "EID Force", "Hr59", 1.1791, 3.9134, 8.3485, 10.3704, 11.2510, 0.0830),
    ("Siemens", "EID Force", "Hr69", 1.3133, 6.1730, 12.2009, 15.4689, 17.5024, 0.0566),
    ("Siemens", "EID Force", "Qr40", 1.0000, 0.0000, 3.8205, 6.5235, 7.9269, 0.1270),
    ("Siemens", "EID Force", "Qr49", 1.0000, 0.0000, 5.4787, 9.3595, 11.3488, 0.0879),
    ("Siemens", "EID Force", "Qr54", 1.0000, 0.0000, 6.6088, 11.3657, 13.4355, 0.0732),
    ("Siemens", "EID Force", "Ub44", 1.0000, 0.0000, 4.2623, 7.2252, 8.8365, 0.1172),
    ("Siemens", "EID Force", "Ur73", 1.0014, 1.7504, 14.0472, 17.5043, 21.2677, 0.0439),
    ("Siemens", "EID Force", "Ur77", 1.0457, 5.2301, 16.2409, 22.0766, 24.7742, 0.0391),
    (
        "Siemens",
        "EID Force",
        "Ur89",
        1.1931,
        13.6312,
        27.0629,
        34.1777,
        60.7086,
        0.0244,
    ),
)


def find_kernel_index(query):
    """Case-insensitive prefix match on the Kernel column. Returns -1 if none."""
    q = query.strip().lower()
    if not q:
        return -1
    for i, k in enumerate(KERNELS):
        if k[2].lower().startswith(q):
            return i
    # fall back to substring
    for i, k in enumerate(KERNELS):
        if q in k[2].lower():
            return i
    return -1


# ===========================================================================
# Model 1: Generalized Gaussian (two parameters, IFFT)
# ===========================================================================
def generalized_gaussian_psf(
    mtf050,
    mtf010,
    mtf002=None,
    size=256,
    pixel_size=1.0,
    weights=None,
):

    freqs = [mtf050, mtf010]
    vals = [0.5, 0.1]

    if mtf002 is not None:
        freqs.append(mtf002)
        vals.append(0.02)

    freqs = np.asarray(freqs, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if np.any(freqs <= 0):
        raise ValueError("MTF frequencies must be positive.")

    if np.any((vals <= 0) | (vals >= 1)):
        raise ValueError("MTF values must be between 0 and 1.")

    # Model:
    #   MTF(f) = exp(-(f/fc)^n)
    #
    # Taking logs:
    #   log(-log(MTF)) = n * log(f) - n * log(fc)
    #
    # This is linear in log-log space:
    #   y = n*x + b
    #   fc = exp(-b/n)

    x = np.log(freqs)
    y = np.log(-np.log(vals))

    if weights is None:
        weights = np.ones_like(freqs)
    else:
        weights = np.asarray(weights, dtype=float)

    A = np.column_stack([x, np.ones_like(x)])

    W = np.sqrt(weights)[:, None]
    coeffs, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)

    n = coeffs[0]
    b = coeffs[1]

    fc = np.exp(-b / n)

    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)

    mtf2d = np.exp(-((fr / fc) ** n))

    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mtf2d))))

    psf /= psf.sum()

    return psf, mtf2d, (n, fc), f


# ===========================================================================
# Model 2: Flexible five-anchor model with optional overshoot
# ===========================================================================
def build_flexible_radial_mtf(f_peak, mtf_peak, f50, f10, f2):
    """Return a callable mtf(f) for f >= 0 anchored at five control points."""
    eps = 1e-6
    f50 = max(f50, f_peak + eps)
    f10 = max(f10, f50 + eps)
    f2 = max(f2, f10 + eps)

    has_overshoot = (f_peak > eps) and (mtf_peak > 1.0 + 1e-4)

    if has_overshoot:
        xs = np.array([f_peak, f50, f10, f2])
        ys = np.array([mtf_peak, 0.5, 0.1, 0.02])
    else:
        f_peak, mtf_peak = 0.0, 1.0
        xs = np.array([0.0, f50, f10, f2])
        ys = np.array([1.0, 0.5, 0.1, 0.02])

    falling = PchipInterpolator(xs, ys, extrapolate=False)
    slope_f2 = min(float(falling.derivative()(f2)), -1e-6)
    k = slope_f2 / 0.02

    def mtf(f):
        f = np.asarray(f, dtype=float)
        shape = f.shape
        f_flat = f.ravel()
        out = np.zeros_like(f_flat)

        if has_overshoot:
            mask_rise = (f_flat >= 0) & (f_flat <= f_peak)
            mask_fall = (f_flat > f_peak) & (f_flat <= f2)
            t = f_flat[mask_rise] / f_peak
            out[mask_rise] = 1.0 + (mtf_peak - 1.0) * (3 * t**2 - 2 * t**3)
        else:
            mask_fall = (f_flat >= 0) & (f_flat <= f2)

        mask_tail = f_flat > f2
        out[mask_fall] = falling(f_flat[mask_fall])
        out[mask_tail] = np.maximum(0.02 * np.exp(k * (f_flat[mask_tail] - f2)), 0.0)
        return out.reshape(shape)

    return mtf


def flexible_psf(f_peak, mtf_peak, f50, f10, f2, size, pixel_size):
    """Synthesize a PSF whose radial MTF passes through five anchor points."""
    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)

    mtf_radial = build_flexible_radial_mtf(f_peak, mtf_peak, f50, f10, f2)
    mtf2d = mtf_radial(fr)

    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mtf2d))))
    psf /= psf.sum()
    params = dict(f_peak=f_peak, mtf_peak=mtf_peak, f50=f50, f10=f10, f2=f2)
    return psf, mtf2d, params, f


# ===========================================================================
# Verification: measured radial MTF from a synthesized PSF
# ===========================================================================
def radial_mtf(psf, pixel_size=1.0):
    """Azimuthally averaged MTF of a centered PSF, normalized so MTF(0)=1."""
    size = psf.shape[0]
    F = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))))
    centre = size // 2
    F = F / (F[centre, centre] if F[centre, centre] > 0 else 1.0)

    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    f_rad = fr[centre, centre:]
    mtf_rad = F[centre, centre:]
    return interp1d(f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0)


# ===========================================================================
# Interactive demo
# ===========================================================================
def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox, Button

    # --- Slider configuration ---
    F50_MIN, F50_MAX, F50_INIT = 1.0, 40.0, 4.34
    F10_MIN, F10_MAX, F10_INIT = 1.5, 50.0, 7.30
    F2_MIN, F2_MAX, F2_INIT = 2.0, 70.0, 9.00
    FP_MIN, FP_MAX, FP_INIT = 0.0, 15.0, 0.00
    MP_MIN, MP_MAX, MP_INIT = 1.0, 1.50, 1.00
    FOV_MIN, FOV_MAX, FOV_INIT = 5.0, 100.0, 50.0

    SIZE = 1025  # fixed grid; pixel_size = FOV / SIZE

    def safe_fov(f2):
        """Pick FOV so that Nyquist > 1.2 * f2, clamped to slider range."""
        return float(np.clip(SIZE / (2.4 * f2), FOV_MIN, FOV_MAX))

    # --- Compute everything for given parameter set ---
    def compute(f50, f10, f2, f_peak, mtf_peak, fov):
        size = SIZE
        pixel_size = fov / size
        nyq = 1 / (2 * pixel_size)
        f10_gg = max(f10, f50 + 1e-3)

        psf_gg, mtf_gg_2d, (n, fc), f = generalized_gaussian_psf(
            f50, f10_gg, mtf002=f2, size=size, pixel_size=pixel_size
        )
        mtf_func_gg = radial_mtf(psf_gg, pixel_size)

        psf_fx, mtf_fx_2d, params_fx, _ = flexible_psf(
            f_peak,
            mtf_peak,
            f50,
            f10,
            f2,
            size=size,
            pixel_size=pixel_size,
        )
        mtf_func_fx = radial_mtf(psf_fx, pixel_size)

        return dict(
            f50=f50,
            f10=f10,
            f2=f2,
            f_peak=params_fx["f_peak"],
            mtf_peak=params_fx["mtf_peak"],
            fov=fov,
            nyq=nyq,
            psf_gg=psf_gg,
            mtf_gg_2d=mtf_gg_2d,
            mtf_func_gg=mtf_func_gg,
            psf_fx=psf_fx,
            mtf_fx_2d=mtf_fx_2d,
            mtf_func_fx=mtf_func_fx,
            n=n,
            fc=fc,
            f=f,
            extent_mtf=[f.min(), f.max(), f.min(), f.max()],
            extent_psf=[-fov / 2, fov / 2, -fov / 2, fov / 2],
        )

    s = compute(F50_INIT, F10_INIT, F2_INIT, FP_INIT, MP_INIT, FOV_INIT)

    # --- Figure layout: extra space at bottom for kernel selector + sliders ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(
        left=0.06, right=0.97, bottom=0.27, top=0.93, hspace=0.40, wspace=0.30
    )

    art = {}

    def psf_clim(psf):
        vmax = float(np.percentile(np.abs(psf), 99.5))
        if vmax <= 0:
            vmax = float(np.abs(psf).max() or 1.0)
        return -vmax, vmax

    def setup_row(
        r,
        label,
        psf,
        mtf2d,
        mtf_func,
        anchor_xy,
        line_color,
        psf_cmap,
        psf_diverging,
        freq_xmax,
        mtf_vmax,
    ):
        if psf_diverging:
            vmin, vmax = psf_clim(psf)
        else:
            vmin, vmax = 0.0, float(psf.max())
        art[f"img_psf_{r}"] = ax[r, 0].imshow(
            psf,
            cmap=psf_cmap,
            extent=s["extent_psf"],
            vmin=vmin,
            vmax=vmax,
        )
        art[f"title_psf_{r}"] = ax[r, 0].set_title(f"{label} — PSF")
        ax[r, 0].set_xlabel("mm")
        ax[r, 0].set_ylabel("mm")

        art[f"img_mtf_{r}"] = ax[r, 1].imshow(
            mtf2d,
            cmap="viridis",
            extent=s["extent_mtf"],
            vmin=0,
            vmax=mtf_vmax,
            origin="lower",
        )
        cs = ax[r, 1].contour(
            mtf2d,
            levels=[0.1, 0.5],
            colors="white",
            linewidths=1,
            extent=s["extent_mtf"],
        )
        ax[r, 1].clabel(cs, fmt={0.1: "0.1", 0.5: "0.5"}, fontsize=8)
        art[f"contour_{r}"] = cs
        ax[r, 1].set_title(f"{label} — 2D MTF")
        ax[r, 1].set_xlabel("$f_x$ (1/mm)")
        ax[r, 1].set_ylabel("$f_y$ (1/mm)")
        fig.colorbar(art[f"img_mtf_{r}"], ax=ax[r, 1], fraction=0.046, pad=0.04)

        fr = np.linspace(0, freq_xmax, 200)
        (art[f"line_{r}"],) = ax[r, 2].plot(fr, mtf_func(fr), color=line_color, lw=1.5)
        ax[r, 2].axhline(0.5, color="k", lw=0.5)
        ax[r, 2].axhline(0.1, color="k", lw=0.5)
        ax[r, 2].axhline(0.02, color="k", lw=0.4, alpha=0.5)
        art[f"vline_f50_{r}"] = ax[r, 2].axvline(F50_INIT, color="k", lw=0.7, ls="--")
        art[f"vline_f10_{r}"] = ax[r, 2].axvline(F10_INIT, color="gray", lw=0.7, ls=":")
        if r == 1:
            art[f"vline_f2_{r}"] = ax[r, 2].axvline(
                F2_INIT, color="gray", lw=0.7, ls=":"
            )
        art[f"scatter_{r}"] = ax[r, 2].scatter(
            anchor_xy[:, 0],
            anchor_xy[:, 1],
            s=30,
            facecolor="white",
            edgecolor="k",
            zorder=5,
        )
        ax[r, 2].set_xlim(0, freq_xmax)
        ax[r, 2].set_ylim(-0.02, max(1.05, MP_MAX))
        ax[r, 2].set_xlabel("Frequency (1/mm)")
        ax[r, 2].set_ylabel("MTF")
        ax[r, 2].set_title(f"{label} — radial MTF")
        ax[r, 2].grid(alpha=0.3)

    anchors_gg = np.array([[F50_INIT, 0.5], [F10_INIT, 0.1]])
    anchors_fx = np.array(
        [
            [0.0, 1.0],
            [FP_INIT, MP_INIT],
            [F50_INIT, 0.5],
            [F10_INIT, 0.1],
            [F2_INIT, 0.02],
        ]
    )
    freq_xmax_init = max(F10_INIT * 2, F2_INIT * 1.2)

    setup_row(
        0,
        "Gen. Gaussian (2-pt)",
        s["psf_gg"],
        s["mtf_gg_2d"],
        s["mtf_func_gg"],
        anchors_gg,
        "C0",
        "RdBu_r",
        False,
        freq_xmax_init,
        1.0,
    )
    setup_row(
        1,
        "Flexible (5-pt)",
        s["psf_fx"],
        s["mtf_fx_2d"],
        s["mtf_func_fx"],
        anchors_fx,
        "C1",
        "RdBu_r",
        True,
        freq_xmax_init,
        max(1.0, MP_INIT),
    )
    art["title_psf_0"].set_text(
        f"Gen. Gaussian (2-pt) — PSF (n={s['n']:.2f}, $f_c$={s['fc']:.2f} 1/mm)"
    )
    art["title_psf_1"].set_text(
        f"Flexible (5-pt) — PSF (peak={s['mtf_peak']:.2f} @ {s['f_peak']:.2f} 1/mm)"
    )

    # --- Kernel selector (top row of bottom panel) ---
    selector_y = 0.20
    label_y = 0.165
    sl_top_y = 0.13
    sl_mid_y = 0.09
    sl_bot_y = 0.05
    sl_h = 0.025

    ax_kernel = plt.axes([0.10, selector_y, 0.18, sl_h])
    ax_prev = plt.axes([0.30, selector_y, 0.05, sl_h])
    ax_next = plt.axes([0.36, selector_y, 0.05, sl_h])

    tb_kernel = TextBox(ax_kernel, "Kernel ", initial="", textalignment="left")
    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")

    # Status label below the selector
    label_text = fig.text(
        0.45,
        selector_y + 0.005,
        "(no kernel loaded)",
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
    )

    # --- Sliders: 2 columns x 3 rows ---
    col_L_x, col_L_w = 0.08, 0.36
    col_R_x, col_R_w = 0.56, 0.36

    ax_f50 = plt.axes([col_L_x, sl_top_y, col_L_w, sl_h])
    ax_f10 = plt.axes([col_L_x, sl_mid_y, col_L_w, sl_h])
    ax_f2 = plt.axes([col_L_x, sl_bot_y, col_L_w, sl_h])
    ax_fp = plt.axes([col_R_x, sl_top_y, col_R_w, sl_h])
    ax_mp = plt.axes([col_R_x, sl_mid_y, col_R_w, sl_h])
    ax_fov = plt.axes([col_R_x, sl_bot_y, col_R_w, sl_h])

    s_f50 = Slider(
        ax_f50, "$f_{50}$ (1/mm)", F50_MIN, F50_MAX, valinit=F50_INIT, valfmt="%.2f"
    )
    s_f10 = Slider(
        ax_f10, "$f_{10}$ (1/mm)", F10_MIN, F10_MAX, valinit=F10_INIT, valfmt="%.2f"
    )
    s_f2 = Slider(
        ax_f2, "$f_{2}$ (1/mm)", F2_MIN, F2_MAX, valinit=F2_INIT, valfmt="%.2f"
    )
    s_fp = Slider(
        ax_fp, "$f_{peak}$ (1/mm)", FP_MIN, FP_MAX, valinit=FP_INIT, valfmt="%.2f"
    )
    s_mp = Slider(ax_mp, "MTF$_{peak}$", MP_MIN, MP_MAX, valinit=MP_INIT, valfmt="%.3f")
    s_fov = Slider(
        ax_fov, "FOV (mm)", FOV_MIN, FOV_MAX, valinit=FOV_INIT, valfmt="%.1f"
    )

    sliders = (s_f50, s_f10, s_f2, s_fp, s_mp, s_fov)

    # State: index of currently-loaded kernel (-1 = none)
    state = {"idx": -1}

    def redraw_contour(r, mtf2d, extent):
        art[f"contour_{r}"].remove()
        cs = ax[r, 1].contour(
            mtf2d,
            levels=[0.1, 0.5],
            colors="white",
            linewidths=1,
            extent=extent,
        )
        ax[r, 1].clabel(cs, fmt={0.1: "0.1", 0.5: "0.5"}, fontsize=8)
        art[f"contour_{r}"] = cs

    def update(_=None):
        s = compute(
            s_f50.val,
            s_f10.val,
            s_f2.val,
            s_fp.val,
            s_mp.val,
            s_fov.val,
        )
        f50_v, f10_v, f2_v = s["f50"], s["f10"], s["f2"]
        fp_v, mp_v = s["f_peak"], s["mtf_peak"]
        extent_mtf, extent_psf = s["extent_mtf"], s["extent_psf"]
        freq_xmax = max(f10_v * 2, f2_v * 1.2)
        fr = np.linspace(0, freq_xmax, 200)

        # Row 0 — Generalized Gaussian
        art["img_psf_0"].set_data(s["psf_gg"])
        art["img_psf_0"].set_extent(extent_psf)
        art["img_psf_0"].set_clim(s["psf_gg"].min(), s["psf_gg"].max())
        art["img_mtf_0"].set_data(s["mtf_gg_2d"])
        art["img_mtf_0"].set_extent(extent_mtf)
        redraw_contour(0, s["mtf_gg_2d"], extent_mtf)
        art["line_0"].set_data(fr, s["mtf_func_gg"](fr))
        art["vline_f50_0"].set_xdata([f50_v, f50_v])
        art["vline_f10_0"].set_xdata([f10_v, f10_v])
        art["scatter_0"].set_offsets(np.array([[f50_v, 0.5], [f10_v, 0.1]]))
        ax[0, 0].set_xlim(extent_psf[0], extent_psf[1])
        ax[0, 0].set_ylim(extent_psf[2], extent_psf[3])
        ax[0, 1].set_xlim(extent_mtf[0], extent_mtf[1])
        ax[0, 1].set_ylim(extent_mtf[2], extent_mtf[3])
        ax[0, 2].set_xlim(0, freq_xmax)
        art["title_psf_0"].set_text(
            f"Gen. Gaussian (2-pt) — PSF (n={s['n']:.2f}, $f_c$={s['fc']:.2f} 1/mm)"
        )

        # Row 1 — Flexible
        vmin, vmax = psf_clim(s["psf_fx"])
        art["img_psf_1"].set_data(s["psf_fx"])
        art["img_psf_1"].set_extent(extent_psf)
        art["img_psf_1"].set_clim(vmin, vmax)
        art["img_mtf_1"].set_data(s["mtf_fx_2d"])
        art["img_mtf_1"].set_extent(extent_mtf)
        art["img_mtf_1"].set_clim(0, max(1.0, mp_v))
        redraw_contour(1, s["mtf_fx_2d"], extent_mtf)
        art["line_1"].set_data(fr, s["mtf_func_fx"](fr))
        art["vline_f50_1"].set_xdata([f50_v, f50_v])
        art["vline_f10_1"].set_xdata([f10_v, f10_v])
        art["vline_f2_1"].set_xdata([f2_v, f2_v])
        art["scatter_1"].set_offsets(
            np.array(
                [
                    [0.0, 1.0],
                    [fp_v, mp_v],
                    [f50_v, 0.5],
                    [f10_v, 0.1],
                    [f2_v, 0.02],
                ]
            )
        )
        ax[1, 0].set_xlim(extent_psf[0], extent_psf[1])
        ax[1, 0].set_ylim(extent_psf[2], extent_psf[3])
        ax[1, 1].set_xlim(extent_mtf[0], extent_mtf[1])
        ax[1, 1].set_ylim(extent_mtf[2], extent_mtf[3])
        ax[1, 2].set_xlim(0, freq_xmax)
        art["title_psf_1"].set_text(
            f"Flexible (5-pt) — PSF (peak={mp_v:.2f} @ {fp_v:.2f} 1/mm)"
        )

        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)

    # --- Kernel loader ---
    def load_kernel(idx):
        if not (0 <= idx < len(KERNELS)):
            return
        manuf, model, name, mtf100, sf100, mtf50, mtf10, mtf2, _ = KERNELS[idx]
        state["idx"] = idx

        # Decide values, clamped to slider ranges
        f50_v = float(np.clip(mtf50, F50_MIN, F50_MAX))
        f10_v = float(np.clip(mtf10, F10_MIN, F10_MAX))
        f2_v = float(np.clip(mtf2, F2_MIN, F2_MAX))
        fp_v = float(np.clip(sf100, FP_MIN, FP_MAX))
        mp_v = float(np.clip(mtf100, MP_MIN, MP_MAX))
        fov_v = safe_fov(mtf2)

        target = [
            (s_f50, f50_v),
            (s_f10, f10_v),
            (s_f2, f2_v),
            (s_fp, fp_v),
            (s_mp, mp_v),
            (s_fov, fov_v),
        ]
        # Suppress per-slider callbacks; redraw once at the end.
        for sl, _ in target:
            sl.eventson = False
        for sl, v in target:
            sl.set_val(v)
        for sl, _ in target:
            sl.eventson = True

        # Update text widgets
        if tb_kernel.text != name:
            tb_kernel.eventson = False
            tb_kernel.set_val(name)
            tb_kernel.eventson = True
        label_text.set_text(
            f"{idx+1:2d}/{len(KERNELS)}  {manuf} {model}  {name}   "
            f"f50={mtf50:6.2f}  f10={mtf10:6.2f}  f2={mtf2:6.2f}  "
            f"f_peak={sf100:5.2f}  MTF_peak={mtf100:.3f}   "
            f"FOV→{fov_v:.1f}mm  Nyq={SIZE/(2*fov_v):.1f}/mm"
        )
        update()

    def on_textbox_submit(text):
        idx = find_kernel_index(text)
        if idx >= 0:
            load_kernel(idx)
        else:
            label_text.set_text(f"(no kernel matches '{text}')")
            fig.canvas.draw_idle()

    def on_prev(_event):
        idx = state["idx"] - 1 if state["idx"] >= 0 else 0
        load_kernel(idx % len(KERNELS))

    def on_next(_event):
        idx = state["idx"] + 1 if state["idx"] >= 0 else 0
        load_kernel(idx % len(KERNELS))

    tb_kernel.on_submit(on_textbox_submit)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Keep widget refs alive (otherwise GC kills interactivity)
    fig._widgets = (*sliders, tb_kernel, btn_prev, btn_next)

    plt.show()


if __name__ == "__main__":
    main()
