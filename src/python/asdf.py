"""
Synthesize 2D PSFs from a target MTF, with interactive sliders and a kernel
selector preloaded with 92 Siemens recon kernels (EID Force + Alpha).

Generalized-Gaussian model with optional cosine overshoot:
    - For f < mtfpeak:  cosine rise from 1 up to sf100 at f = mtfpeak
    - For f >= mtfpeak: blended generalized-Gaussian decay through
                       (mtf050, 0.5), (mtf010, 0.1), (mtf002, 0.02).

UI
--
- Five sliders: sf100, mtfpeak, mtf050, mtf010, mtf002.
- Two checkboxes next to mtf010 and mtf002 toggle each between "use slider
  value" and None (i.e. let the model cross-propagate from the other anchor).
- TextBox + Prev/Next buttons select a kernel from the embedded list.

Conventions
-----------
- Frequencies in cycles/mm; lengths in mm. Nyquist = 1 / (2 * pixel_size).
- PSFs are normalized so sum(psf) == 1. Overshoot kernels produce PSFs with
  negative ringing lobes (physical for edge-enhancing kernels).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, CheckButtons

# ===========================================================================
# Kernel library (Siemens EID Force + Alpha, sorted by Model then Kernel)
# Columns: Manufacturer, Model, Kernel, mtfpeak, SF100, MTF50, MTF10, MTF2, PSFFWHM
#   SF100  = peak MTF value         (slider: sf100)
#   mtfpeak = frequency of peak      (slider: mtfpeak)
#   MTF50  = frequency at MTF=0.5
#   MTF10  = frequency at MTF=0.1
#   MTF2   = frequency at MTF=0.02
# ===========================================================================
# ============================================================
# Constants
# ============================================================
N_COMPONENTS = 5
N_BPS = 6
N_REGIONS = 5
EPS_DEGENERATE = 1e-4
DEFAULT_PERCENTILES = (0.5, 0.1, 0.02)
# df = pd.read_excel(r"src\data\Recon_Kernels.xlsx")
df = pd.read_csv(r"src\data\mtf_summary.csv")
KERNELS = df.to_records(index=False)

V_BPS = np.array(
    [
        [0.9754, 0.6837, 0.0352, 0.0007, 0.0026, 0.0025],
        [0.0083, 0.0397, 0.0353, 0.0006, 0.0026, 0.0025],
        [0.0067, 0.0122, 0.7985, 0.9987, 0.033, 0.0],
        [0.0082, 0.2644, 0.0559, 0.0, 0.6532, 0.0],
        [0.0014, 0.0, 0.0752, 0.0, 0.3086, 0.995],
    ]
)
BUMPS = np.array(
    [
        [-0.0003, -0.0108, -0.0033, 0.0017, -0.0005],
        [0.0024, -0.0108, -0.0033, 0.0017, -0.0005],
        [0.0024, -0.005, 0.0007, 0.0011, -0.0008],
        [0.0024, -0.0048, -0.0021, -0.0006, -0.0002],
        [0, 0.0432, 0.0187, 0.0009, 0.0003],
    ]
)


def find_kernel_index(query):
    """Case-insensitive prefix match on the Kernel column. Returns -1 if none."""
    q = query.strip().lower()
    if not q:
        return -1
    for i, k in enumerate(KERNELS):
        if k[2].lower().startswith(q):
            return i
    for i, k in enumerate(KERNELS):
        if q in k[2].lower():
            return i
    return -1


# ===========================================================================
# Generalized-Gaussian PSF (with optional cosine overshoot)
# ===========================================================================
def generalized_gaussian_psf(
    mtf050,
    mtf010=None,
    mtf002=None,
    mtfpeak=0,
    sf100=1,
    size=256,
    pixel_size=1.0,
):
    def compute_z_components(fr, mtfpeak, sf100, mtf050, mtf010=None, mtf002=None):
        """Five MTF shape components.

        Accepts mtf010 and/or mtf002 as None; in that case the missing anchor
        is cross-propagated from the other (or n=2 if both are None).

        Returns
        -------
        z              : (5, ...) stack of shape components
        n010, n002     : fitted GG exponents
        c010, c002     : fitted GG scale factors
        mtf010_eff     : effective mtf010 (== input if non-None, else solved)
        mtf002_eff     : effective mtf002 (== input if non-None, else solved)
        """
        sf050 = 0.5 * sf100
        sf010 = 0.1 * sf100
        sf002 = 0.02 * sf100

        L050 = np.log(sf100 / sf050)
        L010 = np.log(sf100 / sf010)
        L002 = np.log(sf100 / sf002)

        d050 = max(mtf050 - mtfpeak, EPS_DEGENERATE)

        def _fit_n(mtf, L):
            if mtf is None:
                return None
            d = max(mtf - mtfpeak, EPS_DEGENERATE)
            return np.log(L / L050) / np.log(d / d050)

        n010 = _fit_n(mtf010, L010)
        n002 = _fit_n(mtf002, L002)

        # Cross-propagate; fall back to n=2 if neither point is given
        if n010 is None and n002 is None:
            n010 = n002 = 2.0
        elif n010 is None:
            n010 = n002 if n002 is not None else 2.0
        elif n002 is None:
            n002 = n010 if n010 is not None else 2.0

        assert n010 is not None and n002 is not None
        c010 = d050 / (L050 ** (1.0 / n010))
        c002 = d050 / (L050 ** (1.0 / n002))

        if mtf010 is None:
            mtf010 = mtfpeak + c010 * (L010 ** (1.0 / n010))
        if mtf002 is None:
            mtf002 = mtfpeak + c002 * (L002 ** (1.0 / n002))

        fr = np.asarray(fr, dtype=float)

        # --- z100: cosine rise + plateau ---
        if mtfpeak > EPS_DEGENERATE:
            t = np.clip(fr / mtfpeak, 0.0, 1.0)
            rise = 1 + (sf100 - 1) * (1.0 - np.cos(np.pi * t)) / 2.0
            z100 = np.where(fr <= mtfpeak, rise, sf100)
        else:
            z100 = np.full_like(fr, sf100)

        z050 = np.full_like(fr, sf100)

        z010 = sf100 * np.exp(-(np.maximum((fr - mtfpeak) / c010, 0.0) ** n010))
        z010 = np.where(fr < mtfpeak, sf100, z010)

        z002 = sf100 * np.exp(-(np.maximum((fr - mtfpeak) / c002, 0.0) ** n002))
        z002 = np.where(fr < mtfpeak, sf100, z002)

        z000 = np.zeros_like(fr)

        return (
            np.stack([z100, z050, z010, z002, z000]),
            n010,
            n002,
            c010,
            c002,
            mtf010,
            mtf002,
        )

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

    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    f_max = float(fr.max())

    # NOTE: compute_z_components must be called BEFORE building `bps`, because
    # bps needs the resolved (non-None) mtf010 / mtf002 values.
    z, n010, n002, c010, c002, mtf010_eff, mtf002_eff = compute_z_components(
        fr, mtfpeak, sf100, mtf050, mtf010, mtf002
    )

    bps = np.array([0.0, mtfpeak, mtf050, mtf010_eff, mtf002_eff, f_max], dtype=float)
    for i in range(1, len(bps)):
        if bps[i] <= bps[i - 1]:
            bps[i] = bps[i - 1] + EPS_DEGENERATE

    r_l, r_h = bps[:-1], bps[1:]
    alphas = np.stack(
        [alpha_component(fr, V_BPS[k], BUMPS[k], r_l, r_h) for k in range(N_COMPONENTS)]
    )
    alphap = np.maximum(alphas.sum(axis=0), 1e-12)

    mtf2d = ((alphas / alphap) * z).sum(axis=0)
    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mtf2d))))
    psf /= psf.sum()
    return psf, mtf2d, (n010, n002, c010, c002, mtf010_eff, mtf002_eff), f


# ===========================================================================
# Verification: measured radial MTF from a synthesized PSF
# ===========================================================================
def radial_mtf(psf, pixel_size=1.0):
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

    # --- Slider configuration ---
    SF100_MIN, SF100_MAX, SF100_INIT = 1.0, 1.50, 1.00
    MTFPEAK_MIN, MTFPEAK_MAX, MTFPEAK_INIT = 0.0, 1.50, 0.00
    MTF050_MIN, MTF050_MAX, MTF050_INIT = 0.10, 4.00, 0.434
    MTF010_MIN, MTF010_MAX, MTF010_INIT = 0.15, 5.00, 0.730
    MTF002_MIN, MTF002_MAX, MTF002_INIT = 0.20, 7.00, 0.900

    FOV = 50.0
    SIZE = 512  # fixed grid; pixel_size = FOV / SIZE

    # --- Compute everything for given parameter set ---
    # mtf010 and mtf002 may be None (meaning: don't constrain, let model
    # cross-propagate).
    def compute(sf100, mtfpeak, mtf050, mtf010, mtf002):
        size = SIZE
        fov = FOV
        pixel_size = fov / size
        nyq = 1 / (2 * pixel_size)

        # Enforce monotone ordering on whichever anchors are provided.
        eps = 1e-3
        mtf050_s = max(mtf050, mtfpeak + eps)

        if mtf010 is None:
            mtf010_s = None
            lower_for_002 = mtf050_s
        else:
            mtf010_s = max(mtf010, mtf050_s + eps)
            lower_for_002 = mtf010_s

        if mtf002 is None:
            mtf002_s = None
        else:
            mtf002_s = max(mtf002, lower_for_002 + eps)

        psf, mtf2d, (n010, n002, c010, c002, mtf010_eff, mtf002_eff), f = (
            generalized_gaussian_psf(
                mtf050_s,
                mtf010=mtf010_s,
                mtf002=mtf002_s,
                mtfpeak=mtfpeak,
                sf100=sf100,
                size=size,
                pixel_size=pixel_size,
            )
        )
        mtf_func = radial_mtf(psf, pixel_size)

        return dict(
            sf100=sf100,
            mtfpeak=mtfpeak,
            mtf050=mtf050_s,
            mtf010=mtf010_s,  # slider-clipped input (may be None)
            mtf002=mtf002_s,  # slider-clipped input (may be None)
            mtf010_eff=mtf010_eff,  # effective frequency used by model
            mtf002_eff=mtf002_eff,  # effective frequency used by model
            nyq=nyq,
            psf=psf,
            mtf2d=mtf2d,
            mtf_func=mtf_func,
            n010=n010,
            n002=n002,
            c010=c010,
            c002=c002,
            f=f,
            extent_mtf=[f.min(), f.max(), f.min(), f.max()],
            extent_psf=[-fov / 2, fov / 2, -fov / 2, fov / 2],
        )

    s = compute(SF100_INIT, MTFPEAK_INIT, MTF050_INIT, MTF010_INIT, MTF002_INIT)

    # --- Figure layout: single row of plots, widget area below ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.42, top=0.90, wspace=0.32)

    art = {}

    # Threshold (relative to peak) below which we treat negative values as
    # numerical noise rather than physical ringing.
    NEG_RINGING_THRESH = 0.02

    def psf_display(psf):
        """Choose (cmap, vmin, vmax) for the PSF panel."""
        peak = float(psf.max())
        neg = float(-psf.min())
        if peak <= 0:
            return "gray", 0.0, 1.0
        if neg / peak < NEG_RINGING_THRESH:
            return "gray", 0.0, peak
        vmax = float(np.percentile(np.abs(psf), 99.5)) or peak
        return "gray", 0.0, vmax

    # ---- PSF ----
    cmap, vmin, vmax = psf_display(s["psf"])
    art["img_psf"] = ax[0].imshow(
        s["psf"], cmap=cmap, extent=s["extent_psf"], vmin=vmin, vmax=vmax
    )
    art["title_psf"] = ax[0].set_title("PSF")
    ax[0].set_xlabel("mm")
    ax[0].set_ylabel("mm")

    # ---- 2D MTF ----
    art["img_mtf"] = ax[1].imshow(
        s["mtf2d"],
        cmap="gray",
        extent=s["extent_mtf"],
        vmin=0,
        vmax=max(1.0, SF100_INIT),
        origin="lower",
    )
    cs = ax[1].contour(
        s["mtf2d"],
        levels=[s["sf100"] * 0.02, s["sf100"] * 0.1, s["sf100"] * 0.5],
        colors="white",
        linewidths=1,
        extent=s["extent_mtf"],
    )
    ax[1].clabel(
        cs,
        fmt={
            s["sf100"] * 0.02: "0.02",
            s["sf100"] * 0.1: "0.1",
            s["sf100"] * 0.5: "0.5",
        },
        fontsize=8,
    )
    art["contour"] = cs
    ax[1].set_title("2D MTF")
    ax[1].set_xlabel("$f_x$ (1/mm)")
    ax[1].set_ylabel("$f_y$ (1/mm)")
    fig.colorbar(art["img_mtf"], ax=ax[1], fraction=0.046, pad=0.04)

    # ---- Radial MTF with anchor markers ----
    freq_xmax_init = max(MTF010_INIT * 2, MTF002_INIT * 1.2)
    fr = np.linspace(0, freq_xmax_init, 400)
    (art["line"],) = ax[2].plot(fr, s["mtf_func"](fr), color="C0", lw=1.5)

    # Horizontal reference levels for each anchor MTF value
    art["hline_sf100"] = ax[2].axhline(
        SF100_INIT, color="tab:red", lw=0.5, ls="--", alpha=0.6
    )
    art["hline_050"] = ax[2].axhline(s["sf100"] * 0.5, color="k", lw=0.5)
    art["hline_010"] = ax[2].axhline(s["sf100"] * 0.1, color="k", lw=0.5)
    art["hline_002"] = ax[2].axhline(s["sf100"] * 0.02, color="k", lw=0.4, alpha=0.5)

    # Vertical reference frequencies for each anchor
    art["vline_mtfpeak"] = ax[2].axvline(MTFPEAK_INIT, color="tab:red", lw=0.7, ls="--")
    art["vline_mtf050"] = ax[2].axvline(MTF050_INIT, color="k", lw=0.7, ls="--")
    art["vline_mtf010"] = ax[2].axvline(MTF010_INIT, color="gray", lw=0.7, ls="--")
    art["vline_mtf002"] = ax[2].axvline(MTF002_INIT, color="gray", lw=0.7, ls="--")

    # Five anchor scatter points: (0,1), (mtfpeak,sf100), (mtf050,0.5),
    # (mtf010,0.1), (mtf002,0.02)
    anchors = np.array(
        [
            [0.0, 1.0],
            [MTFPEAK_INIT, SF100_INIT],
            [MTF050_INIT, SF100_INIT * 0.5],
            [MTF010_INIT, SF100_INIT * 0.1],
            [MTF002_INIT, SF100_INIT * 0.02],
        ]
    )
    art["scatter"] = ax[2].scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=45,
        facecolor="white",
        edgecolor="k",
        linewidths=1.2,
        zorder=5,
    )

    # Per-anchor labels (placed just above each point). Offsets are
    # chosen so DC and the peak don't overlap when mtfpeak == 0.
    anchor_labels = ["DC", "peak", "f50p", "f10p", "f2p"]
    anchor_offsets = [(6, 6), (6, -14), (6, 6), (6, 6), (6, 6)]
    art["labels"] = []
    for (xa, ya), lab, off in zip(anchors, anchor_labels, anchor_offsets):
        t = ax[2].annotate(
            lab,
            (xa, ya),
            xytext=off,
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
        art["labels"].append(t)

    ax[2].set_xlim(0, freq_xmax_init)
    ax[2].set_ylim(-0.02, max(1.05, SF100_MAX))
    ax[2].set_xlabel("Frequency (1/mm)")
    ax[2].set_ylabel("MTF")
    ax[2].set_title("Radial MTF")
    ax[2].grid(alpha=0.3)

    art["title_psf"].set_text(
        f"PSF (n010={s['n010']:.2f}, n002={s['n002']:.2f}, "
        f"c010={s['c010']:.2f}, c002={s['c002']:.2f})"
    )

    # --- Kernel selector ---
    selector_y = 0.31
    sl_h = 0.025

    ax_kernel = plt.axes([0.10, selector_y, 0.18, sl_h])
    ax_prev = plt.axes([0.30, selector_y, 0.05, sl_h])
    ax_next = plt.axes([0.36, selector_y, 0.05, sl_h])

    tb_kernel = TextBox(ax_kernel, "Kernel ", initial="", textalignment="left")
    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")

    label_text = fig.text(
        0.45,
        selector_y + 0.005,
        "(no kernel loaded)",
        fontsize=9,
        family="monospace",
        verticalalignment="bottom",
    )

    # --- Sliders: 2 columns x 3 rows ---
    sl_top_y = 0.22
    sl_mid_y = 0.15
    sl_bot_y = 0.08

    # Shrink the left slider column to make room for the use-anchor checkboxes
    # that sit immediately to its right.
    col_L_x, col_L_w = 0.08, 0.30
    col_R_x, col_R_w = 0.56, 0.36

    # Left column: the three falling-edge frequencies (mtf050, mtf010, mtf002)
    ax_mtf050 = plt.axes([col_L_x, sl_top_y, col_L_w, sl_h])
    ax_mtf010 = plt.axes([col_L_x, sl_mid_y, col_L_w, sl_h])
    ax_mtf002 = plt.axes([col_L_x, sl_bot_y, col_L_w, sl_h])

    # Right column: peak overshoot (sf100, mtfpeak)
    ax_sf100 = plt.axes([col_R_x, sl_top_y, col_R_w, sl_h])
    ax_mtfpeak = plt.axes([col_R_x, sl_mid_y, col_R_w, sl_h])

    s_mtf050 = Slider(
        ax_mtf050, "mtf050", MTF050_MIN, MTF050_MAX, valinit=MTF050_INIT, valfmt="%.3f"
    )
    s_mtf010 = Slider(
        ax_mtf010, "mtf010", MTF010_MIN, MTF010_MAX, valinit=MTF010_INIT, valfmt="%.3f"
    )
    s_mtf002 = Slider(
        ax_mtf002, "mtf002", MTF002_MIN, MTF002_MAX, valinit=MTF002_INIT, valfmt="%.3f"
    )
    s_sf100 = Slider(
        ax_sf100, "sf100", SF100_MIN, SF100_MAX, valinit=SF100_INIT, valfmt="%.4f"
    )
    s_mtfpeak = Slider(
        ax_mtfpeak,
        "mtfpeak",
        MTFPEAK_MIN,
        MTFPEAK_MAX,
        valinit=MTFPEAK_INIT,
        valfmt="%.3f",
    )

    sliders = (s_sf100, s_mtfpeak, s_mtf050, s_mtf010, s_mtf002)

    # --- Use/None checkboxes for mtf010 and mtf002 ---
    # Placed just to the right of the slider's valfmt text on the same row.
    chk_x = 0.46
    chk_w = 0.06
    ax_chk_010 = plt.axes([chk_x, sl_mid_y, chk_w, sl_h])
    ax_chk_002 = plt.axes([chk_x, sl_bot_y, chk_w, sl_h])
    # Hide checkbox-axes spines/ticks so they don't look like extra plots.
    for a in (ax_chk_010, ax_chk_002):
        a.set_xticks([])
        a.set_yticks([])
    chk_010 = CheckButtons(ax_chk_010, ["use"], [True])
    chk_002 = CheckButtons(ax_chk_002, ["use"], [True])

    def use_010():
        return chk_010.get_status()[0]

    def use_002():
        return chk_002.get_status()[0]

    state = {"idx": -1}

    def redraw_contour(mtf2d, extent):
        art["contour"].remove()
        cs = ax[1].contour(
            mtf2d,
            levels=[s["sf100"] * 0.02, s["sf100"] * 0.1, s["sf100"] * 0.5],
            colors="white",
            linewidths=1,
            extent=extent,
        )
        ax[1].clabel(
            cs,
            fmt={
                s["sf100"] * 0.02: "0.02",
                s["sf100"] * 0.1: "0.1",
                s["sf100"] * 0.5: "0.5",
            },
            fontsize=8,
        )
        art["contour"] = cs

    def _set_slider_dim(sl, dimmed):
        """Visually fade a slider when its value isn't being read."""
        alpha = 0.35 if dimmed else 1.0
        try:
            sl.poly.set_alpha(alpha)
        except AttributeError:
            pass
        sl.label.set_alpha(alpha)
        sl.valtext.set_alpha(alpha)

    def update(_=None):
        u010 = use_010()
        u002 = use_002()

        s = compute(
            s_sf100.val,
            s_mtfpeak.val,
            s_mtf050.val,
            s_mtf010.val if u010 else None,
            s_mtf002.val if u002 else None,
        )
        sf100_v = s["sf100"]
        mtfpeak_v = s["mtfpeak"]
        mtf050_v = s["mtf050"]
        mtf010_v = s["mtf010_eff"]  # always non-None; use for display
        mtf002_v = s["mtf002_eff"]  # always non-None; use for display
        extent_mtf = s["extent_mtf"]
        extent_psf = s["extent_psf"]
        freq_xmax = max(mtf010_v * 2, mtf002_v * 1.2)
        fr = np.linspace(0, freq_xmax, 400)

        # PSF
        cmap, vmin, vmax = psf_display(s["psf"])
        art["img_psf"].set_cmap(cmap)
        art["img_psf"].set_data(s["psf"])
        art["img_psf"].set_extent(extent_psf)
        art["img_psf"].set_clim(vmin, vmax)
        ax[0].set_xlim(extent_psf[0], extent_psf[1])
        ax[0].set_ylim(extent_psf[2], extent_psf[3])

        # 2D MTF
        art["img_mtf"].set_data(s["mtf2d"])
        art["img_mtf"].set_extent(extent_mtf)
        art["img_mtf"].set_clim(0, max(1.0, sf100_v))
        redraw_contour(s["mtf2d"], extent_mtf)
        ax[1].set_xlim(extent_mtf[0], extent_mtf[1])
        ax[1].set_ylim(extent_mtf[2], extent_mtf[3])

        # Radial MTF curve
        art["line"].set_data(fr, s["mtf_func"](fr))

        # Reference lines
        art["hline_sf100"].set_ydata([sf100_v, sf100_v])
        art["vline_mtfpeak"].set_xdata([mtfpeak_v, mtfpeak_v])
        art["vline_mtf050"].set_xdata([mtf050_v, mtf050_v])
        art["vline_mtf010"].set_xdata([mtf010_v, mtf010_v])
        art["vline_mtf002"].set_xdata([mtf002_v, mtf002_v])

        # Dim the vlines / sliders for free (None) anchors so the
        # constrained vs. free distinction is visually obvious.
        art["vline_mtf010"].set_linestyle("--" if u010 else ":")
        art["vline_mtf010"].set_alpha(0.9 if u010 else 0.35)
        art["vline_mtf002"].set_linestyle("--" if u002 else ":")
        art["vline_mtf002"].set_alpha(0.9 if u002 else 0.35)
        _set_slider_dim(s_mtf010, not u010)
        _set_slider_dim(s_mtf002, not u002)

        # Anchor scatter (uses effective frequencies)
        anchors = np.array(
            [
                [0.0, 1.0],
                [mtfpeak_v, sf100_v],
                [mtf050_v, sf100_v * 0.5],
                [mtf010_v, sf100_v * 0.1],
                [mtf002_v, sf100_v * 0.02],
            ]
        )
        art["scatter"].set_offsets(anchors)

        # Style anchors: white fill = constrained, light gray = free.
        face = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0] if u010 else [0.75, 0.75, 0.75, 0.85],
                [1.0, 1.0, 1.0, 1.0] if u002 else [0.75, 0.75, 0.75, 0.85],
            ]
        )
        edge = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0] if u010 else [0.4, 0.4, 0.4, 0.7],
                [0.0, 0.0, 0.0, 1.0] if u002 else [0.4, 0.4, 0.4, 0.7],
            ]
        )
        art["scatter"].set_facecolor(face)
        art["scatter"].set_edgecolor(edge)

        # Reposition + restyle anchor labels (use parentheses for free ones).
        free_flags = [False, False, False, not u010, not u002]
        for t, (xa, ya), base_lab, free in zip(
            art["labels"], anchors, anchor_labels, free_flags
        ):
            t.set_position((xa, ya))
            t.set_text(f"({base_lab})" if free else base_lab)
            t.set_alpha(0.6 if free else 1.0)

        ax[2].set_xlim(0, freq_xmax)
        ax[2].set_ylim(-0.02, max(1.05, sf100_v + 0.05))

        art["title_psf"].set_text(
            f"PSF (n010={s['n010']:.2f}, n002={s['n002']:.2f}, "
            f"c010={s['c010']:.2f}, c002={s['c002']:.2f})"
        )

        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)

    # Checkbox toggles re-run update (which reads their current state).
    chk_010.on_clicked(lambda _label: update())
    chk_002.on_clicked(lambda _label: update())

    # --- Kernel loader ---
    def load_kernel(idx):
        if not (0 <= idx < len(KERNELS)):
            return
        (
            manuf,
            model,
            name,
            sfinitv,
            mtfinitv,
            mtfpeakv,
            sf100freq,
            f50v,
            f10v,
            f2v,
        ) = KERNELS[idx]
        # Column legend:
        #   mtfpeakv   = mtfpeak column = peak MTF value      → slider sf100
        #   sf100freq = SF100  column = freq where peak     → slider mtfpeak
        #   f50v/f10v/f2v = MTF50/MTF10/MTF2 frequencies    → sliders mtf050/010/002
        state["idx"] = idx

        sf100_v = float(np.clip(mtfpeakv, SF100_MIN, SF100_MAX))
        mtfpeak_v = float(np.clip(sf100freq, MTFPEAK_MIN, MTFPEAK_MAX))
        mtf050_v = float(np.clip(f50v, MTF050_MIN, MTF050_MAX))
        mtf010_v = float(np.clip(f10v, MTF010_MIN, MTF010_MAX))
        mtf002_v = float(np.clip(f2v, MTF002_MIN, MTF002_MAX))

        target = [
            (s_sf100, sf100_v),
            (s_mtfpeak, mtfpeak_v),
            (s_mtf050, mtf050_v),
            (s_mtf010, mtf010_v),
            (s_mtf002, mtf002_v),
        ]
        for sl, _ in target:
            sl.eventson = False
        for sl, v in target:
            sl.set_val(v)
        for sl, _ in target:
            sl.eventson = True

        # A loaded kernel provides all four anchors; re-enable any checkbox
        # the user had toggled off so the loaded values are honored.
        if not use_010():
            chk_010.eventson = False
            chk_010.set_active(0)  # toggles the (only) box on/off
            chk_010.eventson = True
        if not use_002():
            chk_002.eventson = False
            chk_002.set_active(0)
            chk_002.eventson = True

        if tb_kernel.text != name:
            tb_kernel.eventson = False
            tb_kernel.set_val(name)
            tb_kernel.eventson = True
        label_text.set_text(
            f"{idx+1:2d}/{len(KERNELS)}  {manuf} {model}  {name}   "
            f"sf100={mtfpeakv:.4f}  mtfpeak={sf100freq:6.3f}  "
            f"mtf050={f50v:6.3f}  mtf010={f10v:6.3f}  mtf002={f2v:6.3f}   "
            f"FOV→{FOV:.1f}mm  Nyq={SIZE/(2*FOV):.1f}/mm"
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

    # Keep widget refs alive
    fig._widgets = (*sliders, tb_kernel, btn_prev, btn_next, chk_010, chk_002)

    plt.show()


if __name__ == "__main__":
    main()
