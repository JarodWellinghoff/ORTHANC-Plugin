"""
Synthesize 2D PSFs from a target MTF, with interactive sliders for f50, f10,
and physical FOV. Array size is fixed (see SIZE constant in main()).

Two models:
    1. Gaussian             — needs only MTF(0.5)
    2. Generalized Gaussian — needs MTF(0.5) and MTF(0.1)

Conventions
-----------
- All frequencies are in cycles per physical unit (e.g. 1/mm).
- pixel_size has the same physical unit (e.g. mm).
- Nyquist = 1 / (2 * pixel_size). Make sure mtf10, mtf50 < Nyquist.
- PSFs are returned with sum == 1 (energy preserving) and centered.
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Model 1: Gaussian (single parameter, closed form)
# ---------------------------------------------------------------------------
def gaussian_psf(mtf50, size=128, pixel_size=1.0):
    """
    Analytical Gaussian PSF and its 2D MTF, matching MTF(f50) = 0.5.

    Returns (psf, mtf2d, sigma, f). Both PSF and MTF are computed in closed
    form (no FFT), so there are no sampling/ringing artifacts.
    """
    sigma = np.sqrt(np.log(2) / (2 * np.pi**2)) / mtf50  # physical units

    # PSF (image domain)
    c = np.arange(size) - size // 2
    xx, yy = np.meshgrid(c * pixel_size, c * pixel_size)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    # MTF (frequency domain): FT of Gaussian is Gaussian, sigma_f = 1/(2*pi*sigma)
    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    mtf2d = np.exp(-2 * np.pi**2 * sigma**2 * (fx**2 + fy**2))

    return psf, mtf2d, sigma, f


# ---------------------------------------------------------------------------
# Model 2: Generalized Gaussian (two parameters, IFFT)
# ---------------------------------------------------------------------------
def generalized_gaussian_psf(mtf50, mtf10, size=256, pixel_size=1.0):
    """
    PSF whose radial MTF = exp(-(f/fc)^n), fit to MTF(f50)=0.5 and MTF(f10)=0.1.
    Returns (psf, mtf2d, params, f) where params = (n, fc).
    """
    n = np.log(np.log(10) / np.log(2)) / np.log(mtf10 / mtf50)
    fc = mtf50 / np.log(2) ** (1.0 / n)

    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)

    mtf2d = np.exp(-((fr / fc) ** n))

    # MTF is real, even, non-negative -> PSF is real, even, non-negative*
    # *non-negativity not guaranteed for arbitrary MTF shapes; check at runtime
    psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mtf2d))))
    psf /= psf.sum()
    return psf, mtf2d, (n, fc), f


# ---------------------------------------------------------------------------
# Verification: measured radial MTF from a synthesized PSF
# ---------------------------------------------------------------------------
def radial_mtf(psf, pixel_size=1.0, n_bins=None):
    """Azimuthally averaged MTF of a centered PSF."""
    size = psf.shape[0]
    n_bins = n_bins or size // 2
    F = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))))
    F /= F.max()  # MTF(0) = 1

    f = np.fft.fftshift(np.fft.fftfreq(size, d=pixel_size))
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    centre = size // 2
    f_rad = fr[centre, centre:]
    mtf_rad = F[centre, centre:]

    # f_max = f.max()
    # edges = np.linspace(0, f_max, n_bins + 1)
    # centers = 0.5 * (edges[:-1] + edges[1:])
    # mtf_r = np.array(
    #     [F[(fr >= lo) & (fr < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
    # )
    return interp1d(f_rad, mtf_rad, kind="linear", bounds_error=False, fill_value=0.0)


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # --- Slider configuration ---
    F50_MIN, F50_MAX, F50_INIT = 1.0, 8.0, 4.34  # 1/mm
    F10_MIN, F10_MAX, F10_INIT = 1.5, 10.0, 7.30  # 1/mm
    FOV_MIN, FOV_MAX = 5.0, 100.0  # mm
    FOV_INIT = max(min(128 / F10_INIT, FOV_MAX), FOV_MIN)

    SIZE = 513  # fixed; pixel_size = fov / SIZE

    # --- Compute everything for given (f50, f10, fov) ---
    def compute(f50, f10, fov):
        size = SIZE
        pixel_size = fov / size
        nyq = 1 / (2 * pixel_size)
        # Internal clamp so generalized-Gaussian exponent stays well-defined
        f10 = max(f10, f50 + 1e-3)

        psf_g, mtf_g_2d, _, f = gaussian_psf(f50, size=size, pixel_size=pixel_size)
        mtf_func_g = radial_mtf(psf_g, pixel_size)
        psf_gg, mtf_gg_2d, (n, fc), _ = generalized_gaussian_psf(
            f50, f10, size=size, pixel_size=pixel_size
        )
        mtf_func_gg = radial_mtf(psf_gg, pixel_size)

        extent_mtf = [f.min(), f.max(), f.min(), f.max()]
        extent_psf = [-fov / 2, fov / 2, -fov / 2, fov / 2]
        return dict(
            f50=f50,
            f10=f10,
            fov=fov,
            size=size,
            pixel_size=pixel_size,
            nyq=nyq,
            psf_g=psf_g,
            mtf_g_2d=mtf_g_2d,
            mtf_func_g=mtf_func_g,
            psf_gg=psf_gg,
            mtf_gg_2d=mtf_gg_2d,
            mtf_func_gg=mtf_func_gg,
            n=n,
            fc=fc,
            f=f,
            extent_mtf=extent_mtf,
            extent_psf=extent_psf,
        )

    s = compute(F50_INIT, F10_INIT, FOV_INIT)

    # --- Figure layout (extra room at the bottom for three sliders) ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(
        left=0.06, right=0.97, bottom=0.18, top=0.94, hspace=0.40, wspace=0.30
    )

    # Hold mutable artist handles in a dict so the update() closure can swap
    # contour sets after .remove() (set_data isn't available on QuadContourSet).
    art = {}

    def setup_row(r, label, psf, mtf2d, mtf_func, line_color):
        # PSF
        art[f"img_psf_{r}"] = ax[r, 0].imshow(psf, cmap="magma", extent=s["extent_psf"])
        art[f"title_psf_{r}"] = ax[r, 0].set_title(f"{label} — PSF")
        ax[r, 0].set_xlabel("mm")
        ax[r, 0].set_ylabel("mm")

        # 2D MTF + iso-contours at 0.1 and 0.5
        art[f"img_mtf_{r}"] = ax[r, 1].imshow(
            mtf2d,
            cmap="viridis",
            extent=s["extent_mtf"],
            vmin=0,
            vmax=1,
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

        # Radial MTF
        fr = np.linspace(0, F10_INIT * 2, 100)
        mtf1d = mtf_func(fr)
        (art[f"line_{r}"],) = ax[r, 2].plot(fr, mtf1d, color=line_color)
        ax[r, 2].axhline(0.5, color="k", lw=0.5)
        ax[r, 2].axhline(0.1, color="k", lw=0.5)
        art[f"vline_f50_{r}"] = ax[r, 2].axvline(F50_INIT, color="k", lw=0.7, ls="--")
        art[f"vline_f10_{r}"] = ax[r, 2].axvline(F10_INIT, color="gray", lw=0.7, ls=":")
        ax[r, 2].set_xlim(0, F10_INIT * 2)
        ax[r, 2].set_ylim(0, 1.02)
        ax[r, 2].set_xlabel("Frequency (1/mm)")
        ax[r, 2].set_ylabel("MTF")
        ax[r, 2].set_title(f"{label} — radial MTF")
        ax[r, 2].grid(alpha=0.3)

    setup_row(
        0, "Gaussian (1 pt fit)", s["psf_g"], s["mtf_g_2d"], s["mtf_func_g"], "C0"
    )
    setup_row(
        1,
        "Gen. Gaussian (2 pt fit)",
        s["psf_gg"],
        s["mtf_gg_2d"],
        s["mtf_func_gg"],
        "C1",
    )
    art["title_psf_1"].set_text(
        f"Gen. Gaussian (2 pt fit) — PSF (n={s['n']:.2f}, $f_c$={s['fc']:.2f} 1/mm)"
    )

    # --- Sliders ---
    ax_fov = plt.axes([0.15, 0.10, 0.72, 0.025])
    ax_f50 = plt.axes([0.15, 0.06, 0.72, 0.025])
    ax_f10 = plt.axes([0.15, 0.02, 0.72, 0.025])

    s_fov = Slider(
        ax_fov, "FOV (mm)", FOV_MIN, FOV_MAX, valinit=FOV_INIT, valfmt="%.1f"
    )
    s_f50 = Slider(
        ax_f50, "$f_{50}$ (1/mm)", F50_MIN, F50_MAX, valinit=F50_INIT, valfmt="%.2f"
    )
    s_f10 = Slider(
        ax_f10, "$f_{10}$ (1/mm)", F10_MIN, F10_MAX, valinit=F10_INIT, valfmt="%.2f"
    )

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
        s = compute(s_f50.val, s_f10.val, s_fov.val)
        f50_v, f10_v = s["f50"], s["f10"]
        extent_mtf = s["extent_mtf"]
        extent_psf = s["extent_psf"]
        fr = np.linspace(0, f10_v * 2, 100)

        # --- Gaussian row ---
        art["img_psf_0"].set_data(s["psf_g"])
        art["img_psf_0"].set_extent(extent_psf)
        art["img_psf_0"].set_clim(s["psf_g"].min(), s["psf_g"].max())
        art["img_mtf_0"].set_data(s["mtf_g_2d"])
        art["img_mtf_0"].set_extent(extent_mtf)
        redraw_contour(0, s["mtf_g_2d"], extent_mtf)
        art["line_0"].set_data(fr, s["mtf_func_g"](fr))
        art["vline_f50_0"].set_xdata([f50_v, f50_v])
        art["vline_f10_0"].set_xdata([f10_v, f10_v])
        ax[0, 0].set_xlim(extent_psf[0], extent_psf[1])
        ax[0, 0].set_ylim(extent_psf[2], extent_psf[3])
        ax[0, 1].set_xlim(extent_mtf[0], extent_mtf[1])
        ax[0, 1].set_ylim(extent_mtf[2], extent_mtf[3])
        ax[0, 2].set_xlim(0, f10_v * 2)

        # --- Gen. Gaussian row ---
        art["img_psf_1"].set_data(s["psf_gg"])
        art["img_psf_1"].set_extent(extent_psf)
        art["img_psf_1"].set_clim(s["psf_gg"].min(), s["psf_gg"].max())
        art["img_mtf_1"].set_data(s["mtf_gg_2d"])
        art["img_mtf_1"].set_extent(extent_mtf)
        redraw_contour(1, s["mtf_gg_2d"], extent_mtf)
        art["line_1"].set_data(fr, s["mtf_func_gg"](fr))
        art["vline_f50_1"].set_xdata([f50_v, f50_v])
        art["vline_f10_1"].set_xdata([f10_v, f10_v])
        ax[1, 0].set_xlim(extent_psf[0], extent_psf[1])
        ax[1, 0].set_ylim(extent_psf[2], extent_psf[3])
        ax[1, 1].set_xlim(extent_mtf[0], extent_mtf[1])
        ax[1, 1].set_ylim(extent_mtf[2], extent_mtf[3])
        ax[1, 2].set_xlim(0, f10_v * 2)

        art["title_psf_1"].set_text(
            f"Gen. Gaussian (2 pt fit) — PSF (n={s['n']:.2f}, $f_c$={s['fc']:.2f} 1/mm)"
        )

        fig.canvas.draw_idle()

    s_fov.on_changed(update)
    s_f50.on_changed(update)
    s_f10.on_changed(update)

    # Keep slider refs alive (otherwise GC kills interactivity)
    fig._sliders = (s_fov, s_f50, s_f10)

    plt.show()


if __name__ == "__main__":
    main()
