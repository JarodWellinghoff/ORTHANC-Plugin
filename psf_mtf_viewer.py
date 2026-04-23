#!/usr/bin/env python3
"""
PSF / MTF Viewer  —  tkinter + matplotlib
Pure Python; only numpy + matplotlib + scipy required.

Layout:
  Left panel  : mode radio-buttons, common params, mode-specific sliders
  Right canvas: PSF image | 2D MTF | 1D radial MTF  +  navigation toolbar
  Status bar  : live MTF50, grid info, render time

Both Gaussian and DoG modes include a "Fit to Target" sub-panel: enter desired
MTF50 and MTF10 (lp/cm) and click "Fit →" to run scipy differential_evolution
in a background thread to find the best parameters.
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from skimage.measure import profile_line
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ── PSF / MTF math ─────────────────────────────────────────────────────────────


def _gauss_iso_norm(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """Unit-volume isotropic 2-D Gaussian."""
    s2 = max(1e-30, sigma * sigma)
    return np.exp(-0.5 * (x * x + y * y) / s2) / (2.0 * np.pi * s2)


def make_psf(p: dict, dx_mm: float, size: int) -> np.ndarray:
    cx = cy = (size - 1) / 2.0
    idx = np.arange(size, dtype=float)
    ii = idx[:, None]
    jj = idx[None, :]
    x_mm = (jj - cy) * dx_mm
    y_mm = (ii - cx) * dx_mm

    mode = p["mode"]
    if mode == "gaussian":
        pi_, pj_ = max(1e-9, p["p_i"]), max(1e-9, p["p_j"])
        psf = (
            np.exp(-pi_ * (ii - cx) ** 2) * np.exp(-pj_ * (jj - cy) ** 2) / (pi_ * pj_)
        )

    elif mode == "ring":
        R = np.sqrt(x_mm**2 + y_mm**2)
        ring = np.exp(-0.5 * ((R - p["r0_mm"]) / max(1e-12, p["sigma_mm"])) ** 2)
        bump = np.exp(-0.5 * (R**2 / max(1e-30, p["core_sigma_mm"] ** 2)))
        cw = float(np.clip(p["core_weight"], 0.0, 1.0))
        psf = (1 - cw) * ring + cw * bump

    else:  # "dog"  —  (1+alpha)*G(sigma0) - alpha*G(sigma1)
        alpha = max(0.0, p["alpha"])
        beta = max(0.0, p["beta"])
        sigma0_mm = max(1e-9, p["sigma0_mm"]) * beta
        sigma1_mm = max(1e-9, p["sigma1_mm"]) * beta
        psf = (1 + alpha) * _gauss_iso_norm(
            x_mm, y_mm, sigma0_mm
        ) - alpha * _gauss_iso_norm(x_mm, y_mm, sigma1_mm)

    nm = p["normalize"]
    if nm in ("unit_energy", "unit", "energy"):
        s = psf.sum()
        if s > 0:
            psf = psf / s
    elif nm in ("peak_one", "peak", "max"):
        m = psf.max()
        if m > 0:
            psf /= m
    return psf


def compute_mtf(psf: np.ndarray, dx: float, size: int):
    half = size // 2
    psf_n = psf / (psf.sum() or 1.0)
    OT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf_n)))
    mtf2d = np.abs(OT)
    dc = mtf2d[half, half]
    if dc > 0:
        mtf2d /= dc

    thetas = np.deg2rad(np.linspace(0, 360, 16)[:-1])
    center = (half, half)
    radius = half
    end_xs = center[1] + radius * np.cos(thetas)
    end_ys = center[0] + radius * np.sin(thetas)

    radial_profile = np.zeros(radius + 1)
    for i, xy in enumerate(zip(end_xs, end_ys), start=1):
        profile = profile_line(mtf2d, center, xy, order=1)
        radial_profile += (profile[: radius + 1] - radial_profile) / i

    fx = np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(size, d=dx))
    Fx, Fy = np.meshgrid(fx, fy)
    F = np.sqrt(Fx**2 + Fy**2)
    return fx, fy, F[half, half:], radial_profile, mtf2d


def _extract_mtf_crossings(f_cm: np.ndarray, mtf_rad: np.ndarray):
    """Return (mtf50, mtf10) in lp/cm, or None for each if not found."""
    results = {}
    for level, key in [(0.5, "mtf50"), (0.1, "mtf10")]:
        idx = np.where(mtf_rad <= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            results[key] = None
            continue
        split_pts = np.where(np.diff(idx) != 1)[0] + 1
        groups = np.split(idx, split_pts)
        first = groups[-1][0]
        results[key] = float(
            np.interp(
                level,
                [mtf_rad[first], mtf_rad[first - 1]],
                [f_cm[first], f_cm[first - 1]],
            )
        )
    return results["mtf50"], results["mtf10"]


# ── Reusable slider-row widget ──────────────────────────────────────────────────


class SliderRow:
    """
    One grid row:  [Label]  [====== tk.Scale ======]  [Entry]
    Bidirectional sync; fires callback(float) on any change.
    """

    def __init__(self, parent, row, label, lo, hi, res, init, callback, fmt="{:.3f}"):
        self._cb = callback
        self.fmt = fmt
        self._busy = False

        ttk.Label(parent, text=label, anchor="e", width=17).grid(
            row=row, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self.var = tk.DoubleVar(value=init)
        self.scale = tk.Scale(
            parent,
            variable=self.var,
            from_=lo,
            to=hi,
            resolution=res,
            orient="horizontal",
            showvalue=False,
            length=145,
            bd=0,
            highlightthickness=0,
            command=self._from_scale,
        )
        self.scale.grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        self.entry = ttk.Entry(parent, width=9)
        self.entry.insert(0, fmt.format(init))
        self.entry.grid(row=row, column=2, padx=(2, 6), pady=2)
        self.entry.bind("<Return>", self._from_entry)
        self.entry.bind("<FocusOut>", self._from_entry)

    def _from_scale(self, val):
        if self._busy:
            return
        self._busy = True
        v = float(val)
        self.entry.delete(0, "end")
        self.entry.insert(0, self.fmt.format(v))
        self._busy = False
        self._cb(v)

    def _from_entry(self, _=None):
        if self._busy:
            return
        try:
            v = float(self.entry.get())
            self._busy = True
            self.scale.set(v)
            self._busy = False
            self._cb(v)
        except ValueError:
            pass

    def get(self) -> float:
        return self.var.get()

    def set(self, v: float):
        self._busy = True
        self.var.set(v)
        self.entry.delete(0, "end")
        self.entry.insert(0, self.fmt.format(v))
        self._busy = False


# ── Fit-to-Target sub-panel widget ─────────────────────────────────────────────


class FitPanel:
    """
    Reusable 'Fit to Target' sub-panel.  Drop it into any mode LabelFrame
    by giving it a parent, the grid row to start at, and a start callback.
    """

    def __init__(self, parent, start_row: int, start_cb):
        self._start_cb = start_cb

        ttk.Separator(parent, orient="horizontal").grid(
            row=start_row, column=0, columnspan=3, sticky="ew", pady=(8, 4)
        )
        ttk.Label(parent, text="Fit to Target", font=("", 9, "bold")).grid(
            row=start_row + 1, column=0, columnspan=3, pady=(0, 4)
        )

        ttk.Label(parent, text="MTF50 (lp/cm):", anchor="e", width=17).grid(
            row=start_row + 2, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self.entry_50 = ttk.Entry(parent, width=9)
        self.entry_50.insert(0, "3.00")
        self.entry_50.grid(
            row=start_row + 2, column=1, columnspan=2, sticky="w", padx=2, pady=2
        )

        ttk.Label(parent, text="MTF10 (lp/cm):", anchor="e", width=17).grid(
            row=start_row + 3, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self.entry_10 = ttk.Entry(parent, width=9)
        self.entry_10.insert(0, "6.00")
        self.entry_10.grid(
            row=start_row + 3, column=1, columnspan=2, sticky="w", padx=2, pady=2
        )

        ttk.Label(parent, text="MTF50 weight:", anchor="e", width=17).grid(
            row=start_row + 4, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self.weight_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            parent,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.weight_var,
            width=9,
            format="%.1f",
        ).grid(row=start_row + 4, column=1, columnspan=2, sticky="w", padx=2, pady=2)

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(
            row=start_row + 5, column=0, columnspan=3, sticky="ew", pady=(6, 2)
        )
        btn_frame.columnconfigure(0, weight=1)

        self.btn = ttk.Button(btn_frame, text="Fit \u2192", command=self._start_cb)
        self.btn.grid(row=0, column=0, sticky="ew", padx=4)

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate", length=120)
        self.progress.grid(row=1, column=0, sticky="ew", padx=4, pady=(3, 0))

        self.result_var = tk.StringVar(value="")
        ttk.Label(
            parent,
            textvariable=self.result_var,
            foreground="#1565C0",
            font=("", 8),
            wraplength=240,
            justify="left",
        ).grid(
            row=start_row + 6, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 4)
        )

    def get_targets(self):
        """Return (target_50, target_10, weight) or raise ValueError."""
        return (
            float(self.entry_50.get()),
            float(self.entry_10.get()),
            float(self.weight_var.get()),
        )

    def set_busy(self, busy: bool):
        self.btn.config(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def set_result(self, text: str):
        self.result_var.set(text)

    def get_target_lines(self):
        """Return (t50, t10) for overlay lines, or (None, None) on parse error."""
        try:
            return float(self.entry_50.get()), float(self.entry_10.get())
        except ValueError:
            return None, None


# ── Presets ────────────────────────────────────────────────────────────────────

PRESETS = {
    "Ring (default)": {
        "mode": "ring",
        "r0_mm": 2.5,
        "sigma_mm": 0.6,
        "core_weight": 0.12,
        "core_sigma_mm": 0.25,
    },
    "Tight ring": {
        "mode": "ring",
        "r0_mm": 1.2,
        "sigma_mm": 0.3,
        "core_weight": 0.05,
        "core_sigma_mm": 0.15,
    },
    "Wide ring": {
        "mode": "ring",
        "r0_mm": 5.0,
        "sigma_mm": 1.2,
        "core_weight": 0.20,
        "core_sigma_mm": 0.5,
    },
    "Gaussian (sym)": {"mode": "gaussian", "p_i": 0.45, "p_j": 0.45},
    "Gaussian (asym)": {"mode": "gaussian", "p_i": 0.80, "p_j": 0.30},
    "DoG (mild sharp)": {
        "mode": "dog",
        "alpha": 0.3,
        "beta": 1.0,
        "sigma0_mm": 0.6,
        "sigma1_mm": 1.4,
    },
    "DoG (strong sharp)": {
        "mode": "dog",
        "alpha": 1.0,
        "beta": 1.0,
        "sigma0_mm": 0.4,
        "sigma1_mm": 1.2,
    },
}


# ── Main Application ────────────────────────────────────────────────────────────


class PSFViewer:
    MODES = ("gaussian", "ring", "dog")
    _OPT_GRID = 129  # fast grid used during optimization (~8x speedup vs 513)

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PSF / MTF Viewer")
        root.minsize(1120, 640)

        self.p: dict = {
            "size": 513,
            "wire_fov_mm": 200.0,
            "normalize": "unit_energy",
            "mode": "ring",
            "p_i": 0.4522,
            "p_j": 0.4599,
            "r0_mm": 2.5,
            "sigma_mm": 0.6,
            "core_weight": 0.12,
            "core_sigma_mm": 0.25,
            "alpha": 0.5,
            "beta": 1.0,
            "sigma0_mm": 0.6,
            "sigma1_mm": 1.2,
        }
        self._pending = False
        self._opt_running = False

        self._build_ui()
        self.update_plots()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        left = ttk.Frame(self.root, width=295)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        left.grid_propagate(False)

        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(8, 2),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))

        self._build_controls(left)
        self._build_canvas(right)

    # ── Left control panel ─────────────────────────────────────────────────────

    def _build_controls(self, parent):
        pf = ttk.LabelFrame(parent, text=" Preset ", padding=(6, 4))
        pf.pack(fill="x", padx=4, pady=(4, 2))
        self._preset_var = tk.StringVar(value="Ring (default)")
        ttk.Combobox(
            pf,
            textvariable=self._preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=26,
        ).pack(side="left", padx=4)
        ttk.Button(pf, text="Load", width=6, command=self._load_preset).pack(
            side="left", padx=4
        )

        mf = ttk.LabelFrame(parent, text=" PSF Mode ", padding=(6, 4))
        mf.pack(fill="x", padx=4, pady=2)
        self._mode_var = tk.StringVar(value=self.p["mode"])
        for m in self.MODES:
            ttk.Radiobutton(
                mf,
                text=m.upper(),
                variable=self._mode_var,
                value=m,
                command=self._on_mode_change,
            ).pack(side="left", expand=True)

        cf = ttk.LabelFrame(parent, text=" Common ", padding=(6, 4))
        cf.pack(fill="x", padx=4, pady=2)
        cf.columnconfigure(1, weight=1)

        ttk.Label(cf, text="Grid size:", anchor="e", width=17).grid(
            row=0, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self._size_var = tk.IntVar(value=self.p["size"])
        ttk.Spinbox(
            cf, from_=65, to=2049, increment=2, textvariable=self._size_var, width=9
        ).grid(row=0, column=1, columnspan=2, sticky="w", padx=2, pady=2)
        self._size_var.trace_add(
            "write",
            lambda *_: self._update(
                {"size": self._safe_int(self._size_var, self.p["size"])}
            ),
        )

        self._fov_s = SliderRow(
            cf,
            1,
            "FOV (mm):",
            5,
            500,
            1,
            self.p["wire_fov_mm"],
            lambda v: self._update({"wire_fov_mm": v}),
            "{:.1f}",
        )

        ttk.Label(cf, text="Normalize:", anchor="e", width=17).grid(
            row=3, column=0, sticky="e", padx=(4, 2), pady=2
        )
        self._norm_var = tk.StringVar(value=self.p["normalize"])
        ttk.Combobox(
            cf,
            textvariable=self._norm_var,
            width=14,
            values=["unit_energy", "peak_one", "none"],
            state="readonly",
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=2, pady=2)
        self._norm_var.trace_add(
            "write", lambda *_: self._update({"normalize": self._norm_var.get()})
        )

        self._mframes: dict[str, ttk.LabelFrame] = {}
        self._build_gaussian_frame(parent)
        self._build_ring_frame(parent)
        self._build_dog_frame(parent)
        self._show_mode(self.p["mode"])

    def _build_gaussian_frame(self, parent):
        lf = ttk.LabelFrame(parent, text=" Gaussian Parameters ", padding=(6, 4))
        lf.columnconfigure(1, weight=1)
        self._mframes["gaussian"] = lf

        self._pi_s = SliderRow(
            lf,
            0,
            "p_i (px\u207b\u00b9):",
            0.01,
            3.0,
            0.001,
            self.p["p_i"],
            lambda v: self._update({"p_i": v}),
        )
        self._pj_s = SliderRow(
            lf,
            1,
            "p_j (px\u207b\u00b9):",
            0.01,
            3.0,
            0.001,
            self.p["p_j"],
            lambda v: self._update({"p_j": v}),
        )

        # Fit panel starts at row 2 (after the 2 sliders)
        self._gauss_fit = FitPanel(
            lf, start_row=2, start_cb=self._start_optimize_gaussian
        )

    def _build_ring_frame(self, parent):
        lf = ttk.LabelFrame(parent, text=" Ring + Core Parameters ", padding=(6, 4))
        lf.columnconfigure(1, weight=1)
        self._mframes["ring"] = lf
        self._r0_s = SliderRow(
            lf,
            0,
            "r\u2080 (mm):",
            0.1,
            20.0,
            0.05,
            self.p["r0_mm"],
            lambda v: self._update({"r0_mm": v}),
        )
        self._sig_s = SliderRow(
            lf,
            1,
            "\u03c3 ring (mm):",
            0.05,
            5.0,
            0.05,
            self.p["sigma_mm"],
            lambda v: self._update({"sigma_mm": v}),
        )
        self._cw_s = SliderRow(
            lf,
            2,
            "Core weight:",
            0.0,
            1.0,
            0.01,
            self.p["core_weight"],
            lambda v: self._update({"core_weight": v}),
        )
        self._cs_s = SliderRow(
            lf,
            3,
            "Core \u03c3 (mm):",
            0.01,
            2.0,
            0.01,
            self.p["core_sigma_mm"],
            lambda v: self._update({"core_sigma_mm": v}),
        )

    def _build_dog_frame(self, parent):
        lf = ttk.LabelFrame(parent, text=" Difference of Gaussians ", padding=(6, 4))
        lf.columnconfigure(1, weight=1)
        self._mframes["dog"] = lf

        self._al_s = SliderRow(
            lf,
            0,
            "Alpha:",
            0.0,
            2.0,
            0.01,
            self.p["alpha"],
            lambda v: self._update({"alpha": v}),
        )
        self._be_s = SliderRow(
            lf,
            1,
            "Beta:",
            0.0,
            2.0,
            0.01,
            self.p["beta"],
            lambda v: self._update({"beta": v}),
        )
        self._s0_s = SliderRow(
            lf,
            2,
            "\u03c3\u2080 (mm):",
            0.05,
            3.0,
            0.05,
            self.p["sigma0_mm"],
            lambda v: self._update({"sigma0_mm": v}),
        )
        self._s1_s = SliderRow(
            lf,
            3,
            "\u03c3\u2081 (mm):",
            0.05,
            5.0,
            0.05,
            self.p["sigma1_mm"],
            lambda v: self._update({"sigma1_mm": v}),
        )

        # Fit panel starts at row 4 (after the 4 sliders)
        self._dog_fit = FitPanel(lf, start_row=4, start_cb=self._start_optimize_dog)

    def _show_mode(self, mode: str):
        for f in self._mframes.values():
            f.pack_forget()
        self._mframes[mode].pack(fill="x", padx=4, pady=2)

    def _on_mode_change(self):
        m = self.p["mode"] = self._mode_var.get()
        self._show_mode(m)
        self.update_plots()

    def _load_preset(self):
        name = self._preset_var.get()
        if name not in PRESETS:
            return
        overrides = PRESETS[name]
        self.p.update(overrides)
        if "mode" in overrides:
            self._mode_var.set(overrides["mode"])
            self._show_mode(overrides["mode"])
        _sync = {
            "r0_mm": getattr(self, "_r0_s", None),
            "sigma_mm": getattr(self, "_sig_s", None),
            "core_weight": getattr(self, "_cw_s", None),
            "core_sigma_mm": getattr(self, "_cs_s", None),
            "p_i": getattr(self, "_pi_s", None),
            "p_j": getattr(self, "_pj_s", None),
            "alpha": getattr(self, "_al_s", None),
            "beta": getattr(self, "_be_s", None),
            "sigma0_mm": getattr(self, "_s0_s", None),
            "sigma1_mm": getattr(self, "_s1_s", None),
        }
        for key, widget in _sync.items():
            if widget and key in overrides:
                widget.set(overrides[key])
        self.update_plots()

    # ── Right canvas ────────────────────────────────────────────────────────────

    def _build_canvas(self, parent):
        self.fig = Figure(dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False).grid(
            row=1, column=0, sticky="ew"
        )

    # ── Update machinery (debounced) ───────────────────────────────────────────

    @staticmethod
    def _safe_int(var, fallback: int) -> int:
        try:
            return int(var.get())
        except Exception:
            return fallback

    def _update(self, kw: dict):
        self.p.update(kw)
        if not self._pending:
            self._pending = True
            self.root.after(40, self._fire)

    def _fire(self):
        self._pending = False
        self.update_plots()

    # ── Shared optimization engine ─────────────────────────────────────────────

    def _compute_mtf_metrics(self, overrides: dict):
        """
        Return (mtf50_lpcm, mtf10_lpcm) using a fast small grid.
        `overrides` is merged into a copy of self.p before evaluation.
        """
        p = dict(self.p)
        p.update(overrides)
        size = self._OPT_GRID
        dx = p["wire_fov_mm"] / size
        psf = make_psf(p, dx, size)
        _, _, f_rad, mtf_rad, _ = compute_mtf(psf, dx, size)
        return _extract_mtf_crossings(f_rad * 10, mtf_rad)

    def _validate_fit_panel(self, panel: FitPanel):
        """
        Pull and validate targets from a FitPanel.
        Returns (t50, t10, w50) on success, or None after writing an error message.
        """
        try:
            t50, t10, w50 = panel.get_targets()
        except ValueError:
            panel.set_result("Enter valid numeric targets first.")
            return None
        if t50 <= 0 or t10 <= 0:
            panel.set_result("Targets must be > 0.")
            return None
        if t50 >= t10:
            panel.set_result("MTF50 must be < MTF10.")
            return None
        return t50, t10, w50

    def _run_optimize(
        self,
        panel: FitPanel,
        t50: float,
        t10: float,
        w50: float,
        bounds: list,
        overrides_from_x,
        apply_result,
    ):
        """
        Generic background optimizer — runs in a daemon thread.

        Parameters
        ----------
        panel            : FitPanel owning progress / result widgets
        t50, t10, w50    : targets and relative MTF50 weight in the loss
        bounds           : list of (lo, hi) for differential_evolution
        overrides_from_x : callable(x) -> dict   optimizer vector -> param overrides
        apply_result     : callable(x)            applies best params to UI (main thread)
        """
        from scipy.optimize import differential_evolution

        iter_counter = [0]

        def objective(x):
            try:
                m50, m10 = self._compute_mtf_metrics(overrides_from_x(x))
            except Exception:
                return 1e6
            loss = 0.0
            loss += w50 * ((m50 - t50) ** 2 if m50 is not None else 1e4)
            loss += (m10 - t10) ** 2 if m10 is not None else 1e4
            return loss

        def callback(xk, convergence=None):
            iter_counter[0] += 1
            if iter_counter[0] % 5 == 0:
                ic = iter_counter[0]
                try:
                    loss = objective(xk)
                    self.root.after(
                        0,
                        lambda l=loss, i=ic: self.status_var.set(
                            f"Fitting\u2026 iter {i}  \u2502  loss: {l:.5f}"
                        ),
                    )
                except Exception:
                    pass

        result = differential_evolution(
            objective,
            bounds,
            maxiter=300,
            tol=1e-5,
            seed=42,
            popsize=10,
            mutation=(0.5, 1.5),
            recombination=0.7,
            updating="deferred",
            workers=1,
            callback=callback,
            polish=True,
        )

        self.root.after(
            0,
            lambda: self._on_optimize_done(
                result, panel, t50, t10, overrides_from_x, apply_result
            ),
        )

    def _on_optimize_done(
        self, result, panel: FitPanel, t50, t10, overrides_from_x, apply_result
    ):
        """Called on the main thread when any optimization finishes."""
        panel.set_busy(False)
        self._opt_running = False

        if result.fun > 1e3:
            panel.set_result(
                f"No valid solution found (loss={result.fun:.2f}).\n"
                "Try relaxing targets or adjusting FOV."
            )
            self.status_var.set("Fit failed — no valid MTF crossings found.")
            return

        apply_result(result.x)

        # Verify achieved values on the fast grid
        m50, m10 = self._compute_mtf_metrics(overrides_from_x(result.x))
        m50_s = f"{m50:.3f}" if m50 is not None else "N/A"
        m10_s = f"{m10:.3f}" if m10 is not None else "N/A"

        panel.set_result(
            f"\u2713 Converged  (loss={result.fun:.5f})\n"
            f"  MTF50: {m50_s} lp/cm  (target {t50:.2f})\n"
            f"  MTF10: {m10_s} lp/cm  (target {t10:.2f})"
        )
        self.status_var.set(f"Fit done \u2014 MTF50: {m50_s}  MTF10: {m10_s}")
        self.update_plots()

    # ── Gaussian fit ───────────────────────────────────────────────────────────

    def _start_optimize_gaussian(self):
        if self._opt_running:
            return
        vals = self._validate_fit_panel(self._gauss_fit)
        if vals is None:
            return
        t50, t10, w50 = vals

        self._opt_running = True
        self._gauss_fit.set_busy(True)
        self._gauss_fit.set_result("Searching parameter space\u2026")

        def overrides_from_x(x):
            pi_, pj_ = x
            return {"mode": "gaussian", "p_i": float(pi_), "p_j": float(pj_)}

        def apply_result(x):
            pi_, pj_ = x
            self.p.update({"mode": "gaussian", "p_i": float(pi_), "p_j": float(pj_)})
            self._mode_var.set("gaussian")
            self._show_mode("gaussian")
            self._pi_s.set(pi_)
            self._pj_s.set(pj_)

        # Wider bounds than the slider (0.01–3) so the solver explores freely;
        # the slider display clamps on apply.
        bounds = [(0.005, 8.0), (0.005, 8.0)]

        threading.Thread(
            target=self._run_optimize,
            args=(
                self._gauss_fit,
                t50,
                t10,
                w50,
                bounds,
                overrides_from_x,
                apply_result,
            ),
            daemon=True,
        ).start()

    # ── DoG fit ────────────────────────────────────────────────────────────────

    def _start_optimize_dog(self):
        if self._opt_running:
            return
        vals = self._validate_fit_panel(self._dog_fit)
        if vals is None:
            return
        t50, t10, w50 = vals

        self._opt_running = True
        self._dog_fit.set_busy(True)
        self._dog_fit.set_result("Searching parameter space\u2026")

        def overrides_from_x(x):
            alpha, beta, sigma0, sigma1 = x
            return {
                "mode": "dog",
                "alpha": float(alpha),
                "beta": float(beta),
                "sigma0_mm": float(sigma0),
                "sigma1_mm": float(sigma1),
            }

        def apply_result(x):
            alpha, beta, sigma0, sigma1 = x
            self.p.update(
                {
                    "mode": "dog",
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "sigma0_mm": float(sigma0),
                    "sigma1_mm": float(sigma1),
                }
            )
            self._mode_var.set("dog")
            self._show_mode("dog")
            self._al_s.set(alpha)
            self._be_s.set(beta)
            self._s0_s.set(sigma0)
            self._s1_s.set(sigma1)

        bounds = [(0.0, 2.5), (0.1, 2.5), (0.05, 4.0), (0.10, 6.0)]

        threading.Thread(
            target=self._run_optimize,
            args=(self._dog_fit, t50, t10, w50, bounds, overrides_from_x, apply_result),
            daemon=True,
        ).start()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def update_plots(self):
        def rolling_average(data, window_size):
            cumsum_vec = np.cumsum(
                np.pad(data, (window_size // 2, window_size // 2), mode="edge")
            )
            return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

        t0 = time.perf_counter()
        p = self.p
        size = int(p["size"])
        if size % 2 == 0:
            size += 1
        dx = p["wire_fov_mm"] / size

        try:
            psf = make_psf(p, dx, size)
        except Exception as exc:
            self.status_var.set(f"\u26a0  PSF error: {exc}")
            return

        fx, fy, f_rad, mtf_rad, mtf2d = compute_mtf(psf, dx, size)
        fov = p["wire_fov_mm"]

        self.fig.clf()
        ax_psf = self.fig.add_subplot(1, 3, 1)
        ax_mtf2 = self.fig.add_subplot(1, 3, 2)
        ax_mtf = self.fig.add_subplot(1, 3, 3)

        ext_mm = [-fov / 2, fov / 2, -fov / 2, fov / 2]
        ax_psf.imshow(psf, extent=ext_mm, cmap="hot", origin="lower", aspect="equal")
        ax_psf.set_title("PSF", fontweight="bold")
        ax_psf.set_xlabel("x (mm)")
        ax_psf.set_ylabel("y (mm)")

        ext_f = [fx.min() * 10, fx.max() * 10, fy.min() * 10, fy.max() * 10]
        ax_mtf2.imshow(
            mtf2d,
            extent=ext_f,
            cmap="viridis",
            origin="lower",
            aspect="equal",
            vmin=0,
            vmax=1,
        )
        ax_mtf2.set_title("MTF 2D", fontweight="bold")
        ax_mtf2.set_xlabel("fx (lp/cm)")
        ax_mtf2.set_ylabel("fy (lp/cm)")

        f_cm = f_rad * 10
        ax_mtf.plot(f_cm, mtf_rad, "#1565C0", lw=2, label="Radial MTF")
        ax_mtf.axhline(0.5, color="#c62828", ls="--", lw=1, alpha=0.85, label="MTF50")
        ax_mtf.axhline(0.1, color="#e65100", ls=":", lw=1, alpha=0.85, label="MTF10")

        mtf50_str, mtf10_str = "N/A", "N/A"
        f50, f10 = _extract_mtf_crossings(f_cm, mtf_rad)

        if f50 is not None:
            ax_mtf.axvline(f50, color="#c62828", ls="--", lw=1, alpha=0.85)
            ax_mtf.text(
                f50 + 0.12, 0.52, f"{f50:.2f}", color="#c62828", fontsize=8, va="bottom"
            )
            mtf50_str = f"{f50:.2f} lp/cm"

        if f10 is not None:
            ax_mtf.axvline(f10, color="#e65100", ls=":", lw=1, alpha=0.85)
            ax_mtf.text(
                f10 + 0.12, 0.12, f"{f10:.2f}", color="#e65100", fontsize=8, va="bottom"
            )
            mtf10_str = f"{f10:.2f} lp/cm"

        # Faint target-line overlays for whichever fit panel is active
        fit_panel: FitPanel | None = {
            "gaussian": self._gauss_fit,
            "dog": self._dog_fit,
        }.get(p["mode"])
        if fit_panel is not None:
            t50_v, t10_v = fit_panel.get_target_lines()
            if t50_v is not None:
                ax_mtf.axvline(t50_v, color="#c62828", ls="-", lw=0.7, alpha=0.28)
            if t10_v is not None:
                ax_mtf.axvline(t10_v, color="#e65100", ls="-", lw=0.7, alpha=0.28)

        ws = 10
        mtf_norm = (mtf_rad - mtf_rad.min()) / (mtf_rad.max() - mtf_rad.min() + 1e-30)
        mtf_ave = rolling_average(mtf_norm, window_size=ws)
        thresh = 3e-5
        f_idx = np.argmax(~(mtf_ave > thresh)) - 1

        ax_mtf.set_title("Radial MTF", fontweight="bold")
        ax_mtf.set_xlabel("Spatial Frequency (lp/cm)")
        ax_mtf.set_ylabel("MTF")
        ax_mtf.set_xlim([0, f_cm[f_idx]])
        ax_mtf.set_ylim([0, mtf_rad.max() * 1.05])
        ax_mtf.grid(True, alpha=0.25, lw=0.8)
        ax_mtf.legend(fontsize=8, loc="upper right")

        self.fig.tight_layout(pad=1.8)
        self.canvas.draw_idle()

        dt = (time.perf_counter() - t0) * 1e3
        if not self._opt_running:
            self.status_var.set(
                f"Mode: {p['mode'].upper()}  \u2502  MTF50: {mtf50_str}  \u2502  "
                f"MTF10: {mtf10_str}  \u2502  Grid: {size}\xd7{size}  \u2502  "
                f"\u0394x: {dx * 1e3:.2f} \u00b5m  \u2502  Render: {dt:.0f} ms"
            )


# ── Entry point ─────────────────────────────────────────────────────────────────


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    style = ttk.Style()
    for theme in ("clam", "alt", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break
    PSFViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
