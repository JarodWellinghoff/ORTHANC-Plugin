"""
MTF GUI — Interactive MTF / PSF calculation for CT DICOM images.

Run
---
    python mtf_gui.py

Requirements
------------
    pip install PyQt6 matplotlib pydicom numpy scipy opencv-python

Usage
-----
    1. File -> Open DICOM Folder  (or Open File)
    2. Click "Auto-detect wire" to place the ROI on the brightest feature,
       or just click anywhere on the image to set the ROI center manually.
    3. Adjust ROI size, slice index, W/L. Plots update automatically.
    4. Tools -> Sweep ROI sizes  to study how f50/f10/f02 vary with ROI size
       (per Kayugawa et al. 2013, JACMP 14(4):3905).
"""

import sys
import os
import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QFileDialog,
    QSlider,
    QCheckBox,
    QGroupBox,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QProgressBar,
    QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from mtf_core import compute_mtf
from dicom_io import DicomSlice, load_dicom_path, load_dicom_folder, auto_detect_wire

# ============================================================================
# Worker thread: runs compute_mtf off the UI thread, debounced via QTimer.
# ============================================================================


class ComputeWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    @pyqtSlot(dict)
    def compute(self, params):
        try:
            res = compute_mtf(**params)
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


# ============================================================================
# DICOM canvas: image + draggable/clickable ROI overlay.
# ============================================================================


class DicomCanvas(FigureCanvas):
    """Matplotlib canvas showing the DICOM image with an ROI rectangle.

    Click sets the ROI center. Click-drag moves it. Emits roi_moved with (cy, cx).
    """

    roi_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.image_hu = None
        self._im = None
        self.cy = 256
        self.cx = 256
        self.psfroi = 40
        # NB: don't name these self.window — QWidget already has a window() method
        # and matplotlib's Qt backend calls it on showEvent.
        self.wl_window = 400.0
        self.wl_level = 0.0
        self._dragging = False

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def set_image(self, image_hu: np.ndarray, autoscale_wl: bool = False):
        first = self.image_hu is None or self.image_hu.shape != image_hu.shape
        self.image_hu = image_hu
        if autoscale_wl:
            p1, p99 = np.percentile(image_hu, [1, 99])
            self.wl_level = float((p1 + p99) / 2)
            self.wl_window = float(max(p99 - p1, 1.0))

        vmin = self.wl_level - self.wl_window / 2
        vmax = self.wl_level + self.wl_window / 2
        if self._im is None or first:
            self.ax.clear()
            self.ax.set_facecolor("#1e1e1e")
            self.ax.set_axis_off()
            self._im = self.ax.imshow(
                image_hu, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
        else:
            self._im.set_data(image_hu)
            self._im.set_clim(vmin, vmax)
        self._draw_roi()
        self.draw_idle()

    def set_window_level(self, window: float, level: float):
        self.wl_window = float(window)
        self.wl_level = float(level)
        if self._im is not None:
            self._im.set_clim(
                self.wl_level - self.wl_window / 2,
                self.wl_level + self.wl_window / 2,
            )
            self.draw_idle()

    def set_roi(self, cy: int, cx: int, psfroi: int):
        self.cy, self.cx, self.psfroi = int(cy), int(cx), int(psfroi)
        self._draw_roi()
        self.draw_idle()

    def _draw_roi(self):
        # Clear existing rectangles (don't touch the image artist)
        for p in list(self.ax.patches):
            p.remove()
        if self.image_hu is None:
            return
        half = self.psfroi / 2
        big_half = half + 2
        # Outer = background ring extent
        self.ax.add_patch(
            Rectangle(
                (self.cx - big_half, self.cy - big_half),
                self.psfroi + 4,
                self.psfroi + 4,
                edgecolor="#ff8800",
                fill=False,
                linewidth=0.8,
                linestyle="--",
            )
        )
        # Inner = actual ROI used for PSF
        self.ax.add_patch(
            Rectangle(
                (self.cx - half, self.cy - half),
                self.psfroi,
                self.psfroi,
                edgecolor="#00ff88",
                fill=False,
                linewidth=1.5,
            )
        )

    # ---- mouse handling --------------------------------------------------

    def _xy_to_yxint(self, event):
        if event.xdata is None or event.ydata is None or self.image_hu is None:
            return None
        cy = int(round(event.ydata))
        cx = int(round(event.xdata))
        cy = max(0, min(cy, self.image_hu.shape[0] - 1))
        cx = max(0, min(cx, self.image_hu.shape[1] - 1))
        return cy, cx

    def _on_press(self, event):
        if event.button != 1:
            return
        pos = self._xy_to_yxint(event)
        if pos is None:
            return
        self._dragging = True
        self.cy, self.cx = pos
        self._draw_roi()
        self.draw_idle()
        self.roi_moved.emit(self.cy, self.cx)

    def _on_motion(self, event):
        if not self._dragging:
            return
        pos = self._xy_to_yxint(event)
        if pos is None:
            return
        self.cy, self.cx = pos
        self._draw_roi()
        self.draw_idle()
        self.roi_moved.emit(self.cy, self.cx)

    def _on_release(self, event):
        self._dragging = False


# ============================================================================
# Plots panel: 2x2 grid (2D PSF, 2D MTF, 1D PSF, 1D MTF).
# ============================================================================


class PlotsPanel(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8), facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        gs = self.fig.add_gridspec(
            2, 2, hspace=0.35, wspace=0.25, left=0.08, right=0.97, top=0.93, bottom=0.08
        )
        self.ax_psf2d = self.fig.add_subplot(gs[0, 0])
        self.ax_mtf2d = self.fig.add_subplot(gs[0, 1])
        self.ax_psf1d = self.fig.add_subplot(gs[1, 0])
        self.ax_mtf1d = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_psf2d, self.ax_mtf2d, self.ax_psf1d, self.ax_mtf1d):
            self._style_axes(ax)

        self.ax_psf2d.set_title("2D PSF", color="#eee")
        self.ax_mtf2d.set_title("2D MTF", color="#eee")
        self.ax_psf1d.set_title("1D PSF", color="#eee")
        self.ax_psf1d.set_xlabel("Distance (cm)")
        self.ax_psf1d.set_ylabel("PSF (normalized)")
        self.ax_mtf1d.set_title("1D MTF", color="#eee")
        self.ax_mtf1d.set_xlabel("Spatial frequency (1/cm)")
        self.ax_mtf1d.set_ylabel("MTF")

        self.fig.text(
            0.5,
            0.5,
            "Load a DICOM image and place an ROI\non a wire or bead to begin.",
            ha="center",
            va="center",
            color="#888",
            fontsize=11,
        )
        self._placeholder_text = self.fig.texts[-1]

    def _style_axes(self, ax):
        ax.set_facecolor("#2a2a2a")
        for spine in ax.spines.values():
            spine.set_color("#666")
        ax.tick_params(colors="#bbb", labelsize=8)
        ax.title.set_color("#eee")
        ax.xaxis.label.set_color("#bbb")
        ax.yaxis.label.set_color("#bbb")
        ax.grid(True, color="#3a3a3a", linewidth=0.5)

    def update_plots(self, res: dict):
        if self._placeholder_text is not None:
            self._placeholder_text.remove()
            self._placeholder_text = None

        # -------- 2D PSF --------
        self.ax_psf2d.clear()
        self._style_axes(self.ax_psf2d)
        self.ax_psf2d.set_title("2D PSF", color="#eee")
        self.ax_psf2d.imshow(res["psf_2d"], cmap="gray")
        self.ax_psf2d.set_axis_off()

        # -------- 2D MTF (crop to interesting freq range) --------
        self.ax_mtf2d.clear()
        self._style_axes(self.ax_mtf2d)
        self.ax_mtf2d.set_title("2D MTF", color="#eee")
        mtf2d = res["mtf_2d"]
        freq = res["freq"]
        mtf_eval = res["mtf_eval"]
        f10 = mtf_eval[1] if not np.isnan(mtf_eval[1]) else None
        f02 = mtf_eval[2] if not np.isnan(mtf_eval[2]) else None
        freqlim = 100.0
        if f10 is not None:
            freqlim = float(np.ceil(f10 * 1.6))
        if f02 is not None and not np.isnan(f02):
            freqlim = float(np.ceil(max(f02 * 1.25, f10 * 1.6 if f10 else 1)))
        freqlim = min(freqlim, 100.0)
        dfreq = freq[1] - freq[0] if len(freq) > 1 else 1.0
        n = int(freqlim / dfreq)
        mtf_size = freq.size
        lo = max(mtf_size - n, 0)
        hi = min(mtf_size + n, mtf2d.shape[0])
        self.ax_mtf2d.imshow(mtf2d[lo:hi, lo:hi], cmap="gray")
        self.ax_mtf2d.set_axis_off()

        # -------- 1D PSF --------
        self.ax_psf1d.clear()
        self._style_axes(self.ax_psf1d)
        self.ax_psf1d.set_title("1D PSF", color="#eee")
        self.ax_psf1d.set_xlabel("Distance (cm)", color="#bbb")
        self.ax_psf1d.set_ylabel("PSF (normalized)", color="#bbb")
        xxx = res["xxx"]
        psf_1d = res["psf_1d_all"]
        self.ax_psf1d.plot(
            xxx, psf_1d[:, -1], color="#00ff88", lw=2, label="360-deg avg"
        )
        self.ax_psf1d.plot(
            xxx, psf_1d[:, 0], color="#88aaff", lw=1, alpha=0.7, label="0-deg"
        )
        self.ax_psf1d.plot(
            xxx, psf_1d[:, 1], color="#ff8888", lw=1, alpha=0.7, label="90-deg"
        )
        self.ax_psf1d.axhline(0.5, color="#666", lw=0.5, linestyle=":")
        self.ax_psf1d.axhline(0, color="#444", lw=0.5)
        self.ax_psf1d.legend(
            loc="upper right",
            facecolor="#2a2a2a",
            edgecolor="#666",
            labelcolor="#ddd",
            fontsize=8,
        )

        # -------- 1D MTF --------
        self.ax_mtf1d.clear()
        self._style_axes(self.ax_mtf1d)
        self.ax_mtf1d.set_title("1D MTF", color="#eee")
        self.ax_mtf1d.set_xlabel("Spatial frequency (1/cm)", color="#bbb")
        self.ax_mtf1d.set_ylabel("MTF", color="#bbb")
        mtf_1d = res["mtf_1d_all"]
        m = min(mtf_1d.shape[0], freq.size)
        self.ax_mtf1d.plot(
            freq[:m], mtf_1d[:m, -1], color="#00ff88", lw=2, label="360-deg avg"
        )
        self.ax_mtf1d.plot(
            freq[:m], mtf_1d[:m, 0], color="#88aaff", lw=1, alpha=0.7, label="0-deg"
        )
        self.ax_mtf1d.plot(
            freq[:m], mtf_1d[:m, 1], color="#ff8888", lw=1, alpha=0.7, label="90-deg"
        )
        # Reference lines + markers
        for y, color in [(0.5, "#ffaa55"), (0.1, "#ffaa55"), (0.02, "#ffaa55")]:
            self.ax_mtf1d.axhline(y, color=color, lw=0.5, linestyle=":")
        labels = ["f50", "f10", "f02"]
        for i, f in enumerate(mtf_eval):
            if not np.isnan(f):
                self.ax_mtf1d.axvline(
                    f, color="#ffaa55", lw=0.8, linestyle="--", alpha=0.7
                )
                self.ax_mtf1d.text(
                    f,
                    0.95 - 0.07 * i,
                    f" {labels[i]}={f:.2f}",
                    color="#ffcc88",
                    fontsize=8,
                    va="top",
                )
        self.ax_mtf1d.set_xlim(0, freqlim)
        self.ax_mtf1d.set_ylim(-0.05, 1.05)
        self.ax_mtf1d.legend(
            loc="upper right",
            facecolor="#2a2a2a",
            edgecolor="#666",
            labelcolor="#ddd",
            fontsize=8,
        )

        self.draw_idle()


# ============================================================================
# ROI-size sweep dialog (Kayugawa-style sensitivity analysis).
# ============================================================================


class SweepDialog(QDialog):
    """Sweep ROI size and plot f50/f10/f02/FWHM vs ROI size."""

    def __init__(self, parent, image_hu, cy, cx, rFOV_cm, size_range=(20, 80), step=2):
        super().__init__(parent)
        self.setWindowTitle("ROI size sweep")
        self.resize(800, 600)

        self.image_hu = image_hu
        self.cy = cy
        self.cx = cx
        self.rFOV_cm = rFOV_cm
        self.sizes = list(range(size_range[0], size_range[1] + 1, step))

        layout = QVBoxLayout(self)
        info = QLabel(
            f"Sweeping psfroi from {size_range[0]} to {size_range[1]} px (step {step}) "
            f"at center=({cy},{cx}). Watch where f10 plateaus — that's roughly where "
            "truncation stops biasing the measurement."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.progress = QProgressBar()
        self.progress.setMaximum(len(self.sizes))
        layout.addWidget(self.progress)

        self.fig = Figure(figsize=(8, 6), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        btns.button(QDialogButtonBox.StandardButton.Save).setText("Export CSV")
        btns.accepted.connect(self._export_csv)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.results = []
        QTimer.singleShot(50, self._run_sweep)

    def _run_sweep(self):
        for i, psfroi in enumerate(self.sizes):
            try:
                res = compute_mtf(self.image_hu, self.cy, self.cx, psfroi, self.rFOV_cm)
                self.results.append(
                    (
                        psfroi,
                        float(res["fwhm_psf"]),
                        float(res["mtf_eval"][0]),
                        float(res["mtf_eval"][1]),
                        float(res["mtf_eval"][2]),
                    )
                )
            except Exception:
                self.results.append((psfroi, np.nan, np.nan, np.nan, np.nan))
            self.progress.setValue(i + 1)
            QApplication.processEvents()
        self._plot()

    def _plot(self):
        if not self.results:
            return
        sizes = [r[0] for r in self.results]
        fwhm = [r[1] for r in self.results]
        f50 = [r[2] for r in self.results]
        f10 = [r[3] for r in self.results]
        f02 = [r[4] for r in self.results]

        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor("#2a2a2a")
            for spine in ax.spines.values():
                spine.set_color("#666")
            ax.tick_params(colors="#bbb", labelsize=8)
            ax.title.set_color("#eee")
            ax.xaxis.label.set_color("#bbb")
            ax.yaxis.label.set_color("#bbb")
            ax.grid(True, color="#3a3a3a", linewidth=0.5)

        ax1.plot(sizes, f50, "o-", color="#88aaff", label="f50")
        ax1.plot(sizes, f10, "s-", color="#00ff88", label="f10")
        ax1.plot(sizes, f02, "^-", color="#ff8888", label="f02")
        ax1.set_ylabel("Cutoff freq (1/cm)", color="#bbb")
        ax1.set_title("MTF cutoff frequencies vs ROI size", color="#eee")
        ax1.legend(facecolor="#2a2a2a", edgecolor="#666", labelcolor="#ddd", fontsize=8)

        ax2.plot(sizes, fwhm, "o-", color="#ffaa55")
        ax2.set_xlabel("psfroi (pixels)", color="#bbb")
        ax2.set_ylabel("FWHM (cm)", color="#bbb")
        ax2.set_title("PSF FWHM vs ROI size", color="#eee")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export sweep CSV", "roi_sweep.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["psfroi_px", "fwhm_cm", "f50_per_cm", "f10_per_cm", "f02_per_cm"]
            )
            w.writerows(self.results)


# ============================================================================
# Main window.
# ============================================================================


class MainWindow(QMainWindow):
    compute_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT MTF / PSF Interactive Viewer")
        self.resize(1500, 900)

        self.slices: List[DicomSlice] = []
        self.current_slice_idx = 0
        self.cy = 256
        self.cx = 256
        self.psfroi = 40
        self.avg_window = 1
        self._last_result: Optional[dict] = None

        self._build_ui()
        self._build_worker()
        self._build_menu()
        self._apply_dark_palette()

    # ---- UI construction ------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)

        # ---- Top parameter bar ----
        topbar = QHBoxLayout()

        roi_group = QGroupBox("ROI")
        rg = QFormLayout(roi_group)
        rg.setContentsMargins(8, 4, 8, 4)
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(10, 200)
        self.roi_size_spin.setValue(self.psfroi)
        self.roi_size_spin.setSuffix(" px")
        self.roi_size_spin.valueChanged.connect(self._on_roi_size_changed)
        self.roi_cy_spin = QSpinBox()
        self.roi_cy_spin.setRange(0, 9999)
        self.roi_cy_spin.valueChanged.connect(self._on_roi_yx_spin)
        self.roi_cx_spin = QSpinBox()
        self.roi_cx_spin.setRange(0, 9999)
        self.roi_cx_spin.valueChanged.connect(self._on_roi_yx_spin)
        rg.addRow("Size:", self.roi_size_spin)
        rg.addRow("Center y (row):", self.roi_cy_spin)
        rg.addRow("Center x (col):", self.roi_cx_spin)
        topbar.addWidget(roi_group)

        slice_group = QGroupBox("Slice")
        sg = QFormLayout(slice_group)
        sg.setContentsMargins(8, 4, 8, 4)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_label = QLabel("0 / 0")
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 50)
        self.avg_spin.setValue(1)
        self.avg_spin.setSuffix(" slices")
        self.avg_spin.valueChanged.connect(self._on_avg_changed)
        sg.addRow("Index:", self.slice_slider)
        sg.addRow("", self.slice_label)
        sg.addRow("Average:", self.avg_spin)
        topbar.addWidget(slice_group, stretch=2)

        wl_group = QGroupBox("Window / Level")
        wlg = QFormLayout(wl_group)
        wlg.setContentsMargins(8, 4, 8, 4)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 10000)
        self.window_spin.setValue(400)
        self.window_spin.valueChanged.connect(self._on_wl_changed)
        self.level_spin = QSpinBox()
        self.level_spin.setRange(-2000, 5000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self._on_wl_changed)
        wlg.addRow("Window:", self.window_spin)
        wlg.addRow("Level:", self.level_spin)
        topbar.addWidget(wl_group)

        actions_group = QGroupBox("Actions")
        ag = QVBoxLayout(actions_group)
        ag.setContentsMargins(8, 4, 8, 4)
        self.detect_btn = QPushButton("Auto-detect wire")
        self.detect_btn.clicked.connect(self._on_auto_detect)
        self.sweep_btn = QPushButton("Sweep ROI sizes…")
        self.sweep_btn.clicked.connect(self._on_sweep)
        self.export_btn = QPushButton("Export current MTF…")
        self.export_btn.clicked.connect(self._on_export)
        ag.addWidget(self.detect_btn)
        ag.addWidget(self.sweep_btn)
        ag.addWidget(self.export_btn)
        ag.addStretch()
        topbar.addWidget(actions_group)

        outer.addLayout(topbar)

        # ---- Main splitter: DICOM | Plots ----
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.dicom_canvas = DicomCanvas()
        self.dicom_canvas.roi_moved.connect(self._on_canvas_roi_moved)
        self.plots_panel = PlotsPanel()
        self.splitter.addWidget(self.dicom_canvas)
        self.splitter.addWidget(self.plots_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([700, 800])
        outer.addWidget(self.splitter, stretch=1)

        # ---- Status bar (metrics readout) ----
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.metrics_label = QLabel("No data")
        f = QFont("Menlo")
        f.setPointSize(10)
        self.metrics_label.setFont(f)
        self.busy_label = QLabel("")
        sb.addWidget(self.metrics_label, 1)
        sb.addPermanentWidget(self.busy_label)

    def _build_worker(self):
        self.worker_thread = QThread(self)
        self.worker = ComputeWorker()
        self.worker.moveToThread(self.worker_thread)
        self.compute_requested.connect(self.worker.compute)
        self.worker.finished.connect(self._on_compute_finished)
        self.worker.failed.connect(self._on_compute_failed)
        self.worker_thread.start()

        # Debounce timer
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(150)
        self.debounce.timeout.connect(self._dispatch_compute)
        self._compute_in_flight = False

    def _build_menu(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")

        act_open_file = QAction("Open DICOM &File…", self)
        act_open_file.setShortcut(QKeySequence.StandardKey.Open)
        act_open_file.triggered.connect(self._open_file)
        m_file.addAction(act_open_file)

        act_open_folder = QAction("Open DICOM Folder…", self)
        act_open_folder.setShortcut("Ctrl+Shift+O")
        act_open_folder.triggered.connect(self._open_folder)
        m_file.addAction(act_open_folder)

        m_file.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_tools = mb.addMenu("&Tools")
        act_detect = QAction("Auto-&detect wire", self)
        act_detect.setShortcut("Ctrl+D")
        act_detect.triggered.connect(self._on_auto_detect)
        m_tools.addAction(act_detect)
        act_sweep = QAction("&Sweep ROI sizes…", self)
        act_sweep.setShortcut("Ctrl+R")
        act_sweep.triggered.connect(self._on_sweep)
        m_tools.addAction(act_sweep)

    def _apply_dark_palette(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ddd; }
            QGroupBox {
                border: 1px solid #444; border-radius: 4px;
                margin-top: 8px; padding-top: 4px; color: #ccc;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton {
                background-color: #2d2d2d; color: #ddd;
                border: 1px solid #555; border-radius: 3px; padding: 4px 10px;
            }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:pressed { background-color: #444; }

            /* Spinboxes: padding-right reserves space for the buttons, and the
               up/down subcontrols are positioned explicitly so their hit
               regions don't collapse under the line-edit. */
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d; color: #ddd;
                border: 1px solid #444; border-radius: 3px;
                padding-right: 18px; padding-left: 4px;
                min-height: 20px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #444;
                border-bottom: 1px solid #444;
                background-color: #353535;
                border-top-right-radius: 3px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #444;
                background-color: #353535;
                border-bottom-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4a4a4a;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #5a5a5a;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #ccc;
                width: 0px; height: 0px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #ccc;
                width: 0px; height: 0px;
            }

            QLabel { color: #ddd; }
            QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #88aaff; width: 14px; margin: -6px 0; border-radius: 7px;
            }
            QStatusBar { background-color: #252525; color: #ccc; }
            QMenuBar { background-color: #252525; color: #ddd; }
            QMenuBar::item:selected { background: #3a3a3a; }
            QMenu { background-color: #252525; color: #ddd; border: 1px solid #444; }
            QMenu::item:selected { background: #3a3a3a; }
            QProgressBar { border: 1px solid #444; background: #2d2d2d; text-align: center; }
            QProgressBar::chunk { background-color: #00aa55; }
            """)

    # ---- File loading ---------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DICOM file",
            "",
            "DICOM (*.dcm *.DCM *.ima *.IMA);;All files (*)",
        )
        if not path:
            return
        try:
            slc = load_dicom_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self._set_slices([slc])

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if not folder:
            return
        try:
            slices = load_dicom_folder(folder)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        if not slices:
            QMessageBox.warning(self, "No DICOMs", "No readable DICOM files in folder.")
            return
        self._set_slices(slices)

    def _set_slices(self, slices):
        self.slices = slices
        self.current_slice_idx = 0
        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, max(0, len(slices) - 1))
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)
        self._update_slice_label()

        # Center ROI in image
        first = slices[0].image_hu
        h, w = first.shape
        self.cy, self.cx = h // 2, w // 2
        self.roi_cy_spin.blockSignals(True)
        self.roi_cx_spin.blockSignals(True)
        self.roi_cy_spin.setRange(0, h - 1)
        self.roi_cx_spin.setRange(0, w - 1)
        self.roi_cy_spin.setValue(self.cy)
        self.roi_cx_spin.setValue(self.cx)
        self.roi_cy_spin.blockSignals(False)
        self.roi_cx_spin.blockSignals(False)

        self._refresh_image(autoscale_wl=True)
        # Auto-fit window/level controls to the data
        self.window_spin.blockSignals(True)
        self.level_spin.blockSignals(True)
        self.window_spin.setValue(int(round(self.dicom_canvas.wl_window)))
        self.level_spin.setValue(int(round(self.dicom_canvas.wl_level)))
        self.window_spin.blockSignals(False)
        self.level_spin.blockSignals(False)
        # Try auto-detect on first load
        self._on_auto_detect()

    # ---- Image refresh + averaging --------------------------------------

    def _current_image(self) -> Optional[np.ndarray]:
        if not self.slices:
            return None
        n = len(self.slices)
        if self.avg_window <= 1:
            return self.slices[self.current_slice_idx].image_hu
        half = self.avg_window // 2
        lo = max(0, self.current_slice_idx - half)
        hi = min(n, lo + self.avg_window)
        lo = max(0, hi - self.avg_window)
        imgs = [s.image_hu for s in self.slices[lo:hi]]
        return np.mean(np.stack(imgs, axis=0), axis=0).astype(np.float32)

    def _refresh_image(self, autoscale_wl: bool = False):
        img = self._current_image()
        if img is None:
            return
        self.dicom_canvas.set_image(img, autoscale_wl=autoscale_wl)
        self.dicom_canvas.set_roi(self.cy, self.cx, self.psfroi)
        self._schedule_compute()

    def _update_slice_label(self):
        n = len(self.slices)
        self.slice_label.setText(
            f"{self.current_slice_idx + 1 if n else 0} / {n}"
            + (f"  (avg {self.avg_window})" if self.avg_window > 1 else "")
        )

    # ---- Signal handlers ------------------------------------------------

    def _on_slice_changed(self, v):
        self.current_slice_idx = int(v)
        self._update_slice_label()
        self._refresh_image()

    def _on_avg_changed(self, v):
        self.avg_window = int(v)
        self._update_slice_label()
        self._refresh_image()

    def _on_wl_changed(self, _):
        self.dicom_canvas.set_window_level(
            self.window_spin.value(), self.level_spin.value()
        )

    def _on_roi_size_changed(self, v):
        self.psfroi = int(v)
        self.dicom_canvas.set_roi(self.cy, self.cx, self.psfroi)
        self._schedule_compute()

    def _on_roi_yx_spin(self, _):
        self.cy = self.roi_cy_spin.value()
        self.cx = self.roi_cx_spin.value()
        self.dicom_canvas.set_roi(self.cy, self.cx, self.psfroi)
        self._schedule_compute()

    def _on_canvas_roi_moved(self, cy, cx):
        self.cy, self.cx = cy, cx
        self.roi_cy_spin.blockSignals(True)
        self.roi_cx_spin.blockSignals(True)
        self.roi_cy_spin.setValue(cy)
        self.roi_cx_spin.setValue(cx)
        self.roi_cy_spin.blockSignals(False)
        self.roi_cx_spin.blockSignals(False)
        self._schedule_compute()

    def _on_auto_detect(self):
        img = self._current_image()
        if img is None:
            return
        self.busy_label.setText("Detecting wire…")
        QApplication.processEvents()
        try:
            pos = auto_detect_wire(img)
        except Exception as e:
            QMessageBox.warning(self, "Auto-detect failed", str(e))
            self.busy_label.setText("")
            return
        if pos is None:
            self.busy_label.setText("No wire found")
            return
        cy, cx = pos
        self.cy, self.cx = cy, cx
        self.roi_cy_spin.blockSignals(True)
        self.roi_cx_spin.blockSignals(True)
        self.roi_cy_spin.setValue(cy)
        self.roi_cx_spin.setValue(cx)
        self.roi_cy_spin.blockSignals(False)
        self.roi_cx_spin.blockSignals(False)
        self.dicom_canvas.set_roi(self.cy, self.cx, self.psfroi)
        self.busy_label.setText("")
        self._schedule_compute()

    def _on_sweep(self):
        img = self._current_image()
        if img is None or not self.slices:
            return
        rFOV = self.slices[self.current_slice_idx].rFOV_cm
        dlg = SweepDialog(self, img, self.cy, self.cx, rFOV)
        dlg.exec()

    def _on_export(self):
        if self._last_result is None:
            QMessageBox.information(self, "Nothing to export", "Compute an MTF first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MTF CSV", "mtf.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        freq = self._last_result["freq"]
        mtf_1d = self._last_result["mtf_1d_all"]
        n = min(freq.size, mtf_1d.shape[0])
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["freq_per_cm", "mtf_0deg", "mtf_90deg", "mtf_radial_avg"])
            for i in range(n):
                w.writerow([freq[i], mtf_1d[i, 0], mtf_1d[i, 1], mtf_1d[i, 2]])
        self.busy_label.setText(f"Saved {os.path.basename(path)}")

    # ---- Compute dispatch -----------------------------------------------

    def _schedule_compute(self):
        self.debounce.start()

    def _dispatch_compute(self):
        if not self.slices or self._compute_in_flight:
            if self._compute_in_flight:
                # Re-arm so we recompute as soon as current job finishes
                self.debounce.start()
            return
        img = self._current_image()
        if img is None:
            return
        slc = self.slices[self.current_slice_idx]
        params = dict(
            img_hu=img,
            cy0=int(self.cy),
            cx0=int(self.cx),
            psfroi=int(self.psfroi),
            rFOV_cm=float(slc.rFOV_cm),
        )
        self._compute_in_flight = True
        self.busy_label.setText("Computing…")
        self.compute_requested.emit(params)

    @pyqtSlot(dict)
    def _on_compute_finished(self, res):
        self._compute_in_flight = False
        self._last_result = res
        self.plots_panel.update_plots(res)
        ev = res["mtf_eval"]
        self.metrics_label.setText(
            f"FWHM = {res['fwhm_psf']*10:.3f} mm   "
            f"f50 = {ev[0]:.2f}/cm   "
            f"f10 = {ev[1]:.2f}/cm   "
            f"f02 = {ev[2]:.2f}/cm   "
            f"BG = {res['background_value']:.0f} HU   "
            f"peak = {res['psf_peak']:.0f} HU"
        )
        self.busy_label.setText("")

    @pyqtSlot(str)
    def _on_compute_failed(self, msg):
        self._compute_in_flight = False
        self.metrics_label.setText(f"⚠ {msg}")
        self.busy_label.setText("")

    # ---- Cleanup --------------------------------------------------------

    def closeEvent(self, a0):
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        super().closeEvent(a0)


# ============================================================================
# Entry point
# ============================================================================


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MTF GUI")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
