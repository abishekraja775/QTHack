"""
plot_s11.py  —  S11 / VSWR / Return-Loss / |Γ| semilog canvas
Professional VNA-style interactive plot with markers and bandwidth shading.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker


# ── Colour palette (Catppuccin Mocha) ────────────────────────────────────────
C = dict(
    bg      = "#1e1e2e",
    axes_bg = "#181825",
    grid    = "#313244",
    border  = "#45475a",
    text    = "#cdd6f4",
    subtext = "#a6adc8",
    blue    = "#89b4fa",
    red     = "#f38ba8",
    green   = "#a6e3a1",
    yellow  = "#f9e2af",
    mauve   = "#cba6f7",
    peach   = "#fab387",
    teal    = "#94e2d5",
)

PLOT_MODES = ["S11 (dB)", "VSWR", "Return Loss (dB)", "Reflection Coeff |Γ|"]


def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


class S11Canvas(FigureCanvas):
    """Embeddable Matplotlib canvas — semilog RF plot with interactive markers."""

    def __init__(self, parent=None, dpi=95):
        self.fig = Figure(figsize=(6, 4), dpi=dpi, facecolor=C["bg"])
        self.ax  = self.fig.add_subplot(111)
        self._result     = None
        self._mode       = "S11 (dB)"
        self._marker_artists = []  # annotation objects to clear on re-draw
        self._bw_patch   = None

        self._init_axes()
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Axes styling ──────────────────────────────────────────────────────────

    def _init_axes(self):
        ax = self.ax
        ax.set_facecolor(C["axes_bg"])
        ax.tick_params(colors=C["text"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.title.set_color(C["text"])
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, color=C["grid"])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("S11 (dB)")
        ax.set_title("S11 vs Frequency")
        self.fig.tight_layout(pad=1.8)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        self._mode = mode
        if self._result is not None:
            self.plot_result(self._result)

    def plot_result(self, result: dict):
        self._result = result
        self._redraw()

    # ── Core drawing ─────────────────────────────────────────────────────────

    def _redraw(self):
        result = self._result
        freqs  = result["frequencies"]
        mode   = self._mode

        # Select Y data based on mode
        if mode == "S11 (dB)":
            y      = result["s11_db"]
            ylabel = "S11 (dB)"
            ycolor = C["blue"]
        elif mode == "VSWR":
            y      = result["vswr"]
            ylabel = "VSWR"
            ycolor = C["peach"]
        elif mode == "Return Loss (dB)":
            y      = result["return_loss"]
            ylabel = "Return Loss (dB)"
            ycolor = C["green"]
        else:  # |Γ|
            y      = np.abs(result["gamma"])
            ylabel = "|Γ|"
            ycolor = C["mauve"]

        self.ax.cla()
        self._init_axes()
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f"{ylabel} vs Frequency")
        self.ax.set_xscale("log")

        # Main trace
        self.ax.plot(freqs, y, color=ycolor, linewidth=1.8, zorder=3)

        # Reference lines (only for dB modes)
        if mode in ("S11 (dB)", "Return Loss (dB)"):
            sign = -1 if mode == "S11 (dB)" else 1
            for ref, col in [(-3*sign, C["peach"]), (-10*sign, C["green"]),
                             (-20*sign, C["teal"])]:
                self.ax.axhline(ref, color=col, linestyle="--",
                                linewidth=0.7, alpha=0.6,
                                label=f"{ref:+.0f} dB")
            self.ax.legend(fontsize=7, facecolor=C["grid"],
                           labelcolor=C["text"], edgecolor=C["border"],
                           loc="upper right")

        # Resonance marker
        res_idx = int(np.argmin(result["s11_db"]))
        res_f   = freqs[res_idx]
        res_y   = y[res_idx]
        self.ax.axvline(res_f, color=C["red"], linestyle=":",
                        linewidth=1.0, alpha=0.8, zorder=2)
        self.ax.plot(res_f, res_y, "o", color=C["red"],
                     markersize=7, zorder=5,
                     label=f"Res: {_fmt_freq(res_f)}")

        # -10 dB bandwidth shading (only in S11 mode)
        if mode == "S11 (dB)":
            bw = result["bandwidth"]
            if bw["valid"]:
                self.ax.axvspan(bw["f_low"], bw["f_high"],
                                alpha=0.12, color=C["blue"], zorder=1)
                self.ax.axvline(bw["f_low"],  color=C["blue"],
                                linestyle="--", linewidth=0.8, alpha=0.5)
                self.ax.axvline(bw["f_high"], color=C["blue"],
                                linestyle="--", linewidth=0.8, alpha=0.5)

        self._marker_artists = []
        self.fig.tight_layout(pad=1.8)
        self.draw()

    # ── Interactive frequency marker ──────────────────────────────────────────

    def place_marker(self, x_freq: float):
        """Called on mouse click — interpolates and annotates the clicked frequency."""
        if self._result is None:
            return None

        result = self._result
        freqs  = result["frequencies"]
        idx    = int(np.argmin(np.abs(freqs - x_freq)))
        f      = freqs[idx]

        # Remove previous marker annotations
        for artist in self._marker_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._marker_artists = []

        mode = self._mode
        if mode == "S11 (dB)":         y_val = result["s11_db"][idx]
        elif mode == "VSWR":           y_val = result["vswr"][idx]
        elif mode == "Return Loss (dB)": y_val = result["return_loss"][idx]
        else:                          y_val = abs(result["gamma"][idx])

        vline = self.ax.axvline(f, color=C["mauve"], linestyle="-.",
                                linewidth=1.0, alpha=0.9, zorder=6)
        dot   = self.ax.plot(f, y_val, "D", color=C["mauve"],
                             markersize=7, zorder=7)[0]

        Z_L   = result["Z_L"][idx]
        Z0    = result["Z0"]
        s11   = result["s11_db"][idx]
        vswr  = result["vswr"][idx]
        gmag  = abs(result["gamma"][idx])

        label = (f"▸ {_fmt_freq(f)}\n"
                 f"  S11 = {s11:.2f} dB\n"
                 f"  VSWR = {vswr:.3f}\n"
                 f"  |Γ| = {gmag:.4f}\n"
                 f"  Z = {Z_L.real:.1f}{Z_L.imag:+.1f}j Ω\n"
                 f"  z = {Z_L.real/Z0:.3f}{Z_L.imag/Z0:+.3f}j")

        ylim = self.ax.get_ylim()
        ann  = self.ax.annotate(
            label,
            xy=(f, y_val),
            xytext=(12, -12),
            textcoords="offset points",
            fontsize=7.5,
            color=C["mauve"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg"],
                      edgecolor=C["mauve"], alpha=0.92),
            zorder=8,
        )

        self._marker_artists = [vline, dot, ann]
        self.draw()

        return dict(f=f, s11=s11, vswr=vswr, gmag=gmag, Z_L=Z_L)
