"""
smith_chart.py  —  Smith chart canvas with frequency markers & key-point highlights
Uses scikit-rf when available; falls back to hand-drawn grid.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

C = dict(
    bg      = "#1e1e2e",
    axes_bg = "#181825",
    grid    = "#2a2a3c",
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


def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


class SmithCanvas(FigureCanvas):

    def __init__(self, parent=None, dpi=95):
        self.fig = Figure(figsize=(5, 4), dpi=dpi, facecolor=C["bg"])
        self.ax  = self.fig.add_subplot(111)
        self._result          = None
        self._marker_artists  = []
        self._draw_empty()
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Empty grid ────────────────────────────────────────────────────────────

    def _draw_empty(self):
        ax = self.ax
        ax.set_facecolor(C["axes_bg"])
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.tick_params(colors=C["text"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.set_title("Smith Chart  (Γ plane)", color=C["text"], fontsize=9)
        ax.set_xlabel("Re(Γ)", color=C["subtext"], fontsize=8)
        ax.set_ylabel("Im(Γ)", color=C["subtext"], fontsize=8)
        self._draw_smith_grid(ax)
        self.fig.tight_layout(pad=1.5)

    @staticmethod
    def _draw_smith_grid(ax):
        theta = np.linspace(0, 2 * np.pi, 400)

        # Outer unit circle
        ax.plot(np.cos(theta), np.sin(theta), color=C["border"], linewidth=1.4)

        # Constant-resistance circles  r = R/(R+1),  radius = 1/(R+1)
        for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
            cx  = r / (r + 1.0)
            rad = 1.0 / (r + 1.0)
            x   = cx + rad * np.cos(theta)
            y   =      rad * np.sin(theta)
            mask = x**2 + y**2 <= 1.002
            ax.plot(np.where(mask, x, np.nan),
                    np.where(mask, y, np.nan),
                    color=C["grid"], linewidth=0.7)
            if r > 0:
                ax.text(cx + rad + 0.015, 0.015, f"{r}",
                        fontsize=5.5, color=C["border"], ha="left")

        # Constant-reactance arcs
        t = np.linspace(0, 2 * np.pi, 600)
        for x_val in [0.2, 0.5, 1.0, 2.0, 5.0]:
            for sign in [1, -1]:
                cx, cy = 1.0, sign / x_val
                rad    = abs(1.0 / x_val)
                pts    = np.column_stack([cx + rad * np.cos(t),
                                          cy + rad * np.sin(t)])
                inside = pts[:, 0]**2 + pts[:, 1]**2 <= 1.002
                ax.plot(np.where(inside, pts[:, 0], np.nan),
                        np.where(inside, pts[:, 1], np.nan),
                        color=C["grid"], linewidth=0.7)

        # Real axis & centre
        ax.axhline(0, color=C["grid"], linewidth=0.7)
        ax.plot(0, 0, "+", color=C["border"], markersize=8)

    # ── Plot full result ──────────────────────────────────────────────────────

    def plot_result(self, result: dict):
        self._result = result
        self.ax.cla()
        self._draw_empty()

        freqs = result["frequencies"]
        gamma = result["gamma"]
        Z_L   = result["Z_L"]
        Z0    = result["Z0"]
        re    = gamma.real
        im    = gamma.imag
        n     = len(re)

        if HAS_SKRF:
            self._plot_via_skrf(freqs, gamma)
        else:
            self._plot_gradient_trace(re, im, n)

        # Key-point markers
        res_idx = int(np.argmin(result["s11_db"]))

        # Start
        self.ax.plot(re[0], im[0], "o", color=C["green"],
                     markersize=8, zorder=6, label=f"Start {_fmt_freq(freqs[0])}")
        # Resonance
        self.ax.plot(re[res_idx], im[res_idx], "*", color=C["yellow"],
                     markersize=11, zorder=7,
                     label=f"Res {_fmt_freq(freqs[res_idx])}")
        # End
        self.ax.plot(re[-1], im[-1], "s", color=C["red"],
                     markersize=7, zorder=6, label=f"Stop {_fmt_freq(freqs[-1])}")

        self.ax.legend(fontsize=6.5, facecolor=C["grid"],
                       labelcolor=C["text"], edgecolor=C["border"],
                       loc="lower left")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def _plot_via_skrf(self, freqs, gamma):
        n       = len(freqs)
        s_array = gamma.reshape(n, 1, 1)
        freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
        net = skrf.Network(frequency=freq_obj, s=s_array)
        try:
            net.plot_s_smith(ax=self.ax, show_legend=False,
                             draw_labels=False, chart_type="z",
                             color=C["blue"], linewidth=1.8)
        except Exception:
            self._plot_gradient_trace(gamma.real, gamma.imag, n)

    def _plot_gradient_trace(self, re, im, n):
        """Colour-gradient trace: blue (low freq) → red (high freq)."""
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            col = (t * 0.9, 0.25, 1.0 - t * 0.9, 0.85)
            self.ax.plot(re[i:i+2], im[i:i+2], color=col, linewidth=1.8)

    # ── Interactive click marker ──────────────────────────────────────────────

    def place_marker(self, x: float, y: float):
        """Find nearest point on trace and annotate it."""
        if self._result is None:
            return None

        gamma  = self._result["gamma"]
        freqs  = self._result["frequencies"]
        Z_L    = self._result["Z_L"]
        Z0     = self._result["Z0"]

        dists  = (gamma.real - x)**2 + (gamma.imag - y)**2
        idx    = int(np.argmin(dists))
        g      = gamma[idx]
        f      = freqs[idx]
        zl     = Z_L[idx]

        for art in self._marker_artists:
            try: art.remove()
            except Exception: pass
        self._marker_artists = []

        dot = self.ax.plot(g.real, g.imag, "D",
                           color=C["mauve"], markersize=8, zorder=8)[0]
        label = (f"▸ {_fmt_freq(f)}\n"
                 f"  Z = {zl.real:.2f}{zl.imag:+.2f}j Ω\n"
                 f"  z = {zl.real/Z0:.3f}{zl.imag/Z0:+.3f}j\n"
                 f"  |Γ| = {abs(g):.4f}")
        ann = self.ax.annotate(
            label, xy=(g.real, g.imag),
            xytext=(10, 10), textcoords="offset points",
            fontsize=7, color=C["mauve"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor=C["bg"],
                      edgecolor=C["mauve"], alpha=0.92),
            zorder=9,
        )
        self._marker_artists = [dot, ann]
        self.draw()
        return dict(f=f, Z_L=zl, gamma=g)
