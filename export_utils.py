"""
export_utils.py  —  CSV export of simulation results
"""

import csv
import numpy as np


def export_csv(path: str, result: dict) -> None:
    """
    Write all computed VNA data to a CSV file.

    Columns:
        Frequency (Hz), Re(Z), Im(Z), |Z|,
        Gamma_Re, Gamma_Im, |Gamma|,
        S11 (dB), VSWR, Return Loss (dB)
    """
    freqs = result["frequencies"]
    Z_L   = result["Z_L"]
    gamma = result["gamma"]
    s11   = result["s11_db"]
    vswr  = result["vswr"]
    rl    = result["return_loss"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frequency (Hz)", "Re(Z) (Ohm)", "Im(Z) (Ohm)", "|Z| (Ohm)",
            "Gamma_Re", "Gamma_Im", "|Gamma|",
            "S11 (dB)", "VSWR", "Return Loss (dB)"
        ])
        for i in range(len(freqs)):
            g  = gamma[i]
            zl = Z_L[i]
            writer.writerow([
                f"{freqs[i]:.6e}",
                f"{zl.real:.6f}", f"{zl.imag:.6f}", f"{abs(zl):.6f}",
                f"{g.real:.8f}",  f"{g.imag:.8f}",  f"{abs(g):.8f}",
                f"{s11[i]:.6f}",  f"{vswr[i]:.6f}", f"{rl[i]:.6f}",
            ])
