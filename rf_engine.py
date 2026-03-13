"""
rf_engine.py  —  Core RF computation engine
All antenna math: impedance, reflection coefficient, S11, VSWR, bandwidth.
"""

import numpy as np


# ─── Frequency sweep ──────────────────────────────────────────────────────────

def frequency_sweep(f_start: float, f_stop: float, n_points: int) -> np.ndarray:
    """Logarithmically spaced frequency array — matches real VNA sweep behaviour."""
    return np.logspace(np.log10(f_start), np.log10(f_stop), int(n_points))


# ─── Impedance model ─────────────────────────────────────────────────────────

def compute_antenna_impedance(freqs: np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    """
    Series RLC model:  Z_L(f) = R + j*(ωL - 1/ωC)
    Produces realistic resonance behaviour around f_res = 1/(2π√LC)
    """
    omega = 2.0 * np.pi * freqs
    # Guard against division-by-zero at f=0 (shouldn't happen with logspace)
    omega = np.where(omega == 0, 1e-30, omega)
    reactance = omega * L - 1.0 / (omega * C)
    return R + 1j * reactance


# ─── Reflection coefficient ───────────────────────────────────────────────────

def compute_gamma(Z_L: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Γ = (Z_L - Z0) / (Z_L + Z0)"""
    denom = Z_L + Z0
    # Prevent division by near-zero denominator
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return (Z_L - Z0) / denom


# ─── Derived quantities ───────────────────────────────────────────────────────

def compute_s11_db(gamma: np.ndarray) -> np.ndarray:
    """S11 (dB) = 20·log10|Γ|"""
    mag = np.clip(np.abs(gamma), 1e-15, None)
    return 20.0 * np.log10(mag)


def compute_vswr(gamma: np.ndarray) -> np.ndarray:
    """VSWR = (1+|Γ|)/(1-|Γ|).  Capped at 999 to avoid display blow-up."""
    mag = np.clip(np.abs(gamma), 0.0, 1.0 - 1e-9)
    return (1.0 + mag) / (1.0 - mag)


def compute_return_loss(gamma: np.ndarray) -> np.ndarray:
    """Return Loss (dB) = -S11(dB) = -20·log10|Γ|   (positive value = good)"""
    return -compute_s11_db(gamma)


# ─── Bandwidth detection ──────────────────────────────────────────────────────

def compute_bandwidth(freqs: np.ndarray, s11_db: np.ndarray,
                      threshold_db: float = -10.0) -> dict:
    """
    Find the -10 dB (or custom threshold) bandwidth around the resonance dip.

    Returns dict with keys:
        f_low, f_high, bandwidth, f_res, s11_min, valid
    """
    res_idx = int(np.argmin(s11_db))
    f_res   = freqs[res_idx]
    s11_min = s11_db[res_idx]

    result = dict(f_res=f_res, s11_min=s11_min, f_low=None,
                  f_high=None, bandwidth=None, valid=False)

    if s11_min > threshold_db:
        # Resonance never crosses the threshold — bandwidth undefined
        return result

    # Walk left from resonance to find lower crossing
    f_low = None
    for i in range(res_idx, -1, -1):
        if s11_db[i] >= threshold_db:
            # Linear interpolation between i and i+1
            if i + 1 <= res_idx:
                t = (threshold_db - s11_db[i]) / (s11_db[i+1] - s11_db[i] + 1e-30)
                f_low = freqs[i] + t * (freqs[i+1] - freqs[i])
            else:
                f_low = freqs[i]
            break

    # Walk right from resonance to find upper crossing
    f_high = None
    for i in range(res_idx, len(freqs)):
        if s11_db[i] >= threshold_db:
            if i > 0:
                t = (threshold_db - s11_db[i-1]) / (s11_db[i] - s11_db[i-1] + 1e-30)
                f_high = freqs[i-1] + t * (freqs[i] - freqs[i-1])
            else:
                f_high = freqs[i]
            break

    if f_low is not None and f_high is not None:
        result.update(f_low=f_low, f_high=f_high,
                      bandwidth=f_high - f_low, valid=True)
    return result


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_simulation(f_start: float, f_stop: float, n_points: int,
                   R: float, L: float, C: float, Z0: float = 50.0) -> dict:
    """Run the complete VNA simulation pipeline and return all computed arrays."""
    freqs = frequency_sweep(f_start, f_stop, n_points)
    Z_L   = compute_antenna_impedance(freqs, R, L, C)
    gamma = compute_gamma(Z_L, Z0)
    s11   = compute_s11_db(gamma)
    vswr  = compute_vswr(gamma)
    rl    = compute_return_loss(gamma)
    bw    = compute_bandwidth(freqs, s11)

    return dict(
        frequencies = freqs,
        Z_L         = Z_L,
        gamma       = gamma,
        s11_db      = s11,
        vswr        = vswr,
        return_loss = rl,
        bandwidth   = bw,
        Z0          = Z0,
    )
