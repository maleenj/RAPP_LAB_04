"""One-Euro filter for adaptive real-time signal smoothing.

When the signal changes slowly (actor still), the cutoff frequency drops and
smoothing is heavy — killing slow drift.  When the signal changes fast (actor
moving), the cutoff rises and the filter becomes responsive.

Reference:
    Casiez, Roussel, Vogel. "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems." CHI 2012.

Typical usage for 6-joint robot at 15 Hz::

    filt = OneEuroFilter(n_signals=6, rate=15.0,
                         min_cutoff=0.3, beta=0.01, d_cutoff=1.0)
    for target in targets:
        smoothed = filt(target)
"""

from __future__ import annotations

import math
import numpy as np


def _smoothing_factor(rate: float, cutoff: float) -> float:
    """Compute α for a simple low-pass filter given rate and cutoff freq."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


class OneEuroFilter:
    """Vectorised 1€ filter for N signals sampled at a fixed rate.

    Parameters
    ----------
    n_signals : int
        Number of parallel signals (e.g. 6 for joints).
    rate : float
        Sampling rate in Hz (e.g. 15.0).
    min_cutoff : float
        Minimum cutoff frequency (Hz).  Lower → smoother when slow.
        Good starting range: 0.15–1.0.
    beta : float
        Speed coefficient.  Higher → more responsive when fast.
        Good starting range: 0.001–0.1.
    d_cutoff : float
        Cutoff frequency for the derivative low-pass filter (Hz).
        Usually left at 1.0.
    """

    def __init__(
        self,
        n_signals: int,
        rate: float,
        min_cutoff: float = 0.3,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self.n = n_signals
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear filter state (next call re-initialises)."""
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter one sample.  Returns smoothed array of same shape."""
        x = np.asarray(x, dtype=np.float64)

        if self._x_prev is None:
            # First sample — initialise state, pass through unchanged
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(self.n, dtype=np.float64)
            return x.copy()

        # Derivative estimate (low-pass filtered)
        dx = (x - self._x_prev) * self.rate
        a_d = _smoothing_factor(self.rate, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff: rises with signal speed
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Smoothed signal
        a = np.array([_smoothing_factor(self.rate, c) for c in cutoff])
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()

        return x_hat
