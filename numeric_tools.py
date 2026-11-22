"""
Numeric distribution helpers (no API calls).

Takes the percentiles a forecaster wrote (ideally 1..99) and returns a
NumericDistribution that is
- smooth (monotone PCHIP interpolation in the question's CDF-location space,
  i.e. log space for log-scaled questions),
- has explicit tails (extrapolated past the outermost declared percentiles,
  clipped at closed bounds, allowed past open bounds),
- mixed with a small uniform component over the question range
  ("tail humility"), and
- unit-checked (a sample whose median is orders of magnitude off the range is
  rescaled by a power of 1000 if that fits, else rejected).
If anything fails, the caller falls back to the template behaviour.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from forecasting_tools import NumericDistribution, NumericQuestion, Percentile

logger = logging.getLogger(__name__)

UNIFORM_WEIGHT = 0.05
TAIL_STRETCH = 1.5
OUTPUT_HEIGHTS = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995,
]


class _Scale:
    def __init__(self, lo: float, hi: float, zero_point: float | None):
        self.lo, self.hi, self.zp = lo, hi, zero_point
        self.log = zero_point is not None
        if self.log:
            self.r = (hi - zero_point) / (lo - zero_point)

    def to_loc(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if not self.log:
            return (v - self.lo) / (self.hi - self.lo)
        rng = self.hi - self.lo
        inner = (v - self.lo) * (self.r - 1) + rng
        inner = np.maximum(inner, 1e-12)
        return (np.log(inner) - math.log(rng)) / math.log(self.r)

    def to_val(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.log:
            return self.lo + x * (self.hi - self.lo)
        rng = self.hi - self.lo
        return self.lo + (rng * np.power(self.r, x) - rng) / (self.r - 1)


def _pchip(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Monotone piecewise cubic Hermite (Fritsch-Carlson)."""
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for k in range(1, n - 1):
            if d[k - 1] * d[k] <= 0:
                m[k] = 0.0
            else:
                w1 = 2 * h[k] + h[k - 1]
                w2 = h[k] + 2 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        m[0] = d[0]
        m[-1] = d[-1]
    yq = np.empty_like(xq)
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    t = (xq - x[idx]) / h[idx]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    yq = h00 * y[idx] + h10 * h[idx] * m[idx] + h01 * y[idx + 1] + h11 * h[idx] * m[idx + 1]
    yq = np.where(xq <= x[0], y[0], yq)
    yq = np.where(xq >= x[-1], y[-1], yq)
    return yq


def unit_check(values: np.ndarray, scale: _Scale) -> np.ndarray:
    """Return values, rescaled (powers of 1000, or x100 / /100 for percent mixups)
    if the median sits far outside the question range and a rescale fits it."""
    med_loc = float(np.median(scale.to_loc(values)))
    if -0.5 <= med_loc <= 1.5:
        return values
    best = None
    for f in [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e2, 1e3, 1e6, 1e9, 1e12]:
        loc = float(np.median(scale.to_loc(values * f)))
        if -0.1 <= loc <= 1.1 and (best is None or abs(loc - 0.5) < best[1]):
            best = (f, abs(loc - 0.5))
    if best is not None:
        logger.warning(
            f"Unit check: forecast median {np.median(values)} looks off-scale for range "
            f"[{scale.lo}, {scale.hi}]; rescaling by {best[0]}"
        )
        return values * best[0]
    if -1.5 <= med_loc <= 2.5:
        return values
    raise ValueError(
        f"Unit check failed: median of forecast {np.median(values)} is far outside "
        f"range [{scale.lo}, {scale.hi}]; rejecting this sample"
    )


def smooth_distribution(percentiles: list[Percentile], question: NumericQuestion) -> NumericDistribution:
    lo, hi = float(question.lower_bound), float(question.upper_bound)
    scale = _Scale(lo, hi, question.zero_point)
    open_lo, open_hi = bool(question.open_lower_bound), bool(question.open_upper_bound)

    pts = sorted(((float(p.percentile), float(p.value)) for p in percentiles), key=lambda t: t[0])
    # dedupe heights
    hs, vs = [], []
    for h, v in pts:
        if 0 < h < 1 and (not hs or h > hs[-1] + 1e-6):
            hs.append(h)
            vs.append(v)
    if len(hs) < 3:
        raise ValueError("Need at least 3 percentiles to smooth")
    vals = np.maximum.accumulate(np.array(vs))
    vals = unit_check(vals, scale)
    heights = np.array(hs)

    x = scale.to_loc(vals)
    # closed bounds: nothing can lie beyond them
    if not open_lo:
        x = np.maximum(x, 0.0)
    if not open_hi:
        x = np.minimum(x, 1.0)
    # enforce strictly increasing locations
    span = max(x[-1] - x[0], 1e-3)
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + span * 1e-4

    # tail anchors (height 0 / 1), stretched for humility
    slope_lo = (heights[1] - heights[0]) / max(x[1] - x[0], 1e-6)
    slope_hi = (heights[-1] - heights[-2]) / max(x[-1] - x[-2], 1e-6)
    x_lo = x[0] - TAIL_STRETCH * heights[0] / max(slope_lo, 1e-6)
    x_hi = x[-1] + TAIL_STRETCH * (1 - heights[-1]) / max(slope_hi, 1e-6)
    x_lo = max(x_lo, x[0] - 1.0)
    x_hi = min(x_hi, x[-1] + 1.0)
    if not open_lo:
        x_lo = max(x_lo, 0.0)
    if not open_hi:
        x_hi = min(x_hi, 1.0)
    xs = [x_lo] if x_lo < x[0] - 1e-9 else []
    ys = [0.0] if xs else []
    xs += list(x)
    ys += list(heights)
    if x_hi > x[-1] + 1e-9:
        xs.append(x_hi)
        ys.append(1.0)
    xs, ys = np.array(xs), np.array(ys)
    if ys[0] > 0:  # closed lower bound with mass at the bound
        xs = np.insert(xs, 0, xs[0] - 1e-6)
        ys = np.insert(ys, 0, 0.0)
    if ys[-1] < 1:
        xs = np.append(xs, xs[-1] + 1e-6)
        ys = np.append(ys, 1.0)

    grid = np.linspace(min(xs[0], 0.0), max(xs[-1], 1.0), 4001)
    F = _pchip(xs, ys, grid)
    U = np.clip(grid, 0.0, 1.0)
    F = (1 - UNIFORM_WEIGHT) * F + UNIFORM_WEIGHT * U
    F = np.maximum.accumulate(F) + np.linspace(0, 1e-9, len(F))

    out_x = np.interp(OUTPUT_HEIGHTS, F, grid)
    out_v = scale.to_val(out_x)

    rng = abs(hi - lo)
    buffer = 1 if rng > 100 else 0.01 * rng
    final: list[Percentile] = []
    for h, v in zip(OUTPUT_HEIGHTS, out_v):
        if not open_lo and v <= lo + buffer * 1.05:
            continue
        if not open_hi and v >= hi - buffer * 1.05:
            continue
        if final and v <= final[-1].value + rng * 1e-6:
            continue
        final.append(Percentile(percentile=float(h), value=float(v)))
    if len(final) < 3:
        raise ValueError("Too few usable percentiles after smoothing")
    dist = NumericDistribution.from_question(final, question)
    dist.get_cdf()  # validate now so failures fall back cleanly
    return dist
