"""Rule-based particle identification from track morphology.

Cloud chamber rules of thumb:
  alpha    short (1-5 cm), fat, bright, straight     - heavy, slow, densely ionising
  muon     long, thin, ruler-straight, often edge-to-edge - cosmic ray, minimum ionising
  beta     thin, medium-long, wandering / kinked     - electron scattering off nuclei
  electron very short curly squiggle                 - low-energy electron (Compton, delta ray)

Each rule produces a score in 0..1; the best score wins, and the margin over
second place becomes the confidence. Thresholds live in ClassifyConfig.
"""
from __future__ import annotations

from .config import ClassifyConfig
from .detector import TrackEvent

KINDS = ("alpha", "beta", "muon", "electron")


def _ramp(v, lo, hi):
    """0 at lo, 1 at hi (works for lo > hi too)."""
    if hi == lo:
        return float(v >= hi)
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


class Classifier:
    def __init__(self, cfg: ClassifyConfig):
        self.c = cfg

    def scores(self, e: TrackEvent) -> dict[str, float]:
        c = self.c
        L, W, wig = e.path_mm, e.width_mm, e.wiggle
        straight = _ramp(wig, c.muon_max_wiggle * 2.5, c.muon_max_wiggle * 0.5)
        thin = _ramp(W, c.alpha_min_width_mm, c.muon_max_width_mm * 0.6)
        fat = _ramp(W, c.alpha_min_width_mm * 0.75, c.alpha_min_width_mm * 1.25)
        short = _ramp(L, c.short_max_mm * 1.4, c.short_max_mm * 0.8)
        not_short = 1.0 - short
        long_ = _ramp(L, c.muon_min_length_mm * 0.7, c.muon_min_length_mm * 1.2)
        if e.clipped:  # edge-to-edge tracks are probably longer than we see
            long_ = max(long_, 0.7 * _ramp(L, 15, c.muon_min_length_mm))
        alpha_len = _ramp(L, c.alpha_max_length_mm * 1.2, c.alpha_max_length_mm * 0.9)
        return {
            "alpha": fat * alpha_len * not_short ** 0.5,
            "muon": thin * straight * long_,
            "beta": thin * not_short * (1.0 - 0.8 * straight * long_),
            "electron": short * (1.0 - 0.7 * fat),
        }

    def classify(self, e: TrackEvent) -> TrackEvent:
        s = self.scores(e)
        ranked = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
        (k1, s1), (_, s2) = ranked[0], ranked[1]
        if s1 < 0.15:
            e.kind, e.confidence = "unknown", 0.0
        else:
            e.kind, e.confidence = k1, round(min(1.0, s1 - s2 + 0.5 * s1), 3)
        e.extra["scores"] = {k: round(v, 3) for k, v in s.items()}
        return e
