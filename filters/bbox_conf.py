# filters/bbox_conf.py
from __future__ import annotations
import numpy as np
from ..types import Observation, FilterScore

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _smoothstep(edge0, edge1, x):
    # 0..1
    t = _clamp01((x - edge0) / (edge1 - edge0 + 1e-12))
    return t * t * (3 - 2*t)

class BBoxConfFilter:
    """
    bbox_conf in [0,1]
    - hard constraints: min_h, min_area, ar range
    - soft penalties: clipping ratio, too-small size
    """
    def __init__(
        self,
        use_bbox_filter: bool = True,
        ar_min: float = 0.35,
        ar_max: float = 0.65,
        min_h: int = 100,
        min_area: int = 80*80,
        max_clip_ratio: float = 0.10,  # if bbox is clipped by >10% -> penalize strongly
    ):
        self.use = bool(use_bbox_filter)
        self.ar_min = float(ar_min)
        self.ar_max = float(ar_max)
        self.min_h = int(min_h)
        self.min_area = int(min_area)
        self.max_clip = float(max_clip_ratio)

    def eval(self, obs: Observation, frame_shape) -> FilterScore:
        if obs.bbox_xyxy is None:
            return FilterScore(False, 0.0, "no bbox")

        bbox = np.asarray(obs.bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size < 4:
            return FilterScore(False, 0.0, "bbox invalid")

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]

        # compute raw size
        bw_raw = max(0.0, x2 - x1)
        bh_raw = max(0.0, y2 - y1)
        if bw_raw <= 1.0 or bh_raw <= 1.0:
            return FilterScore(False, 0.0, "bbox degenerate")

        # clipping: how much bbox goes outside frame
        x1c = max(0.0, min(w, x1))
        x2c = max(0.0, min(w, x2))
        y1c = max(0.0, min(h, y1))
        y2c = max(0.0, min(h, y2))
        bw = max(0.0, x2c - x1c)
        bh = max(0.0, y2c - y1c)

        if bw <= 1.0 or bh <= 1.0:
            return FilterScore(False, 0.0, "bbox outside frame")

        ar = bw / (bh + 1e-12)
        area = bw * bh

        if not self.use:
            # still return a meaningful score
            score = _clamp01(0.5 + 0.5*_smoothstep(self.min_h, 2*self.min_h, bh))
            return FilterScore(True, score, f"bbox filter disabled ar={ar:.3f} area={area:.0f} h={bh:.1f}")

        # hard constraints
        if bh < self.min_h:
            return FilterScore(False, 0.0, f"bh<{self.min_h} (bh={bh:.1f})")
        if area < self.min_area:
            return FilterScore(False, 0.0, f"area<{self.min_area} (area={area:.0f})")
        if not (self.ar_min <= ar <= self.ar_max):
            return FilterScore(False, 0.0, f"ar out [{self.ar_min},{self.ar_max}] (ar={ar:.3f})")

        # soft score: size + clipping
        clip_w = (bw_raw - bw) / (bw_raw + 1e-12)
        clip_h = (bh_raw - bh) / (bh_raw + 1e-12)
        clip = max(0.0, clip_w, clip_h)  # worst axis
        # penalty: clip=0 ->1, clip=max_clip ->0.2, clip>max_clip ->0
        clip_score = _clamp01(1.0 - clip / (self.max_clip + 1e-12))
        clip_score = 0.2 + 0.8*clip_score  # keep a floor

        size_score = _smoothstep(self.min_h, 2*self.min_h, bh)  # 0..1

        bbox_conf = _clamp01(0.65*size_score + 0.35*clip_score)

        return FilterScore(True, bbox_conf, f"bbox_conf={bbox_conf:.3f} ar={ar:.3f} area={area:.0f} h={bh:.1f} clip={clip:.3f}")
