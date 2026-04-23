# filters/distance_conf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
import cv2

from ..types import Observation, FilterScore

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _smoothstep(a: float, b: float, x: float) -> float:
    t = _clamp01((x - a) / (b - a + 1e-12))
    return t * t * (3 - 2*t)

@dataclass
class DepthIOConfig:
    # depth format handling
    depth_unit: str = "auto"     # "auto" | "m" | "mm"
    mm_to_m: float = 0.001

    # invalid handling
    invalid_is_zero: bool = True
    min_valid_m: float = 0.1
    max_valid_m: float = 20.0

    # patch sampling
    patch_radius: int = 4        # radius=4 -> 9x9 patch
    min_valid_count: int = 6     # at least N valid pixels in patch

@dataclass
class DistanceFilterConfig:
    enabled: bool = True
    min_cam_dist_m: float = 3.5
    max_cam_dist_m: float = 7.0
    margin_m: float = 0.5                # soft score margin
    unknown_policy_pass: bool = True     # if depth missing/invalid -> pass?
    # head point from pose
    min_kpt_conf: float = 0.15

class DepthDistanceEstimator:
    """
    Estimate camera distance using depth map aligned to RGB image.
    - Robust: median of valid depths within patch around head point.
    """
    def __init__(
        self,
        io_cfg: Optional[DepthIOConfig] = None,
        kpt_min_conf: float = 0.15,
        depth_loader: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.io = io_cfg or DepthIOConfig()
        self.kpt_min_conf = float(kpt_min_conf)
        self.depth_loader = depth_loader or self._default_load_depth

    def _default_load_depth(self, path: str) -> np.ndarray:
        # Try numpy first
        if path.endswith(".npy"):
            arr = np.load(path)
            return np.asarray(arr)
        # png: often uint16 depth in mm
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        return d

    def _pick_head_point(self, obs: Observation, frame_shape: Tuple[int,int,int]) -> Optional[Tuple[int,int]]:
        h, w = frame_shape[:2]

        kp = obs.keypoints_xy
        kc = obs.keypoints_conf
        if kp is not None and kc is not None:
            kp = np.asarray(kp, dtype=np.float32)
            kc = np.asarray(kc, dtype=np.float32).reshape(-1)
            K = min(len(kp), len(kc))

            def valid_xy(i: int) -> Optional[Tuple[float,float]]:
                if 0 <= i < K and float(kc[i]) >= self.kpt_min_conf:
                    return float(kp[i,0]), float(kp[i,1])
                return None

            # COCO-17 common: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear
            nose = valid_xy(0)
            if nose is not None:
                x, y = nose
                return int(round(_clamp01(x/(w-1))* (w-1))), int(round(_clamp01(y/(h-1))* (h-1)))

            le = valid_xy(1); re = valid_xy(2)
            if le is not None and re is not None:
                x = 0.5*(le[0] + re[0]); y = 0.5*(le[1] + re[1])
                return int(round(max(0, min(w-1, x)))), int(round(max(0, min(h-1, y))))

            le = valid_xy(3); re = valid_xy(4)
            if le is not None and re is not None:
                x = 0.5*(le[0] + re[0]); y = 0.5*(le[1] + re[1])
                return int(round(max(0, min(w-1, x)))), int(round(max(0, min(h-1, y))))

        # fallback to bbox-based head point
        if obs.bbox_xyxy is None:
            return None
        bb = np.asarray(obs.bbox_xyxy, dtype=np.float32).reshape(-1)
        if bb.size < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        cx = 0.5*(x1 + x2)
        bh = max(1.0, (y2 - y1))
        hy = y1 + 0.15*bh  # upper area as "head"
        x = int(round(max(0, min(w-1, cx))))
        y = int(round(max(0, min(h-1, hy))))
        return x, y

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        d = np.asarray(depth)

        # auto detect
        if self.io.depth_unit == "auto":
            if d.dtype == np.uint16 or d.dtype == np.uint32:
                # commonly mm
                d_m = d.astype(np.float32) * float(self.io.mm_to_m)
            else:
                # assume meters
                d_m = d.astype(np.float32)
        elif self.io.depth_unit == "mm":
            d_m = d.astype(np.float32) * float(self.io.mm_to_m)
        else:  # "m"
            d_m = d.astype(np.float32)

        return d_m

    def estimate_distance_m(self, obs: Observation, frame_shape: Tuple[int,int,int]) -> Tuple[Optional[float], str]:
        # obtain depth map
        if obs.depth is not None:
            depth = obs.depth
        elif obs.depth_path is not None:
            try:
                depth = self.depth_loader(obs.depth_path)
            except Exception as e:
                return None, f"depth load failed: {e}"
        else:
            return None, "no depth provided"

        depth = np.asarray(depth)
        if depth.ndim == 3:
            # sometimes depth is (H,W,1)
            depth = depth[..., 0]
        if depth.ndim != 2:
            return None, f"depth shape invalid: {depth.shape}"

        H, W = depth.shape[:2]
        fh, fw = frame_shape[:2]
        if (H != fh) or (W != fw):
            # If size mismatch, assume aligned but different resolution -> scale point accordingly
            # (You can also enforce strict match if you prefer.)
            pass

        # pick head point in RGB coords
        p = self._pick_head_point(obs, frame_shape)
        if p is None:
            return None, "no head point (no pose and no bbox)"

        x_rgb, y_rgb = p

        # map to depth coords if different size
        if (H != fh) or (W != fw):
            sx = W / float(fw)
            sy = H / float(fh)
            x = int(round(x_rgb * sx))
            y = int(round(y_rgb * sy))
            x = max(0, min(W-1, x))
            y = max(0, min(H-1, y))
        else:
            x, y = x_rgb, y_rgb

        d_m = self._depth_to_meters(depth)

        r = int(self.io.patch_radius)
        x1 = max(0, x - r); x2 = min(W, x + r + 1)
        y1 = max(0, y - r); y2 = min(H, y + r + 1)

        patch = d_m[y1:y2, x1:x2].reshape(-1)

        # filter invalid values
        valid = patch
        if self.io.invalid_is_zero:
            valid = valid[valid > 0.0]
        valid = valid[np.isfinite(valid)]
        valid = valid[(valid >= float(self.io.min_valid_m)) & (valid <= float(self.io.max_valid_m))]

        if valid.size < int(self.io.min_valid_count):
            return None, f"not enough valid depth ({valid.size}) at ({x},{y})"

        dist = float(np.median(valid))
        return dist, f"median depth patch={2*r+1}x{2*r+1} at ({x},{y}) valid={valid.size}"

class DistanceConfFilter:
    """
    Uses DepthDistanceEstimator to fill obs.cam_dist_m and then filter.
    """
    def __init__(self, cfg: DistanceFilterConfig, io_cfg: Optional[DepthIOConfig] = None):
        self.cfg = cfg
        self.est = DepthDistanceEstimator(io_cfg=io_cfg, kpt_min_conf=cfg.min_kpt_conf)

    def eval(self, obs: Observation, frame_shape) -> FilterScore:
        if not self.cfg.enabled:
            return FilterScore(True, 1.0, "distance filter disabled")

        # If already present, keep it; otherwise estimate from depth
        d = None
        reason = ""
        if obs.has_dist and obs.cam_dist_m >= 0.0:
            d = float(obs.cam_dist_m)
            reason = "precomputed"
        else:
            d, reason = self.est.estimate_distance_m(obs, frame_shape)
            if d is not None:
                obs.cam_dist_m = float(d)
                obs.has_dist = True

        if d is None:
            ok = bool(self.cfg.unknown_policy_pass)
            return FilterScore(ok, -1.0, f"unknown distance -> {'pass' if ok else 'fail'} ({reason})")

        dmin = float(self.cfg.min_cam_dist_m)
        dmax = float(self.cfg.max_cam_dist_m)
        ok = (d >= dmin) and (d < dmax)

        # soft score with margin
        m = float(self.cfg.margin_m)
        if d < dmin:
            score = _clamp01((d - (dmin - m)) / (m + 1e-12))
        elif d >= dmax:
            score = _clamp01(((dmax + m) - d) / (m + 1e-12))
        else:
            score = 1.0

        return FilterScore(ok, float(score), f"d={d:.2f} range=[{dmin},{dmax}) score={score:.2f} ({reason})")
