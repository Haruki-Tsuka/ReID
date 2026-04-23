# filters/front_conf.py
from __future__ import annotations
import numpy as np
from ..types import Observation, FilterScore

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))

class FrontConfFilter:
    """
    front_conf in [0,1]
    If pose is missing -> score = -1 (unknown), ok depends on policy.
    """
    def __init__(
        self,
        require_front: bool = True,
        front_conf_thresh: float = 0.90,
        min_kpt_conf_face: float = 0.15,
        min_kpt_conf_body: float = 0.15,
        mirror: bool = False,
        unknown_policy_pass: bool = True,
    ):
        self.require_front = require_front
        self.th = float(front_conf_thresh)
        self.min_face = float(min_kpt_conf_face)
        self.min_body = float(min_kpt_conf_body)
        self.mirror = bool(mirror)
        self.unknown_pass = bool(unknown_policy_pass)

    def eval(self, obs: Observation) -> FilterScore:
        if not self.require_front:
            return FilterScore(True, 1.0, "front filter disabled")

        # if upstream already computed front_conf, trust it
        if obs.front_conf is not None:
            fc = float(obs.front_conf)
            return FilterScore(fc >= self.th, _clamp01(fc), f"precomputed fc={fc:.3f} th={self.th}")

        kp = obs.keypoints_xy
        kc = obs.keypoints_conf
        if kp is None or kc is None:
            # unknown
            ok = self.unknown_pass
            return FilterScore(ok, -1.0, "no pose -> unknown")

        kp = np.asarray(kp, dtype=np.float32)
        kc = np.asarray(kc, dtype=np.float32)
        K = min(len(kp), len(kc))

        def conf(i):
            if 0 <= i < K:
                return float(kc[i])
            return None

        def xy(i):
            if 0 <= i < K:
                return float(kp[i,0]), float(kp[i,1])
            return None

        # COCO-17 indices (common): 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 11 l_hip, 12 r_hip
        face_ids = [0,1,2,3,4]
        face_confs = [conf(i) for i in face_ids]
        face_confs = [c for c in face_confs if c is not None and c >= self.min_face]
        face_score = _clamp01(_safe_mean(face_confs) or 0.0)

        # left-right shoulder order vote
        ls = xy(5); rs = xy(6)
        lh = xy(11); rh = xy(12)

        body_votes = []
        def lr_vote(a, b):
            # expects a = left, b = right in image coords
            if a is None or b is None:
                return None
            ax, _ = a; bx, _ = b
            # in many datasets, "left" is person's left (viewer mirrored). allow mirror option
            sgn = -1.0 if self.mirror else 1.0
            # front tends to have left keypoint on image-right of right keypoint? depends on convention.
            # We define: after applying mirror, "front" means ax > bx (left-shoulder appears right side of image).
            return 1.0 if (sgn*ax) > (sgn*bx) else 0.0

        # vote from shoulders and hips
        if ls is not None and rs is not None and conf(5) is not None and conf(6) is not None:
            if conf(5) >= self.min_body and conf(6) >= self.min_body:
                body_votes.append(lr_vote(ls, rs))
        if lh is not None and rh is not None and conf(11) is not None and conf(12) is not None:
            if conf(11) >= self.min_body and conf(12) >= self.min_body:
                body_votes.append(lr_vote(lh, rh))

        lr_score = _clamp01(_safe_mean([v for v in body_votes if v is not None]) or 0.0)

        # nose centered between shoulders
        nose = xy(0)
        if nose is not None and ls is not None and rs is not None:
            nx, _ = nose
            lxs, _ = ls; rxs, _ = rs
            midx = 0.5*(lxs + rxs)
            shoulder_w = abs(lxs - rxs) + 1e-6
            dev = abs(nx - midx) / shoulder_w
            # closer is better; dev=0 -> 1, dev>=0.6 -> 0
            nose_center_score = _clamp01(1.0 - dev/0.6)
        else:
            nose_center_score = 0.0

        # Final (you can cite this as your method)
        fc = _clamp01(0.55*face_score + 0.30*lr_score + 0.15*nose_center_score)

        obs.front_conf = fc
        ok = fc >= self.th
        return FilterScore(ok, fc, f"fc={fc:.3f} face={face_score:.2f} lr={lr_score:.2f} noseC={nose_center_score:.2f} th={self.th}")
