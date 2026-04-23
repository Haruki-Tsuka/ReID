# filters/visual_conf.py
from __future__ import annotations

import numpy as np
from ..types import Observation, FilterScore


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _iou_xyxy(a, b) -> float:
    """
    a,b: array-like (4,) [x1,y1,x2,y2]
    returns IoU in [0,1]
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
    b = np.asarray(b, dtype=np.float32).reshape(-1)[:4]

    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12

    return float(inter / union)


class VisualConfFilter:
    """
    visual_conf = Cr = (1 - maxIoU) * (kpvisible/kpall) * Ck

    - maxIoU: 同一フレーム内の他bboxとのIoU最大
             ※obs.all_bboxes_xyxy (N,4) が必要（同一フレーム分のみ）
    - kpvisible/kpall: conf>=min_kpt_conf の割合
    - Ck: 可視kp(conf>=min_kpt_conf) の平均conf（無ければ0）
    """

    def __init__(
        self,
        require_visual_conf: bool = True,
        visual_conf_thresh: float = 0.60,  # ★積なので 0.90 は厳しめ。まず 0.4〜0.7 推奨
        unknown_policy_pass: bool = True,
        min_kpt_conf: float = 0.15,
        use_iou_term: bool = True,
        self_iou_skip_thr: float = 0.999,  # ★自分自身除外用（完全一致ではなく IoU で除外）
        no_kpt_policy: str = "fail",        # "fail" or "neutral"
        # "fail": keypointsが無い/全滅なら ratio=Ck=0 → Cr=0
        # "neutral": keypointsが無い/全滅なら ratio=1,Ck=1 として「kp項を無視」
    ):
        self.require = bool(require_visual_conf)
        self.th = float(visual_conf_thresh)
        self.unknown_pass = bool(unknown_policy_pass)
        self.min_kpt = float(min_kpt_conf)
        self.use_iou_term = bool(use_iou_term)
        self.self_iou_skip_thr = float(self_iou_skip_thr)
        self.no_kpt_policy = str(no_kpt_policy)

    def eval(self, obs: Observation, frame_shape) -> FilterScore:
        if not self.require:
            return FilterScore(True, 1.0, "visual filter disabled")

        # upstream override（不要なら削除OK）
        if obs.visual_conf is not None:
            vc = _clamp01(float(obs.visual_conf))
            return FilterScore(vc >= self.th, vc, f"precomputed vc={vc:.3f} th={self.th}")

        # bbox 必須
        if obs.bbox_xyxy is None:
            ok = self.unknown_pass
            return FilterScore(ok, -1.0, "no bbox -> unknown")

        bb4 = np.asarray(obs.bbox_xyxy, dtype=np.float32).reshape(-1)[:4]
        if bb4.size < 4:
            ok = self.unknown_pass
            return FilterScore(ok, -1.0, "bbox invalid -> unknown")

        # --- kpvisible/kpall と Ck ---
        ratio = 0.0
        Ck = 0.0

        if obs.keypoints_conf is not None:
            kc = np.asarray(obs.keypoints_conf, dtype=np.float32).reshape(-1)
            K = int(kc.size)
            if K > 0:
                vis = kc >= self.min_kpt
                kpvisible = int(vis.sum())
                ratio = float(kpvisible) / float(K)
                if kpvisible > 0:
                    Ck = float(kc[vis].mean())

        # keypointsが無い/全滅の扱い
        if (ratio == 0.0 or Ck == 0.0) and self.no_kpt_policy == "neutral":
            ratio, Ck = 1.0, 1.0

        # --- (1 - maxIoU) ---
        one_minus_max_iou = 1.0
        max_iou = 0.0

        if self.use_iou_term:
            # ★meta ではなく Observation の明示フィールドを使う想定
            all_bbs = getattr(obs, "all_bboxes_xyxy", None)

            if all_bbs is None:
                ok = self.unknown_pass
                return FilterScore(ok, -1.0, "no all_bboxes_xyxy -> unknown")

            all_bbs = np.asarray(all_bbs, dtype=np.float32).reshape(-1, 4)

            for b in all_bbs:
                iou = _iou_xyxy(bb4, b)
                # 自分自身（またはほぼ同一）を除外
                if iou >= self.self_iou_skip_thr:
                    continue
                if iou > max_iou:
                    max_iou = iou

            one_minus_max_iou = _clamp01(1.0 - float(max_iou))

        Cr = _clamp01(one_minus_max_iou * ratio * Ck)
        obs.visual_conf = Cr

        ok = (Cr >= self.th)
        reason = (
            f"Cr={Cr:.3f} (1-maxIoU)={one_minus_max_iou:.2f} "
            f"maxIoU={max_iou:.2f} ratio={ratio:.2f} Ck={Ck:.2f} th={self.th}"
        )
        return FilterScore(ok, Cr, reason)
