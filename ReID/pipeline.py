# pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from .types import Observation, FilterScore
from .filters.bbox_conf import BBoxConfFilter
from .filters.visual_conf import VisualConfFilter
from .filters.front_conf import FrontConfFilter
from .filters.distance_conf import DistanceConfFilter, DistanceFilterConfig, DepthIOConfig  # ★追加


@dataclass
class PipelineResult:
    ok: bool
    scores: Dict[str, FilterScore]


class FilterPipeline:
    def __init__(self, cfg):
        self.bbox = BBoxConfFilter(
            use_bbox_filter=cfg.use_bbox_filter,
            ar_min=cfg.ar_min,
            ar_max=cfg.ar_max,
            min_h=cfg.min_h,
            min_area=cfg.min_area,
        )
        self.visual = VisualConfFilter(
            require_visual_conf=cfg.require_visual_conf,
            visual_conf_thresh=cfg.visual_conf_thresh,
        )
        self.front = FrontConfFilter(
            require_front=cfg.require_front,
            front_conf_thresh=cfg.front_conf_thresh,
        )

        self.use_distance = bool(getattr(cfg, "use_distance_filter", False))
        self.dist = None
        if self.use_distance:
            # ★m固定：depth_unit="m"
            io_cfg = DepthIOConfig(depth_unit="m")

            # ★cfg側に min/max がある前提（無ければデフォルト値に fallback）
            dist_cfg = DistanceFilterConfig(
                enabled=True,
                min_cam_dist_m=float(getattr(cfg, "min_cam_dist_m", 3.5)),
                max_cam_dist_m=float(getattr(cfg, "max_cam_dist_m", 7.0)),
                unknown_policy_pass=bool(getattr(cfg, "distance_unknown_policy_pass", True)),
                min_kpt_conf=float(getattr(cfg, "min_body_kpt_conf", 0.15)),
            )
            self.dist = DistanceConfFilter(dist_cfg, io_cfg=io_cfg)

    def run(self, obs: Observation, frame_shape) -> PipelineResult:
        scores: Dict[str, FilterScore] = {}

        r_bbox = self.bbox.eval(obs, frame_shape)
        scores["bbox"] = r_bbox
        if not r_bbox.ok:
            return PipelineResult(False, scores)

        r_vis = self.visual.eval(obs, frame_shape)
        scores["visual"] = r_vis
        if not r_vis.ok:
            return PipelineResult(False, scores)

        r_front = self.front.eval(obs)
        scores["front"] = r_front
        if not r_front.ok:
            return PipelineResult(False, scores)

        if self.use_distance and self.dist is not None:
            r_dist = self.dist.eval(obs, frame_shape)
            scores["distance"] = r_dist
            if not r_dist.ok:
                return PipelineResult(False, scores)

        return PipelineResult(True, scores)
