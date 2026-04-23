# types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Dict
import numpy as np

@dataclass
class Observation:
    tid: int
    timestamp: float
    bbox_xyxy: Optional[np.ndarray] = None   # (4,) float/int ok

    # ★追加：同一フレーム内の bbox 一覧（検出 or トラック、どちらでも）
    all_bboxes_xyxy: Optional[np.ndarray] = None  # (N,4)

    # Optional pose (e.g., COCO-17)
    keypoints_xy: Optional[np.ndarray] = None      # (K,2)
    keypoints_conf: Optional[np.ndarray] = None    # (K,)

    # Optional distance inputs
    head_xyz: Optional[np.ndarray] = None  # (3,) head position (camera/world)
    cam_xyz: Optional[np.ndarray] = None   # (3,) camera position (same frame)
    cam_dist_m: float = -1.0
    has_dist: bool = False

    # types.py の Observation に追加
    depth: Optional[np.ndarray] = None      # (H,W) depth map aligned to RGB
    depth_path: Optional[str] = None        # optional: path to depth file

    # Optional precomputed confidences (if upstream computed them)
    front_conf: Optional[float] = None
    visual_conf: Optional[float] = None

    # free-form
    meta: Optional[Dict[str, Any]] = None


@dataclass
class FilterScore:
    ok: bool
    score: float          # 0..1 (or -1 for unknown)
    reason: str = ""


class TrackRef(Protocol):
    """Tracker-independent interface to read/write global identity."""
    def tid(self) -> int: ...
    def get_gid(self) -> str: ...
    def get_gid_source(self) -> str: ...
    def get_gid_conf(self) -> float: ...

    def set_gid(self, gid: str, conf: float, src: str) -> None: ...
    def clear_gid(self, src: str) -> None: ...


class SimpleTrackRef:
    """Adapter around any tracker track object."""
    def __init__(self, track: Any):
        self.t = track

    def tid(self) -> int:
        if hasattr(self.t, "get_local_id"):
            try: return int(self.t.get_local_id())
            except: pass
        for name in ("local_id", "track_id", "tracker_id", "id", "tid"):
            if hasattr(self.t, name):
                try: return int(getattr(self.t, name))
                except: pass
        return -1

    def get_gid(self) -> str:
        return str(getattr(self.t, "global_id", getattr(self.t, "person_id", ""))).strip()

    def get_gid_source(self) -> str:
        return str(getattr(self.t, "person_id_source", "")).strip()

    def get_gid_conf(self) -> float:
        return float(getattr(self.t, "person_id_conf", 0.0))

    def set_gid(self, gid: str, conf: float, src: str) -> None:
        setattr(self.t, "global_id", gid)
        setattr(self.t, "person_id", gid)
        setattr(self.t, "person_id_conf", float(conf))
        setattr(self.t, "person_id_source", str(src))

    def clear_gid(self, src: str) -> None:
        self.set_gid("", 0.0, src)
