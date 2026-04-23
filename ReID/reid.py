# reid.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Any, List

import numpy as np
import cv2

try:
    import torch
    from torchreid.utils import FeatureExtractor
except Exception:
    torch = None
    FeatureExtractor = None

from .types import Observation, TrackRef
from .pipeline import FilterPipeline

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _THIS_DIR / "weights" / "osnet_x1_0_market.pth"

@dataclass
class ReIDConfig:
    #4種のフィルター関連
    enabled: bool = True
    
    require_visual_conf: bool = False
    visual_conf_thresh: float = 0.90

    require_front: bool = True
    front_conf_thresh: float = 0.90

    use_distance_filter: bool = True
    min_cam_dist_m: float = 3.5
    max_cam_dist_m: float = 7.0

    use_bbox_filter: bool = False
    ar_min: float = 0.35
    ar_max: float = 0.65
    min_h: int = 100
    min_area: int = 80 * 80

    #類似度閾値
    ok_thresh: float = 0.80
    reassign_thresh: float = 0.80
    new_person_thresh: float = 0.80  # keep as you want

    #ReIDモデル
    model_name: str = "osnet_x1_0"
    model_path: Optional[str] = str(_DEFAULT_WEIGHTS)   #market1501で学習済み
    input_size: Tuple[int, int] = (256, 128)  # (H, W)

    #比較データベース関連
    gallery_dir: str = "reid_gallery"
    save_cooldown_sec: float = 2.0
    max_save_per_id: int = 2000

    # face lock
    allow_overwrite_face: bool = False

    # debug
    verbose: bool = True
    debug_call: bool = False
    debug_filters: bool = False
    debug_gallery: bool = False
    gallery_topk: int = 5
    debug_save: bool = False


class ReIDManager:
    def __init__(self, cfg: Optional[ReIDConfig] = None):
        self.cfg = cfg or ReIDConfig()
        self.pipeline = FilterPipeline(self.cfg)

        self.device = "cpu"
        self.extractor: Optional[Any] = None

        self.gallery_dir = Path(self.cfg.gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.gallery: Dict[str, np.ndarray] = {}  # label -> (N,D) L2-normalized

        self._reserved_labels: Set[str] = set()
        self._next_id_num: int = 1
        self._last_save_time_map: Dict[str, float] = {}

        if (not self.cfg.enabled) or FeatureExtractor is None or torch is None:
            self.cfg.enabled = False
            return

        self._init_model()
        self._load_gallery()
        self._init_id_allocator()

    # ---------------- public ----------------
    def update(
        self,
        tracks: Iterable[TrackRef],
        observations: Dict[int, Observation],   # tid -> Observation
        frame_bgr: np.ndarray,
        timestamp: float,
        all_tracks: Optional[Iterable[TrackRef]] = None,
    ) -> None:
        if not self._ready(frame_bgr):
            return

        update_tracks: List[TrackRef] = list(tracks)
        all_tracks_list: List[TrackRef] = list(all_tracks) if all_tracks is not None else update_tracks

        if self.cfg.debug_call:
            self._v(f"[ReID] ts={timestamp:.3f} n_update={len(update_tracks)} n_all={len(all_tracks_list)} gallery={len(self.gallery)}")

        # 0) old state
        old_gid: Dict[int, str] = {}
        old_src: Dict[int, str] = {}
        for tr in all_tracks_list:
            tid = int(tr.tid())
            old_gid[tid] = str(tr.get_gid()).strip()
            old_src[tid] = str(tr.get_gid_source()).strip()

        # 1) feature extraction (only passed tracks)
        feat_map: Dict[int, np.ndarray] = {}
        for tr in update_tracks:
            tid = int(tr.tid())
            obs = observations.get(tid)
            if obs is None:
                continue

            pres = self.pipeline.run(obs, frame_shape=frame_bgr.shape)
            if not pres.ok:
                continue

            feat = self._extract_feature_from_bbox(frame_bgr, obs.bbox_xyxy)
            if feat is None:
                continue
            feat_map[tid] = feat

        # 2) match & assign (update_tracks only)
        used_labels: Set[str] = set()

        def set_gid(tr: TrackRef, gid: str, conf: float, src: str):
            tr.set_gid(gid, float(conf), src)

        for tr in update_tracks:
            tid = int(tr.tid())
            if tid not in feat_map:
                continue

            feat = feat_map[tid]
            cur = str(tr.get_gid()).strip()
            cur_src = str(tr.get_gid_source()).strip()

            if cur == "":
                best_label, best_sim = self._match_gallery(feat, forbid=used_labels, tid=tid, cur_label="")

                # (1) 既存に十分似ている → 既存IDを付与
                if best_label is not None and best_sim >= float(self.cfg.reassign_thresh):
                    set_gid(tr, best_label, best_sim, "reid_take_from_blank")
                    used_labels.add(best_label)
                    self._maybe_save(best_label, feat, timestamp, reason=f"take_from_blank tid={tid} sim={best_sim:.3f}")
                    continue

                # (2) 既存に全然似ていない → 新規IDを発行
                if best_sim < float(self.cfg.new_person_thresh):
                    new_label = self._new_identity_label()
                    set_gid(tr, new_label, 0.0, "reid_new_from_blank")
                    used_labels.add(new_label)
                    self._maybe_save(new_label, feat, timestamp, reason=f"new_from_blank tid={tid} best={best_label}:{best_sim:.3f}")
                    continue

                # (3) 中間（new_person_thresh <= best_sim < reassign_thresh）→ NEWしないで保留（blank維持）
                # 何もしない（gidは空のまま）
                continue


            # has id -> check own
            sim_own = self._gallery_sim(cur, feat)
            if sim_own >= float(self.cfg.ok_thresh):
                # keep and refresh conf
                set_gid(tr, cur, sim_own, cur_src if cur_src else "reid_keep")
                used_labels.add(cur)
                self._maybe_save(cur, feat, timestamp, reason=f"keep tid={tid} sim_own={sim_own:.3f}")
                continue

            # face lock
            if (not self.cfg.allow_overwrite_face) and (cur_src == "face"):
                used_labels.add(cur)
                # keep as-is
                set_gid(tr, cur, sim_own, "face")
                continue

            best_label, best_sim = self._match_gallery(feat, forbid=used_labels, tid=tid, cur_label=cur)

            if best_label is None or best_sim < float(self.cfg.reassign_thresh) or best_label == cur:
                used_labels.add(cur)
                # keep as-is (conf update)
                set_gid(tr, cur, sim_own, cur_src if cur_src else "reid_keep_low")
                continue

            # take
            set_gid(tr, best_label, best_sim, "reid_reassign_take")
            used_labels.add(best_label)
            self._maybe_save(best_label, feat, timestamp, reason=f"reassign_take tid={tid} sim={best_sim:.3f}")

        # 3) swap/clear victims (including update-outside tracks)
        changes: List[Tuple[int, str, str]] = []
        for tr in update_tracks:
            tid = int(tr.tid())
            old = old_gid.get(tid, "")
            new = str(tr.get_gid()).strip()
            if new and old != new:
                changes.append((tid, old, new))

        def is_label_free(label: str, exclude_tids: Set[int]) -> bool:
            if label == "":
                return True
            for tr2 in all_tracks_list:
                t2 = int(tr2.tid())
                if t2 in exclude_tids:
                    continue
                if str(tr2.get_gid()).strip() == label:
                    return False
            return True

        # helper tid->TrackRef lookup
        tid_to_track: Dict[int, TrackRef] = {int(tr.tid()): tr for tr in all_tracks_list}

        for winner_tid, oldA, newA in changes:
            victims: List[TrackRef] = []
            for tB, old_label in old_gid.items():
                if tB == winner_tid:
                    continue
                if old_label == newA:
                    vb = tid_to_track.get(tB)
                    if vb is not None:
                        victims.append(vb)

            for vb in victims:
                tidB = int(vb.tid())
                swap_to = oldA

                if swap_to != "" and (not is_label_free(swap_to, exclude_tids={winner_tid, tidB})):
                    swap_to = ""

                if swap_to == "":
                    vb.clear_gid("reid_swap_clear")
                else:
                    vb.set_gid(swap_to, 0.0, "reid_swap")

        # 4) final uniqueness over all_tracks_list
        label_map: Dict[str, List[TrackRef]] = {}
        for tr in all_tracks_list:
            lab = str(tr.get_gid()).strip()
            if not lab:
                continue
            label_map.setdefault(lab, []).append(tr)

        for lab, trs in label_map.items():
            if len(trs) <= 1:
                continue

            def score(tr: TrackRef):
                src = str(tr.get_gid_source()).strip()
                face = 1 if src == "face" else 0
                conf = float(tr.get_gid_conf())
                return (face, conf)

            winner = max(trs, key=score)
            win_tid = int(winner.tid())

            for tr in trs:
                tid = int(tr.tid())
                if tid == win_tid:
                    continue
                tr.clear_gid("reid_unique_clear")

    # ---------------- feature extraction ----------------
    def _extract_feature_from_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if bbox_xyxy is None:
            return None
        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox.size < 4:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        img_rgb = self._preprocess(crop)
        return self._extract_feat(img_rgb)

    def _ready(self, frame_bgr) -> bool:
        return bool(self.cfg.enabled and self.extractor is not None and frame_bgr is not None)

    def _v(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg)

    def _init_model(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs = dict(model_name=self.cfg.model_name, device=self.device)

        if self.cfg.model_path:
            p = Path(self.cfg.model_path)
            if not p.exists():
                raise FileNotFoundError(f"[ReID] model_path not found: {p}")
            kwargs["model_path"] = str(p)

        print("[ReID] FeatureExtractor kwargs:", kwargs)  # ★確認用
        self.extractor = FeatureExtractor(**kwargs)


    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        H, W = self.cfg.input_size
        resized = cv2.resize(crop_bgr, (W, H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _extract_feat(self, img_rgb: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = self.extractor([img_rgb])
        if isinstance(out, torch.Tensor):
            f = out[0].detach().cpu().numpy().astype(np.float32)
        else:
            f = np.asarray(out[0], dtype=np.float32)
        return self._l2_normalize(f)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    # ---------------- gallery persistence ----------------
    def _load_gallery(self) -> None:
        self.gallery = {}
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        for npy_path in self.gallery_dir.glob("*.npy"):
            label = npy_path.stem
            arr = np.load(npy_path)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = self._l2_normalize_rows(arr)
            self.gallery[label] = arr

    def _init_id_allocator(self) -> None:
        nums = []
        for label in self.gallery.keys():
            if label.startswith("ID"):
                try:
                    nums.append(int(label[2:]))
                except Exception:
                    pass
        self._next_id_num = (max(nums) + 1) if nums else 1

    @staticmethod
    def _l2_normalize_rows(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        n = np.linalg.norm(a, axis=1, keepdims=True) + eps
        return (a / n).astype(np.float32)

    def _new_identity_label(self) -> str:
        while True:
            label = f"ID{self._next_id_num:04d}"
            self._next_id_num += 1
            if label in self.gallery:
                continue
            if label in self._reserved_labels:
                continue
            self._reserved_labels.add(label)
            return label

    def _maybe_save(self, label: str, feat: np.ndarray, timestamp: float, reason: str = "") -> None:
        if not label:
            return

        if label in self.gallery and self.gallery[label].shape[0] >= int(self.cfg.max_save_per_id):
            return

        last = float(self._last_save_time_map.get(label, 0.0))
        if (float(timestamp) - last) < float(self.cfg.save_cooldown_sec):
            return

        self._save_feature_to_gallery(label, feat)
        self._last_save_time_map[label] = float(timestamp)

    def _save_feature_to_gallery(self, label: str, feat: np.ndarray) -> None:
        feat = np.asarray(feat, dtype=np.float32)
        feat = self._l2_normalize(feat)

        npy_path = self.gallery_dir / f"{label}.npy"
        if npy_path.exists():
            old = np.load(npy_path)
            old = np.asarray(old, dtype=np.float32)
            if old.ndim == 1:
                old = old[None, :]
            new = np.vstack([old, feat[None, :]])
        else:
            new = feat[None, :]

        np.save(npy_path, new)
        self.gallery[label] = self._l2_normalize_rows(new)

    # ---------------- gallery match ----------------
    def _match_gallery(self, feat: np.ndarray, forbid: Set[str], tid: int = -1, cur_label: str = "") -> Tuple[Optional[str], float]:
        if not self.gallery:
            return None, -1.0

        feat = self._l2_normalize(feat)

        best_label = None
        best_sim = -1.0

        top = []
        for label, feats in self.gallery.items():
            if label in forbid:
                continue
            sims = feats @ feat
            sim = float(np.max(sims)) if sims.size else -1.0
            if sim > best_sim:
                best_sim = sim
                best_label = label
            if self.cfg.debug_gallery and self.cfg.gallery_topk > 0:
                top.append((sim, label))

        if self.cfg.debug_gallery and self.cfg.gallery_topk > 0:
            top.sort(reverse=True, key=lambda x: x[0])
            topk = top[: int(self.cfg.gallery_topk)]
            topk_str = ", ".join([f"{lab}:{sim:.3f}" for sim, lab in topk])
            self._v(f"[ReID-GALLERY] tid={tid} cur={cur_label} best={best_label}:{best_sim:.3f} topk=[{topk_str}] forbid={len(forbid)}")

        return best_label, best_sim

    def _gallery_sim(self, label: str, feat: np.ndarray) -> float:
        if not label:
            return -1.0
        feats = self.gallery.get(label)
        if feats is None or feats.size == 0:
            return -1.0
        feat = self._l2_normalize(feat)
        return float(np.max(feats @ feat))