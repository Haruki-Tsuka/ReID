#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OC-SORT + ReID（あなたの提案手法）を「画像列→CSV」実験用に統合した最小プログラム。

出力CSV列（指定どおり）:
  timestamp, local_id, global_id, x1, y1, x2, y2
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import csv
import time
import bisect
from typing import Dict, Optional, List

import numpy as np
import cv2


def natural_key(p: Path):
    s = p.stem
    m = re.search(r"\d+", s)
    return int(m.group()) if m else s


def iter_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
    imgs.sort(key=natural_key)
    return imgs


# -------------------------
# ReID 用の最小互換データ
# -------------------------
@dataclass
class SimplePointData:
    bbox: np.ndarray              # (4,) xyxy
    visual_conf: float            # det confidence 代用
    front_conf: float = 1.0       # 実験用（姿勢推定があれば置換）
    coord: Optional[np.ndarray] = None  # 距離フィルタ用（ここでは未使用）

    def get_bbox(self):
        return self.bbox

    def get_coord(self):
        if self.coord is None:
            raise AttributeError("coord is None")
        return self.coord


class SimpleKalman:
    """ReIDManager が参照/更新する属性を持つだけのラッパ"""
    def __init__(self, local_id: int):
        self.local_id = int(local_id)
        self.person_id = ""              # ReID が付与する global label (e.g., ID0001)
        self.global_id = ""              # 互換のため同じ値を入れる
        self.person_id_conf = 0.0
        self.person_id_source = ""
        self.last_seen_idx = 0

    def get_local_id(self):
        return int(self.local_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_csv", default="output/ocsort_reid_tracks.csv")

    # detector / tracker
    ap.add_argument("--yolo_weights", default="yolo11m-pose.pt")
    ap.add_argument("--det_thresh", type=float, default=0.5)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--device", default=None)

    # virtual camera time
    ap.add_argument("--fps", type=float, default=30.0)

    # ReID
    ap.add_argument("--reid_gallery_dir", default="reid_gallery")
    ap.add_argument("--reid_ok", type=float, default=0.80)
    ap.add_argument("--reid_reassign", type=float, default=0.80)
    ap.add_argument("--reid_new", type=float, default=0.85)

    ap.add_argument("--reid_require_front", action="store_true")
    ap.add_argument("--reid_front_thresh", type=float, default=0.90)

    ap.add_argument("--reid_use_distance", action="store_true")
    ap.add_argument("--reid_min_dist", type=float, default=3.5)
    ap.add_argument("--reid_max_dist", type=float, default=7.0)

    ap.add_argument("--reid_use_bbox_filter", action="store_true")
    ap.add_argument("--reid_ar_min", type=float, default=0.35)
    ap.add_argument("--reid_ar_max", type=float, default=0.65)
    ap.add_argument("--reid_min_h", type=int, default=100)
    ap.add_argument("--reid_min_area", type=int, default=80 * 80)

    ap.add_argument("--reid_verbose", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    imgs = iter_images(img_dir)
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")

    # filename stem を timestamp(int) として扱う
    ts_list: List[int] = []
    for p in imgs:
        try:
            ts_list.append(int(p.stem))
        except ValueError:
            raise RuntimeError(f"filename stem must be int timestamp: {p.name}")
    order = np.argsort(ts_list)
    imgs = [imgs[i] for i in order]
    ts_list = [ts_list[i] for i in order]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    from trackers.ocsort_tracker.ocsort import OCSort
    from ultralytics import YOLO
    from reid import ReIDManager, ReIDConfig

    reid_cfg = ReIDConfig(
        enabled=True,
        require_visual_conf=True,
        visual_conf_thresh=float(args.det_thresh),

        require_front=bool(args.reid_require_front),
        front_conf_thresh=float(args.reid_front_thresh),

        use_distance_filter=bool(args.reid_use_distance),
        min_cam_dist_m=float(args.reid_min_dist),
        max_cam_dist_m=float(args.reid_max_dist),

        use_bbox_filter=bool(args.reid_use_bbox_filter),
        ar_min=float(args.reid_ar_min),
        ar_max=float(args.reid_ar_max),
        min_h=int(args.reid_min_h),
        min_area=int(args.reid_min_area),

        ok_thresh=float(args.reid_ok),
        reassign_thresh=float(args.reid_reassign),
        new_person_thresh=float(args.reid_new),

        gallery_dir=str(args.reid_gallery_dir),

        verbose=bool(args.reid_verbose),
        debug_call=bool(args.reid_verbose),
        debug_filters=bool(args.reid_verbose),
        debug_gallery=bool(args.reid_verbose),
        debug_save=bool(args.reid_verbose),
    )
    reid_mgr = ReIDManager(cfg=reid_cfg, node=None)

    model = YOLO(args.yolo_weights)
    tracker = OCSort(
        det_thresh=args.det_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_thr,
        asso_func="iou",
        use_byte=False,
    )

    current_ts = ts_list[0]
    frame_period_ms = 1000.0 / float(args.fps)

    # local_id -> kalman
    kalmans: Dict[int, SimpleKalman] = {}
    frame_idx = 0

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        # ★ 指定どおりこの7列だけ
        w.writerow(["timestamp", "local_id", "global_id", "x1", "y1", "x2", "y2"])

        while True:
            i = bisect.bisect_left(ts_list, int(current_ts))
            if i >= len(imgs):
                break

            img_path = imgs[i]
            timestamp_str = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                current_ts += frame_period_ms
                frame_idx += 1
                continue

            H, W = img.shape[:2]
            t0 = time.time()

            # YOLO推論
            if args.device is None:
                results = model.predict(source=img, verbose=False)
            else:
                results = model.predict(source=img, verbose=False, device=args.device)

            if results is None or len(results) == 0 or results[0].boxes is None or results[0].boxes.xyxy is None:
                dets = np.empty((0, 5), dtype=np.float32)
            else:
                xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)
                dets = np.empty((0, 5), dtype=np.float32) if xyxy.shape[0] == 0 else np.concatenate(
                    [xyxy, conf.reshape(-1, 1)], axis=1
                )

            # OC-SORT update
            tracks = tracker.update(dets, (H, W), (H, W))

            assosiate_dict = {}
            update_kalmans = []

            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)

                # kalman確保
                k = kalmans.get(tid)
                if k is None:
                    k = SimpleKalman(tid)
                    kalmans[tid] = k
                k.last_seen_idx = frame_idx
                update_kalmans.append(k)

                # det_conf 近似（bbox中心が近い detection の conf）
                det_conf = 1.0
                if dets.shape[0] > 0:
                    bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                    dxyxy = dets[:, :4]
                    centers = (dxyxy[:, :2] + dxyxy[:, 2:4]) * 0.5
                    c = (bb[:2] + bb[2:]) * 0.5
                    dist2 = np.sum((centers - c[None, :]) ** 2, axis=1)
                    j = int(np.argmin(dist2))
                    det_conf = float(dets[j, 4])

                pd = SimplePointData(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    visual_conf=float(det_conf),
                    front_conf=1.0,   # 実験用（姿勢推定を繋げるなら差し替え）
                    coord=None,       # 距離フィルタを使うなら差し替え
                )
                assosiate_dict[tid] = pd

            # ReID 実行（person_id を kalman 側に付与）
            reid_mgr.check_and_switch(
                update_kalmans=update_kalmans,
                assosiate_dict=assosiate_dict,
                frame_bgr=img,
                timestamp=float(int(timestamp_str)),
                all_kalmans=kalmans.values(),
            )

            # 古い kalman を掃除
            dead = []
            for tid, k in kalmans.items():
                if (frame_idx - k.last_seen_idx) > int(args.max_age):
                    dead.append(tid)
            for tid in dead:
                kalmans.pop(tid, None)

            # CSV出力（指定7列だけ）
            used = set()
            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)
                if tid in used:
                    continue
                used.add(tid)

                k = kalmans.get(tid)
                gid = ""
                if k is not None:
                    gid = str(getattr(k, "person_id", "")).strip()

                w.writerow([
                    timestamp_str, tid, gid,
                    int(round(float(x1))), int(round(float(y1))),
                    int(round(float(x2))), int(round(float(y2))),
                ])

            f.flush()

            # 仮想時刻を進める
            proc_dt = time.time() - t0
            current_ts += max(proc_dt * 1000.0, frame_period_ms)
            frame_idx += 1

    print("Saved:", out_csv)
    print("ReID gallery dir:", Path(args.reid_gallery_dir).resolve())


if __name__ == "__main__":
    main()
