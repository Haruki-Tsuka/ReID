import argparse
from pathlib import Path
import re
import csv
import time
import bisect

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


def iter_depths(depth_dir: Path):
    npys = [p for p in depth_dir.iterdir() if p.suffix.lower() == ".npy"]
    npys.sort(key=natural_key)
    return npys


# --- IoU ---
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union


def _resize_for_preview(img, max_w: int):
    if max_w <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_tracks_id(img_bgr, tracks, tid_state):
    """描画するのは bbox と ID だけ。表示ID = global_id があれば global_id、なければ tracker_id"""
    vis = img_bgr.copy()
    for x1, y1, x2, y2, tid in tracks:
        tid = int(tid)
        st = tid_state.get(tid, {})
        gid = str(st.get("gid", "")).strip()
        show_id = gid if gid else str(tid)

        p1 = (int(round(float(x1))), int(round(float(y1))))
        p2 = (int(round(float(x2))), int(round(float(y2))))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)

        y = max(0, p1[1] - 7)
        cv2.putText(vis, show_id, (p1[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


# --- depth helpers ---
def load_depth_npy(depth_path: Path) -> np.ndarray:
    """(H,W) depth map. dtype/units are dataset-dependent."""
    d = np.load(str(depth_path))
    if d.ndim != 2:
        # たまに (H,W,1) などがある場合
        d = np.squeeze(d)
    return d


def get_depth_at_uv(depth: np.ndarray, u: float, v: float, area: int = 5, limit: int = 10) -> float:
    """
    (u,v) の近傍 area×area から 0を除外して中央値を返す。
    取得できなければ -1 を返す。
    """
    H, W = depth.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))

    if x < limit or x >= W - limit or y < limit or y >= H - limit:
        return -1.0

    x_min = max(0, x - area)
    x_max = min(W, x + area + 1)
    y_min = max(0, y - area)
    y_max = min(H, y + area + 1)

    patch = depth[y_min:y_max, x_min:x_max]
    patch = np.asarray(patch)
    nonzero = patch[patch != 0]

    if nonzero.size == 0:
        return -1.0

    return float(np.median(nonzero))


def depth_to_meters(d: float, unit: str) -> float:
    """
    depthの単位を meters に揃える。
    unit: "m" | "mm" | "raw"（rawはそのまま）
    """
    if d < 0:
        return -1.0
    unit = unit.lower().strip()
    if unit == "m":
        return float(d)
    if unit == "mm":
        return float(d) * 1e-3
    # "raw" はそのまま返す（ユーザ側で解釈）
    return float(d)


# --- TrackRef wrapper ---
class OCSortTrackRef:
    def __init__(self, tid: int, state: dict):
        self._tid = int(tid)
        self._st = state

    def tid(self) -> int:
        return self._tid

    def get_gid(self) -> str:
        return str(self._st.get("gid", "")).strip()

    def get_gid_source(self) -> str:
        return str(self._st.get("src", "")).strip()

    def get_gid_conf(self) -> float:
        return float(self._st.get("conf", 0.0))

    def set_gid(self, gid: str, conf: float, src: str) -> None:
        self._st["gid"] = str(gid)
        self._st["conf"] = float(conf)
        self._st["src"] = str(src)

    def clear_gid(self, src: str) -> None:
        self.set_gid("", 0.0, src)


def main():
    # ---- ReID init ----
    from ReID.reid import ReIDManager, ReIDConfig
    from ReID.types import Observation

    # ★距離フィルタを使いたいなら True にする
    cfg = ReIDConfig(
    )
    reid = ReIDManager(cfg)
    tid_state = {}  # tid -> {"gid":..., "conf":..., "src":...}

    # ---- args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="RGB画像フォルダ（ファイル名stemがtimestamp）")
    ap.add_argument("--depth_dir", required=True, help="深度npyフォルダ（画像と同stemの.np yがある想定）")
    ap.add_argument("--depth_unit", default="mm", choices=["m", "mm", "raw"], help="depth npyの単位")
    ap.add_argument("--depth_area", type=int, default=5, help="鼻周辺の深度中央値パッチ半径（5なら11x11）")
    ap.add_argument("--out_csv", default="output/ocsort_tracks.csv")
    ap.add_argument("--yolo_weights", default="yolo11m-pose.pt")
    ap.add_argument("--det_thresh", type=float, default=0.5)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fps", type=float, default=30.0, help="仮想カメラFPS（timestamp進行速度）")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview_stride", type=int, default=1)
    ap.add_argument("--preview_max_w", type=int, default=1280)
    ap.add_argument("--save_preview_dir", default="")
    ap.add_argument("--save_preview_stride", type=int, default=10)
    ap.add_argument("--quit_key", default="q")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    depth_dir = Path(args.depth_dir)

    imgs = iter_images(img_dir)
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")

    # ---- timestamp list (stem must be int) ----
    ts_list = []
    for p in imgs:
        try:
            ts_list.append(int(p.stem))
        except ValueError:
            raise RuntimeError(f"filename stem must be int timestamp: {p.name}")

    order = np.argsort(ts_list)
    imgs = [imgs[i] for i in order]
    ts_list = [ts_list[i] for i in order]

    # depth file mapping (stem -> path)
    depth_map = {}
    for dp in iter_depths(depth_dir):
        depth_map[dp.stem] = dp

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---- tracker & yolo ----
    from trackers.ocsort_tracker.ocsort import OCSort
    from ultralytics import YOLO

    model = YOLO(args.yolo_weights)
    tracker = OCSort(
        det_thresh=args.det_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_thr,
        asso_func="iou",
        use_byte=False,
    )

    # ---- virtual time ----
    current_ts = ts_list[0]
    frame_period_ms = 1000.0 / float(args.fps)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "local_id", "global_id", "x1", "y1", "x2", "y2"])

        frame_idx = 0
        last_i = -1
        same_i_count = 0

        while True:
            i = bisect.bisect_left(ts_list, int(current_ts))
            if i >= len(imgs):
                break

            # ---- 同じフレームが続いたら次へ ----
            if i == last_i:
                same_i_count += 1
                # 1回でも被ったら次へ送る（必要なら >=2 にしてもOK）
                i = last_i + 1
                if i >= len(imgs):
                    break
                # ★仮想時刻も「次フレームのtimestamp」に追従させる
                current_ts = float(ts_list[i])
            else:
                same_i_count = 0

            last_i = i


            img_path = imgs[i]
            timestamp = int(img_path.stem)

            img = cv2.imread(str(img_path))
            if img is None:
                current_ts += frame_period_ms
                continue

            # depth load (同stemのnpy)
            depth_path = depth_map.get(str(timestamp), None)
            depth = None
            if depth_path is not None and depth_path.exists():
                try:
                    depth = load_depth_npy(depth_path)
                except Exception:
                    depth = None

            H, W = img.shape[:2]
            t0 = time.time()

            dets = np.empty((0, 5), dtype=np.float32)
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            kp_xy = None
            kp_conf = None

            # ---- YOLO ----
            if args.device is None:
                results = model.predict(source=img, verbose=False)
            else:
                results = model.predict(source=img, verbose=False, device=args.device)

            if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)
                if xyxy.shape[0] > 0:
                    dets = np.concatenate([xyxy, conf.reshape(-1, 1)], axis=1)

                if getattr(results[0], "keypoints", None) is not None and results[0].keypoints.xy is not None:
                    kp_xy = results[0].keypoints.xy.detach().cpu().numpy().astype(np.float32)  # (N,K,2)
                    if getattr(results[0].keypoints, "conf", None) is not None:
                        kp_conf = results[0].keypoints.conf.detach().cpu().numpy().astype(np.float32)  # (N,K)

            # ---- OC-SORT update ----
            tracks = tracker.update(dets, (H, W), (H, W))
            tracks_bbs = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in tracks], dtype=np.float32)

            track_refs = []
            obs_map = {}

            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)

                st = tid_state.setdefault(tid, {"gid": "", "conf": 0.0, "src": ""})
                tr = OCSortTrackRef(tid, st)
                track_refs.append(tr)

                tb = np.array([x1, y1, x2, y2], dtype=np.float32)

                # associate to best det by IoU (for kp + depth sampling)
                best_j = -1
                best_iou = 0.0
                for j in range(xyxy.shape[0]):
                    v = iou_xyxy(tb, xyxy[j])
                    if v > best_iou:
                        best_iou = v
                        best_j = j

                obs = Observation(
                    tid=tid,
                    timestamp=float(timestamp),
                    bbox_xyxy=tb,
                    all_bboxes_xyxy=tracks_bbs,  # visual_conf用（同一フレームbbox群）
                    keypoints_xy=(kp_xy[best_j] if (best_j >= 0 and kp_xy is not None) else None),
                    keypoints_conf=(kp_conf[best_j] if (best_j >= 0 and kp_conf is not None) else None),
                    visual_conf=None,            # Cr式で計算する
                    front_conf=None,

                    # ★距離フィルタ用：深度を渡す
                    depth=depth,          # (H,W) np.ndarray
                    has_dist=False,
                    cam_dist_m=-1.0,
                )
                obs_map[tid] = obs

            # ---- call ReID ----
            reid.update(track_refs, obs_map, frame_bgr=img, timestamp=float(timestamp), all_tracks=track_refs)

            # ---- CSV output ----
            used = set()
            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)
                if tid in used:
                    continue
                used.add(tid)

                st = tid_state.get(tid, {})
                gid = st.get("gid", "")
                w.writerow([
                    timestamp, tid, gid,
                    int(round(float(x1))), int(round(float(y1))),
                    int(round(float(x2))), int(round(float(y2)))
                ])

            f.flush()

            # ---- advance virtual time by processing time ----
            proc_dt = time.time() - t0
            current_ts += proc_dt * 1000.0

            # ---- preview ----
            frame_idx += 1
            do_show = args.preview and (frame_idx % max(1, args.preview_stride) == 0)
            do_save = (args.save_preview_dir != "") and (frame_idx % max(1, args.save_preview_stride) == 0)

            if do_show or do_save:
                vis = draw_tracks_id(img, tracks, tid_state)

                if do_show:
                    show_img = _resize_for_preview(vis, args.preview_max_w)
                    cv2.imshow("OCSORT+ReID Preview", show_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(args.quit_key):
                        print("[Preview] Quit key pressed. Exiting.")
                        break

                if do_save:
                    out_dir = Path(args.save_preview_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{timestamp}.jpg"
                    cv2.imwrite(str(out_path), vis)

    if args.preview:
        cv2.destroyAllWindows()

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
