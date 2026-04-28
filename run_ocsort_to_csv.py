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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_csv", default="output/ocsort_tracks.csv")
    ap.add_argument("--yolo_weights", default="yolo11m-pose.pt")
    ap.add_argument("--det_thresh", type=float, default=0.5)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--max_age", type=int, default=30)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fps", type=float, default=30.0, help="仮想カメラFPS（timestamp進行速度）")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    imgs = iter_images(img_dir)
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")

    # ---- timestamp(int)リストを作る（ms想定）----
    ts_list = []
    for p in imgs:
        try:
            ts_list.append(int(p.stem))
        except ValueError:
            raise RuntimeError(f"filename stem must be int timestamp: {p.name}")
    # 念のためソート
    order = np.argsort(ts_list)
    imgs = [imgs[i] for i in order]
    ts_list = [ts_list[i] for i in order]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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

    # ==============================
    # 仮想カメラ時刻（ms）
    # 画像列の先頭timestampから開始
    # ==============================
    current_ts = ts_list[0]
    i = 0  # 画像インデックス
    frame_period_ms = 1000.0 / float(args.fps)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "id", "x1", "y1", "x2", "y2"])

        while True:
            # current_ts に最も近い（=次に到達した）画像へジャンプ
            # 「処理中に進んだ分」だけ current_ts が先に進むので自然に飛ぶ
            i = bisect.bisect_left(ts_list, int(current_ts))
            if i >= len(imgs):
                break

            img_path = imgs[i]
            timestamp = img_path.stem  # CSVはファイル名固定

            img = cv2.imread(str(img_path))
            if img is None:
                # 読めないなら次へ（timestamp進める）
                current_ts += frame_period_ms
                continue

            H, W = img.shape[:2]

            t0 = time.time()

            # ---- YOLO推論 ----
            if args.device is None:
                results = model.predict(source=img, verbose=False)
            else:
                results = model.predict(source=img, verbose=False, device=args.device)

            if results is None or len(results) == 0 or results[0].boxes is None or results[0].boxes.xyxy is None:
                dets = np.empty((0, 5), dtype=np.float32)
            else:
                xyxy = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)
                dets = np.empty((0, 5), dtype=np.float32) if xyxy.shape[0] == 0 else np.concatenate([xyxy, conf.reshape(-1, 1)], axis=1)

            # ---- OC-SORT update ----
            tracks = tracker.update(dets, (H, W), (H, W))

            # ---- CSV出力 ----
            used = set()
            for x1, y1, x2, y2, tid in tracks:
                tid = int(tid)
                if tid in used:
                    continue
                used.add(tid)

                w.writerow([timestamp, tid,
                            int(round(float(x1))), int(round(float(y1))),
                            int(round(float(x2))), int(round(float(y2)))])

            f.flush()

            # ==============================
            # ★ここが肝：処理時間分だけ仮想時刻を進める
            # ==============================
            proc_dt = time.time() - t0  # 秒
            current_ts += proc_dt * 1000.0

            # 「カメラが一定FPSで時刻が進む」感を強めたいなら
            # 最低でも 1フレーム分は進める（処理が速すぎても同じフレームに張り付かない）

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
