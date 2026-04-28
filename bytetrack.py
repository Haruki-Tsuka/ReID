import argparse
from pathlib import Path
import re

import cv2
from ultralytics import YOLO


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
    ap.add_argument("--img_dir", required=True, help="画像フォルダ")
    ap.add_argument("--yolo_weights", default="yolo11m-pose.pt", help="detect/pose/seg いずれもOK")
    ap.add_argument("--device", default=None, help="例: 0, 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25, help="検出conf閾値")
    ap.add_argument("--imgsz", type=int, default=640, help="推論解像度")
    ap.add_argument("--quit_key", default="q")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    imgs = iter_images(img_dir)
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")

    model = YOLO(args.yolo_weights)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print("skip:", img_path)
            continue

        # ★ByteTrack に変更（tracker="bytetrack.yaml"）
        if args.device is None:
            r = model.track(
                img,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
        else:
            r = model.track(
                img,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

        vis = r.plot()

        if r.boxes is not None and r.boxes.is_track and r.boxes.id is not None:
            tids = r.boxes.id.int().cpu().tolist()
            print(img_path.name, "track_ids:", tids)

        cv2.imshow("ByteTrack Tracking", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(args.quit_key):
            print("quit")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
