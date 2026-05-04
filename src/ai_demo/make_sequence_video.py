#!/usr/bin/env python3
"""Build Input/Inference/GT frame sequences and optionally render a video with ffmpeg."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2

SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def scene_index(path: Path) -> int:
    match = SCENE_IMAGE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected scene filename: {path.name}")
    return int(match.group(1))


def list_split_images(dataset_root: Path, split: str) -> List[Path]:
    split_dir = dataset_root / "images" / split
    images = [p for p in split_dir.glob("scene_*.png") if p.is_file() and SCENE_IMAGE_RE.fullmatch(p.name)]
    images.sort(key=scene_index)
    return images


def resolve_default_detection_weights(root: Path, dataset_root: Path) -> Path:
    default_root_checkpoint = root / "yolov8s.pt"
    if default_root_checkpoint.exists():
        return default_root_checkpoint.resolve()

    trained_models_path = dataset_root / "trained_models.json"
    if trained_models_path.exists():
        payload = json.loads(trained_models_path.read_text(encoding="utf-8"))
        det_rel = payload.get("detection")
        if det_rel:
            p = Path(det_rel)
            if not p.is_absolute():
                p = root / p
            if p.exists():
                return p.resolve()

    fallback = root / "yolov8n.pt"
    if not fallback.exists():
        raise FileNotFoundError(
            "Could not resolve detection weights. Pass --weights explicitly."
        )
    return fallback.resolve()


def resolve_weights(weights_arg: str | None, root: Path, dataset_root: Path) -> Path:
    if not weights_arg:
        return resolve_default_detection_weights(root, dataset_root)
    p = Path(weights_arg).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p


def parse_gt_boxes(label_path: Path, w: int, h: int) -> List[tuple[int, int, int, int]]:
    boxes: List[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            continue

        cx, cy, bw, bh = (float(v) for v in toks[1:5])
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def label_path_for_image(dataset_root: Path, split: str, image_stem: str) -> Path:
    candidates = [
        dataset_root / "labels" / "yolo" / split / f"{image_stem}.txt",
        dataset_root / "labels" / split / f"{image_stem}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def draw_gt_overlay(image_bgr, boxes: List[tuple[int, int, int, int]]):
    overlay = image_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


def add_panel_title(image_bgr, title: str):
    bar_h = 38
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def compose_triptych(input_bgr, pred_bgr, gt_bgr):
    h, w = input_bgr.shape[:2]
    pred = cv2.resize(pred_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_AREA)

    p1 = add_panel_title(input_bgr, "Input")
    p2 = add_panel_title(pred, "Inference")
    p3 = add_panel_title(gt, "GT (BBox)")
    return cv2.hconcat([p1, p2, p3])


def configure_ultralytics(root: Path) -> None:
    config_dir = root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def draw_prediction_overlay(image_bgr, boxes, scores, title: str):
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        overlay,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"car {float(score):.2f}",
            (x1, max(48, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


class DetectorAdapter:
    def predict_overlay(self, image_bgr):
        raise NotImplementedError


class UltralyticsAdapter(DetectorAdapter):
    def __init__(self, root: Path, weights_path: Path, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed in this environment.") from exc

        configure_ultralytics(root)
        self.model = YOLO(str(weights_path))
        self.device = device
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict):
            self.names = {int(k): str(v).lower() for k, v in raw_names.items()}
        else:
            self.names = {idx: str(v).lower() for idx, v in enumerate(raw_names)}

    def predict_overlay(self, image_bgr):
        result = self.model(image_bgr, device=self.device, verbose=False)[0]
        boxes = []
        scores = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, cls_id in zip(xyxy, conf, cls_ids):
                if self.names.get(int(cls_id), "") != "car":
                    continue
                boxes.append(tuple(int(round(v)) for v in box.tolist()))
                scores.append(float(score))
        return draw_prediction_overlay(image_bgr, boxes, scores, "Inference")


class FasterRCNNAdapter(DetectorAdapter):
    def __init__(self, device: str):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.torch = torch
        device_str = "cpu" if device == "cpu" else (f"cuda:{device}" if device.isdigit() else device)
        self.device = torch.device(device_str)
        self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()
        self.car_label_id = 3

    def predict_overlay(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            outputs = self.model([tensor])[0]

        labels = outputs["labels"].detach().cpu().numpy()
        boxes_xyxy = outputs["boxes"].detach().cpu().numpy()
        scores_conf = outputs["scores"].detach().cpu().numpy()

        boxes = []
        scores = []
        for label, box, score in zip(labels, boxes_xyxy, scores_conf):
            if int(label) != self.car_label_id:
                continue
            boxes.append(tuple(int(round(v)) for v in box.tolist()))
            scores.append(float(score))
        return draw_prediction_overlay(image_bgr, boxes, scores, "Inference")


def load_detector(root: Path, weights_arg: str | None, device: str, dataset_root: Path) -> tuple[DetectorAdapter, str]:
    if weights_arg == "fasterrcnn_resnet50_fpn_v2":
        return FasterRCNNAdapter(device=device), weights_arg

    weights_path = resolve_weights(weights_arg, root, dataset_root)
    return UltralyticsAdapter(root=root, weights_path=weights_path, device=device), str(weights_path)


def ffmpeg_command(frames_dir: Path, fps: int, out_path: Path) -> str:
    return (
        "ffmpeg -y "
        f"-framerate {fps} "
        f"-i {shlex.quote(str(frames_dir / 'frame_%06d.png'))} "
        "-c:v libx264 -pix_fmt yuv420p "
        f"{shlex.quote(str(out_path))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Input/Inference/GT sequence and video")
    parser.add_argument("--dataset", type=str, default="./test_visiondrive3d")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Detection model weights path or 'fasterrcnn_resnet50_fpn_v2' for torchvision pretrained inference",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <dataset>/ai_results/sequence_frames")
    parser.add_argument("--video-output", type=str, default=None, help="Defaults to <dataset>/ai_results/inference_input_gt.mp4")
    parser.add_argument("--run-ffmpeg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)

    images = list_split_images(dataset_root, args.split)
    if not images:
        raise RuntimeError(f"No images found in {dataset_root / 'images' / args.split}")

    start = max(0, int(args.start_index))
    if start >= len(images):
        raise ValueError(f"start-index {start} is out of range for {len(images)} images")

    end = min(len(images), start + max(1, int(args.max_frames)))
    selected = images[start:end]

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (dataset_root / "ai_results" / "sequence_frames")
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out = Path(args.video_output).expanduser() if args.video_output else (dataset_root / "ai_results" / "inference_input_gt.mp4")
    if not video_out.is_absolute():
        video_out = (root / video_out).resolve()

    detector, weights_label = load_detector(root=root, weights_arg=args.weights, device=args.device, dataset_root=dataset_root)

    print(f"[INFO] Using weights/model: {weights_label}")
    print(f"[INFO] Frames: {len(selected)} ({selected[0].name} .. {selected[-1].name})")
    print(f"[INFO] Output dir: {out_dir}")

    for idx, img_path in enumerate(selected):
        input_bgr = cv2.imread(str(img_path))
        if input_bgr is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue

        pred_bgr = detector.predict_overlay(input_bgr)

        h, w = input_bgr.shape[:2]
        gt_label_path = label_path_for_image(dataset_root, args.split, img_path.stem)
        gt_boxes = parse_gt_boxes(gt_label_path, w=w, h=h)
        gt_bgr = draw_gt_overlay(input_bgr, gt_boxes)

        frame = compose_triptych(input_bgr, pred_bgr, gt_bgr)
        frame_path = out_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)

    cmd = ffmpeg_command(out_dir, fps=int(args.fps), out_path=video_out)
    print("\n[INFO] ffmpeg command:")
    print(cmd)

    if args.run_ffmpeg:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed. Run without --run-ffmpeg and execute the printed command where ffmpeg is available.")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[INFO] Video saved to: {video_out}")


if __name__ == "__main__":
    main()
