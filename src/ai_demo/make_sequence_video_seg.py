#!/usr/bin/env python3
"""Build Input/Inference/GT(segmentation) frame sequences and optionally render a video with ffmpeg."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np

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


def resolve_default_segmentation_weights(root: Path, dataset_root: Path) -> Path:
    trained_models_path = dataset_root / "trained_models.json"
    if trained_models_path.exists():
        payload = json.loads(trained_models_path.read_text(encoding="utf-8"))
        seg_rel = payload.get("segmentation")
        if seg_rel:
            p = Path(seg_rel)
            if not p.is_absolute():
                p = root / p
            if p.exists():
                return p.resolve()

    fallback = root / "yolov8n-seg.pt"
    if not fallback.exists():
        raise FileNotFoundError(
            "Could not resolve segmentation weights. Pass --weights explicitly."
        )
    return fallback.resolve()


def resolve_weights(weights_arg: str | None, root: Path, dataset_root: Path) -> Path:
    if not weights_arg:
        return resolve_default_segmentation_weights(root, dataset_root)
    p = Path(weights_arg).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Weights not found: {p}")
    return p


def label_path_for_image(dataset_root: Path, split: str, image_stem: str) -> Path:
    return dataset_root / "labels_seg" / split / f"{image_stem}.txt"


def parse_gt_polygons(label_path: Path, w: int, h: int) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    if not label_path.exists():
        return polygons

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        toks = line.split()
        if len(toks) < 7:
            continue

        coord_vals = toks[1:]
        if len(coord_vals) % 2 != 0:
            coord_vals = coord_vals[:-1]
        if len(coord_vals) < 6:
            continue

        coords = np.array([float(v) for v in coord_vals], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0] * float(w), 0, float(w - 1))
        coords[:, 1] = np.clip(coords[:, 1] * float(h), 0, float(h - 1))
        poly = np.round(coords).astype(np.int32)
        if poly.shape[0] >= 3:
            polygons.append(poly)

    return polygons


def draw_gt_overlay(image_bgr: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    overlay = image_bgr.copy()
    if not polygons:
        return overlay

    fill = overlay.copy()
    cv2.fillPoly(fill, polygons, color=(0, 255, 0))
    overlay = cv2.addWeighted(fill, 0.30, overlay, 0.70, 0.0)
    cv2.polylines(overlay, polygons, isClosed=True, color=(0, 255, 0), thickness=2)
    return overlay


def add_panel_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
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


def compose_triptych(input_bgr: np.ndarray, pred_bgr: np.ndarray, gt_bgr: np.ndarray) -> np.ndarray:
    h, w = input_bgr.shape[:2]
    pred = cv2.resize(pred_bgr, (w, h), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_AREA)

    p1 = add_panel_title(input_bgr, "Input")
    p2 = add_panel_title(pred, "Inference")
    p3 = add_panel_title(gt, "GT (Mask)")
    return cv2.hconcat([p1, p2, p3])


def ffmpeg_command(frames_dir: Path, fps: int, out_path: Path) -> str:
    return (
        "ffmpeg -y "
        f"-framerate {fps} "
        f"-i {shlex.quote(str(frames_dir / 'frame_%06d.png'))} "
        "-c:v libx264 -pix_fmt yuv420p "
        f"{shlex.quote(str(out_path))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create segmentation Input/Inference/GT sequence and video")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--weights", type=str, default=None, help="Segmentation model weights path")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <dataset>/ai_results/sequence_frames_seg")
    parser.add_argument("--video-output", type=str, default=None, help="Defaults to <dataset>/ai_results/inference_input_gt_seg.mp4")
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

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else (dataset_root / "ai_results" / "sequence_frames_seg_triptych")
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_out = Path(args.video_output).expanduser() if args.video_output else (dataset_root / "ai_results" / "inference_input_gt_seg.mp4")
    if not video_out.is_absolute():
        video_out = (root / video_out).resolve()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed in this environment.") from exc

    weights_path = resolve_weights(args.weights, root, dataset_root)
    model = YOLO(str(weights_path))

    print(f"[INFO] Using weights: {weights_path}")
    print(f"[INFO] Frames: {len(selected)} ({selected[0].name} .. {selected[-1].name})")
    print(f"[INFO] Output dir: {out_dir}")

    for idx, img_path in enumerate(selected):
        input_bgr = cv2.imread(str(img_path))
        if input_bgr is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue

        result = model(str(img_path), device=args.device, verbose=False)[0]
        pred_bgr = result.plot()

        h, w = input_bgr.shape[:2]
        gt_label_path = label_path_for_image(dataset_root, args.split, img_path.stem)
        gt_polygons = parse_gt_polygons(gt_label_path, w=w, h=h)
        gt_bgr = draw_gt_overlay(input_bgr, gt_polygons)

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
