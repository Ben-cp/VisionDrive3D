#!/usr/bin/env python3
"""Detection benchmark pipeline for VisionDrive3D."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCENE_IMAGE_RE = re.compile(r"scene_(\d+)\.png$")
IOU_THRESHOLDS = tuple(round(v, 2) for v in np.arange(0.5, 1.0, 0.05))
CAR_CLASS_NAMES = {"car"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    loader: str
    checkpoint: str | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(key="yolov8s", display_name="YOLOv8s", loader="ultralytics", checkpoint="yolov8s.pt"),
    ModelSpec(key="rtdetr_l", display_name="RT-DETR-L", loader="ultralytics", checkpoint="rtdetr-l.pt"),
    ModelSpec(key="fasterrcnn_resnet50_fpn_v2", display_name="Faster R-CNN ResNet50 FPN v2", loader="torchvision"),
)


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


def list_test_images(dataset_root: Path) -> List[Path]:
    test_dir = dataset_root / "images" / "test"
    images = [p for p in test_dir.glob("scene_*.png") if p.is_file() and SCENE_IMAGE_RE.fullmatch(p.name)]
    images.sort(key=scene_index)
    return images


def count_annotations(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    with label_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def select_top_annotated_images(dataset_root: Path, test_images: Sequence[Path], top_k: int = 5) -> List[Path]:
    labels_test_dir = dataset_root / "labels" / "yolo" / "test"
    scored: List[tuple[int, int, Path]] = []
    for img_path in test_images:
        ann_count = count_annotations(labels_test_dir / f"{img_path.stem}.txt")
        scored.append((ann_count, scene_index(img_path), img_path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def count_split(dataset_root: Path, split_name: str) -> int:
    split_dir = dataset_root / "images" / split_name
    return len(list(split_dir.glob("scene_*.png"))) if split_dir.exists() else 0


def resolve_runtime_device(requested_device: str) -> str:
    requested = str(requested_device).strip()
    if requested.lower() == "cpu":
        return "cpu"

    wants_cuda = requested.isdigit() or requested.lower().startswith("cuda")
    if not wants_cuda:
        return requested

    try:
        import torch
    except Exception:
        return requested

    if not torch.cuda.is_available():
        print(
            f"[WARN] Requested CUDA device '{requested}' but CUDA is unavailable. "
            "Falling back to CPU."
        )
        return "cpu"

    return requested


def configure_ultralytics(root: Path) -> None:
    config_dir = root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def resolve_checkpoint_path(root: Path, filename: str) -> Path:
    checkpoint = root / filename
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing required checkpoint at project root: {checkpoint}")
    return checkpoint.resolve()


def label_path_for_image(dataset_root: Path, image_stem: str) -> Path:
    candidates = [
        dataset_root / "labels" / "yolo" / "test" / f"{image_stem}.txt",
        dataset_root / "labels" / "test" / f"{image_stem}.txt",
        dataset_root / "labels" / "yolo" / f"{image_stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_gt_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) < 5:
            continue

        cls_id = int(float(toks[0]))
        if cls_id != 0:
            continue

        cx, cy, bw, bh = (float(v) for v in toks[1:5])
        x1 = max(0.0, min(float(w - 1), (cx - bw / 2.0) * float(w)))
        y1 = max(0.0, min(float(h - 1), (cy - bh / 2.0) * float(h)))
        x2 = max(0.0, min(float(w - 1), (cx + bw / 2.0) * float(w)))
        y2 = max(0.0, min(float(h - 1), (cy + bh / 2.0) * float(h)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_counts(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    keep = pred_scores >= float(score_threshold)
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    if pred_boxes.size == 0:
        return 0, 0, int(len(gt_boxes))

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    fp = 0

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if matched_gt[gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_idx] = True
            tp += 1
        else:
            fp += 1

    fn = int(len(gt_boxes) - matched_gt.sum())
    return tp, fp, fn


def compute_average_precision(
    pred_records: Sequence[dict[str, object]],
    gt_by_image: Dict[str, np.ndarray],
    iou_threshold: float,
) -> float:
    total_gt = int(sum(len(gt_boxes) for gt_boxes in gt_by_image.values()))
    if total_gt == 0:
        return 0.0

    sorted_preds = sorted(pred_records, key=lambda item: float(item["score"]), reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)
    matched: Dict[str, np.ndarray] = {
        image_id: np.zeros(len(gt_boxes), dtype=bool) for image_id, gt_boxes in gt_by_image.items()
    }

    for idx, pred in enumerate(sorted_preds):
        image_id = str(pred["image_id"])
        pred_box = np.asarray(pred["box"], dtype=np.float32)
        gt_boxes = gt_by_image[image_id]
        used = matched[image_id]

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if used[gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used[best_gt_idx] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(float(total_gt), 1e-8)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])

    changing_points = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[changing_points + 1] - recalls[changing_points]) * precisions[changing_points + 1]))
    return ap


def draw_boxes(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, title: str) -> np.ndarray:
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
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
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


class BaseDetectorAdapter:
    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class UltralyticsDetectorAdapter(BaseDetectorAdapter):
    def __init__(self, weights_path: Path, device: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install it in the active environment first.") from exc

        self.model = YOLO(str(weights_path))
        self.device = device
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict):
            self.names = {int(k): str(v).lower() for k, v in raw_names.items()}
        else:
            self.names = {idx: str(name).lower() for idx, name in enumerate(raw_names)}

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.model(image_bgr, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        keep = np.array([self.names.get(int(cls_id), "") in CAR_CLASS_NAMES for cls_id in classes], dtype=bool)
        if not keep.any():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return boxes[keep], scores[keep]


class TorchvisionFasterRCNNAdapter(BaseDetectorAdapter):
    def __init__(self, device: str):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.torch = torch
        device_str = "cpu" if device == "cpu" else (f"cuda:{device}" if device.isdigit() else device)
        self.device = torch.device(device_str)
        self.model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()
        self.car_label_id = 3  # COCO class id for car

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            outputs = self.model([tensor])[0]

        labels = outputs["labels"].detach().cpu().numpy()
        boxes = outputs["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = outputs["scores"].detach().cpu().numpy().astype(np.float32)
        keep = labels == self.car_label_id
        if not keep.any():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return boxes[keep], scores[keep]


def load_detector(spec: ModelSpec, root: Path, device: str) -> BaseDetectorAdapter:
    if spec.loader == "ultralytics":
        configure_ultralytics(root)
        if spec.checkpoint is None:
            raise RuntimeError(f"Checkpoint is required for {spec.display_name}")
        checkpoint_path = resolve_checkpoint_path(root, spec.checkpoint)
        return UltralyticsDetectorAdapter(checkpoint_path, device=device)
    if spec.loader == "torchvision":
        return TorchvisionFasterRCNNAdapter(device=device)
    raise ValueError(f"Unsupported loader '{spec.loader}'")


def evaluate_model(
    detector: BaseDetectorAdapter,
    spec: ModelSpec,
    *,
    dataset_root: Path,
    test_images: Sequence[Path],
    qualitative_images: Sequence[Path],
    qualitative_dir: Path,
) -> Dict[str, object]:
    gt_by_image: Dict[str, np.ndarray] = {}
    pred_records: list[dict[str, object]] = []
    precision_tp = 0
    precision_fp = 0
    precision_fn = 0
    inference_times_ms: list[float] = []

    for img_path in test_images:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        gt_boxes = parse_gt_boxes(label_path_for_image(dataset_root, img_path.stem), w=w, h=h)
        gt_by_image[img_path.stem] = gt_boxes

        t0 = time.perf_counter()
        pred_boxes, pred_scores = detector.predict(image_bgr)
        inference_times_ms.append((time.perf_counter() - t0) * 1000.0)

        for box, score in zip(pred_boxes, pred_scores):
            pred_records.append(
                {
                    "image_id": img_path.stem,
                    "box": box.tolist(),
                    "score": float(score),
                }
            )

        tp, fp, fn = match_counts(
            pred_boxes,
            pred_scores,
            gt_boxes,
            score_threshold=0.25,
            iou_threshold=0.5,
        )
        precision_tp += tp
        precision_fp += fp
        precision_fn += fn

    ap50 = compute_average_precision(pred_records, gt_by_image, iou_threshold=0.5)
    ap_values = [compute_average_precision(pred_records, gt_by_image, iou_threshold=t) for t in IOU_THRESHOLDS]
    precision = precision_tp / max(precision_tp + precision_fp, 1)
    recall = precision_tp / max(precision_tp + precision_fn, 1)

    model_qual_dir = qualitative_dir / spec.key
    model_qual_dir.mkdir(parents=True, exist_ok=True)
    for img_path in qualitative_images:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        pred_boxes, pred_scores = detector.predict(image_bgr)
        keep = pred_scores >= 0.25
        rendered = draw_boxes(
            image_bgr,
            pred_boxes[keep],
            pred_scores[keep],
            title=spec.display_name,
        )
        cv2.imwrite(str(model_qual_dir / img_path.name), rendered)

    return {
        "model": spec.display_name,
        "type": "Detection",
        "mAP50": float(ap50),
        "mAP50_95": float(np.mean(ap_values) if ap_values else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "inference_ms": float(np.mean(inference_times_ms) if inference_times_ms else 0.0),
    }


def write_csv(csv_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "mAP50", "mAP50_95", "precision", "recall", "inference_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_results(rows: Sequence[Dict[str, object]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    models = [str(row["model"]) for row in rows]
    map50 = [float(row["mAP50"]) for row in rows]
    speed = [float(row["inference_ms"]) for row in rows]

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_bar.bar(models, map50, color="#4A90D9")
    ax_bar.set_title("mAP@50 by Detector")
    ax_bar.set_ylabel("mAP@50")
    ax_bar.set_ylim(0.0, max(map50) * 1.15 if map50 else 1.0)
    ax_bar.grid(axis="y", alpha=0.3)

    ax_scatter.scatter(speed, map50, s=160, c="#4A90D9", alpha=0.8, edgecolors="black", linewidths=0.5)
    ax_scatter.set_title("Speed vs mAP@50")
    ax_scatter.set_xlabel("Inference time (ms)")
    ax_scatter.set_ylabel("mAP@50")
    ax_scatter.grid(alpha=0.3)
    for x, y, label in zip(speed, map50, models):
        ax_scatter.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def print_summary_table(rows: Sequence[Dict[str, object]]) -> None:
    print("Model                              | mAP@50 | mAP@50-95 | Speed(ms)")
    for row in rows:
        print(
            f"{str(row['model']):<34} | "
            f"{float(row['mAP50']):.3f}  | "
            f"{float(row['mAP50_95']):.3f}     | "
            f"{float(row['inference_ms']):.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate three pretrained 2D detectors on VisionDrive3D test split")
    parser.add_argument("--dataset", type=str, default="./test_visiondrive3d")
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)
    runtime_device = resolve_runtime_device(args.device)

    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_yaml}")

    test_images = list_test_images(dataset_root)
    if not test_images:
        raise RuntimeError(
            f"No test images found in {dataset_root / 'images' / 'test'}. "
            "Prepare the dataset split first."
        )

    qualitative_images = select_top_annotated_images(dataset_root, test_images, top_k=5)
    qualitative_dir = dataset_root / "ai_results" / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    for img_path in qualitative_images:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            cv2.imwrite(str(qualitative_dir / f"input_{img_path.name}"), image_bgr)

    resolution = "unknown"
    sample_image = cv2.imread(str(test_images[0]), cv2.IMREAD_COLOR)
    if sample_image is not None:
        resolution = f"{sample_image.shape[1]}x{sample_image.shape[0]}"

    result_rows: list[Dict[str, object]] = []
    for spec in MODEL_SPECS:
        print(f"[INFO] Evaluating {spec.display_name}")
        detector = load_detector(spec, root=root, device=runtime_device)
        result_rows.append(
            evaluate_model(
                detector,
                spec,
                dataset_root=dataset_root,
                test_images=test_images,
                qualitative_images=qualitative_images,
                qualitative_dir=qualitative_dir,
            )
        )

    results_payload = {
        "meta": {
            "resolution": resolution,
            "classes": ["car"],
            "splits": {
                "train": count_split(dataset_root, "train"),
                "val": count_split(dataset_root, "val"),
                "test": count_split(dataset_root, "test"),
            },
            "qualitative_scenes": [img.stem for img in qualitative_images],
        },
        "models": result_rows,
    }

    results_path = dataset_root / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    csv_path = dataset_root / "backbone_comparison.csv"
    write_csv(csv_path, result_rows)

    plot_path = dataset_root / "ai_results" / "backbone_comparison.png"
    plot_results(result_rows, plot_path)

    print_summary_table(result_rows)
    print(f"Saved results to: {results_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved qualitative predictions to: {qualitative_dir}")


if __name__ == "__main__":
    main()
