#!/usr/bin/env python3
"""Master script to run training/evaluation/backbone experiments and page asset sync."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_root(dataset_arg: str, root: Path) -> Path:
    candidate = Path(dataset_arg).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def run_python_script(root: Path, script_rel: str, args: Iterable[str]) -> None:
    cmd = [sys.executable, str(root / script_rel), *args]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=root, check=True)


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_dataset_grid(dataset_root: Path, root: Path) -> Path | None:
    previews_dir = dataset_root / "previews"
    candidate_paths = [
        previews_dir / "dataset_grid.png",
        previews_dir / "grid_overview.png",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    # Attempt generation via existing visualizer CLI.
    try:
        cmd = [
            sys.executable,
            "-m",
            "src.vis.dataset_visualizer",
            "--dataset",
            str(dataset_root),
            "--grid",
            "--output",
            str(previews_dir),
        ]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=root, check=True)
    except Exception as exc:
        print(f"[WARN] Failed to generate dataset grid preview: {exc}")

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return None


def sync_project_page_assets(dataset_root: Path, root: Path) -> None:
    page_root = root / "project_page"
    assets_data = page_root / "assets" / "data"
    assets_images = page_root / "assets" / "images"
    assets_qual = assets_images / "qualitative"
    assets_plots = page_root / "assets" / "plots"

    assets_data.mkdir(parents=True, exist_ok=True)
    assets_images.mkdir(parents=True, exist_ok=True)
    assets_qual.mkdir(parents=True, exist_ok=True)
    assets_plots.mkdir(parents=True, exist_ok=True)

    results_src = dataset_root / "results.json"
    backbone_csv_src = dataset_root / "backbone_comparison.csv"
    backbone_plot_src = dataset_root / "ai_results" / "backbone_comparison.png"

    if not safe_copy(results_src, assets_data / "results.json"):
        print(f"[WARN] Missing results.json at {results_src}")
    if not safe_copy(backbone_csv_src, assets_data / "backbone_comparison.csv"):
        print(f"[WARN] Missing backbone_comparison.csv at {backbone_csv_src}")
    if not safe_copy(backbone_plot_src, assets_plots / "backbone_comparison.png"):
        print(f"[WARN] Missing backbone comparison plot at {backbone_plot_src}")

    grid_src = ensure_dataset_grid(dataset_root, root)
    if grid_src is not None:
        safe_copy(grid_src, assets_images / "dataset_grid.png")
    else:
        print("[WARN] Dataset grid image not found or generated.")

    qualitative_src = dataset_root / "ai_results" / "qualitative"
    if qualitative_src.exists():
        for img_path in sorted(qualitative_src.glob("*.png")):
            safe_copy(img_path, assets_qual / img_path.name)
    else:
        print(f"[WARN] Qualitative directory not found: {qualitative_src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VisionDrive3D AI pipeline")
    parser.add_argument("--dataset", type=str, default="./output_dataset")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_root = resolve_dataset_root(args.dataset, root)

    dataset_arg = str(dataset_root)

    if not args.skip_train:
        run_python_script(root, "src/ai_demo/train.py", ["--dataset", dataset_arg, "--batch", "32"])
    else:
        print("[INFO] Skipping training step (--skip-train)")

    run_python_script(root, "src/ai_demo/evaluate.py", ["--dataset", dataset_arg])

    if not args.skip_backbone:
        run_python_script(root, "src/ai_demo/experiment_backbones.py", ["--dataset", dataset_arg])
    else:
        print("[INFO] Skipping backbone experiment step (--skip-backbone)")

    sync_project_page_assets(dataset_root, root)
    print("Done. Open project_page/index.html to view results.")


if __name__ == "__main__":
    main()
