# annotations.py
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import OpenGL.GL as GL
from PIL import Image


class BBoxCalculator:
    """
    CPU-only 2D bbox extraction:
    local vertices -> Model -> View -> Projection -> NDC -> Pixel bbox
    """

    def __init__(self):
        pass

    @staticmethod
    def _to_homogeneous(vertices: np.ndarray) -> np.ndarray:
        ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
        return np.concatenate([vertices.astype(np.float32), ones], axis=1)

    def calculate(
        self,
        entities,
        projection: np.ndarray,
        view: np.ndarray,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        bboxes: List[Dict[str, Any]] = []
        width = int(width)
        height = int(height)

        for ent in entities:
            if ent.mesh.vertices is None or ent.mesh.vertices.size == 0:
                continue

            model = ent.world_matrix().astype(np.float32)
            mvp = (projection @ view @ model).astype(np.float32)

            v_local_h = self._to_homogeneous(ent.mesh.vertices)  # Nx4
            clip = (mvp @ v_local_h.T).T  # Nx4

            w = clip[:, 3]
            valid_w = np.abs(w) > 1e-6
            if not np.any(valid_w):
                continue

            ndc = clip[valid_w, :3] / w[valid_w, None]  # Nx3 in [-inf, inf]

            # Remove points behind far plane or near plane.
            # Requirement explicitly asks to remove objects behind camera with Z > 1.0.
            z_ok = (ndc[:, 2] <= 1.0) & (ndc[:, 2] >= -1.0)
            if not np.any(z_ok):
                continue
            ndc = ndc[z_ok]

            # Entire object outside viewport check
            if np.max(ndc[:, 0]) < -1.0 or np.min(ndc[:, 0]) > 1.0:
                continue
            if np.max(ndc[:, 1]) < -1.0 or np.min(ndc[:, 1]) > 1.0:
                continue

            # Clip against screen edges in NDC
            x_ndc = np.clip(ndc[:, 0], -1.0, 1.0)
            y_ndc = np.clip(ndc[:, 1], -1.0, 1.0)

            # NDC -> pixel
            x_pix = (x_ndc * 0.5 + 0.5) * (width - 1)
            y_pix = (1.0 - (y_ndc * 0.5 + 0.5)) * (height - 1)

            x_min = float(np.min(x_pix))
            y_min = float(np.min(y_pix))
            x_max = float(np.max(x_pix))
            y_max = float(np.max(y_pix))

            if x_max - x_min < 1.0 or y_max - y_min < 1.0:
                continue

            bboxes.append(
                {
                    "entity_name": ent.name,
                    "class_id": int(ent.class_id),
                    "xyxy": [x_min, y_min, x_max, y_max],
                }
            )

        return bboxes


class DatasetExporter:
    """
    Read framebuffer outputs and write:
    - RGB PNG
    - Mask PNG
    - Depth PNG
    - YOLO label TXT
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.mask_dir = os.path.join(output_dir, "mask")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.label_dir = os.path.join(output_dir, "labels")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    @staticmethod
    def read_rgb_buffer(width: int, height: int) -> np.ndarray:
        data = GL.glReadPixels(
            0, 0, width, height,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return np.flipud(img).copy()

    @staticmethod
    def read_depth_buffer_as_gray(width: int, height: int) -> np.ndarray:
        # Depth pass shader already writes linear depth into RGB channels.
        data = GL.glReadPixels(
            0, 0, width, height,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        img_rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        img_rgb = np.flipud(img_rgb)
        return img_rgb[:, :, 0].copy()

    @staticmethod
    def _xyxy_to_yolo(xyxy: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = xyxy
        bw = max(0.0, x_max - x_min)
        bh = max(0.0, y_max - y_min)
        cx = x_min + 0.5 * bw
        cy = y_min + 0.5 * bh

        return (
            cx / max(1, width),
            cy / max(1, height),
            bw / max(1, width),
            bh / max(1, height),
        )

    def save_frame(
        self,
        frame_idx: int,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth_gray: np.ndarray,
        bboxes: List[Dict[str, Any]],
        width: int,
        height: int,
    ):
        stem = f"{frame_idx:06d}"

        Image.fromarray(rgb, mode="RGB").save(os.path.join(self.rgb_dir, f"{stem}.png"))
        Image.fromarray(mask, mode="RGB").save(os.path.join(self.mask_dir, f"{stem}.png"))
        Image.fromarray(depth_gray, mode="L").save(os.path.join(self.depth_dir, f"{stem}.png"))

        label_path = os.path.join(self.label_dir, f"{stem}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for box in bboxes:
                cls_id = int(box["class_id"])
                x, y, w, h = self._xyxy_to_yolo(box["xyxy"], width, height)

                # Clamp to [0, 1] to avoid numerical overflow near clip edges.
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))

                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")