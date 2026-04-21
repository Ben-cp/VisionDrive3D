# renderers.py
import os
from typing import Tuple, Optional

import OpenGL.GL as GL
import numpy as np

from libs.shader import Shader
from libs.buffer import UManager
from libs.lighting import LightingManager
from annotations import BBoxCalculator, DatasetExporter


class RGBRenderer:
    def __init__(self, base_dir: str):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "phong.vert"),
            os.path.join(base_dir, "shaders", "phong.frag"),
        )

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        uma.upload_uniform_vector3fv(np.array([1.0, 1.0, 1.0], dtype=np.float32), "light_color")
        uma.upload_uniform_vector3fv(np.array([0.0, 8.0, 5.0], dtype=np.float32), "light_pos")
        uma.upload_uniform_scalar1f(64.0, "shininess")
        uma.upload_uniform_scalar1f(0.28, "ambient_strength")
        uma.upload_uniform_scalar1f(0.35, "specular_strength")

        for ent in scene.entities:
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)


class MaskRenderer:
    def __init__(self, base_dir: str):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "mask.vert"),
            os.path.join(base_dir, "shaders", "mask.frag"),
        )

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        # Clear specific to Mask: Sky semantic class = 1 (Bright Green visually)
        GL.glClearColor(0.0, 1.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        for ent in scene.entities:
            uma.upload_uniform_vector3fv(ent.instance_color, "instance_color")
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)


class DepthRenderer:
    def __init__(self, base_dir: str, near: float = 0.1, far: float = 150.0):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "depth.vert"),
            os.path.join(base_dir, "shaders", "depth.frag"),
        )
        self.near = float(near)
        self.far = float(far)

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)
        uma.upload_uniform_scalar1f(self.near, "near")
        uma.upload_uniform_scalar1f(self.far, "far")

        for ent in scene.entities:
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)


class RenderManager:
    """
    Multi-pass manager:
    - interactive draw mode
    - one-call export_current_frame(): RGB + Mask + Depth + YOLO labels
    """

    def __init__(self, scene, base_dir: str, output_dir: str = "outputs", near: float = 0.1, far: float = 150.0):
        self.scene = scene
        self.rgb_renderer = RGBRenderer(base_dir)
        self.mask_renderer = MaskRenderer(base_dir)
        self.depth_renderer = DepthRenderer(base_dir, near=near, far=far)

        self.bbox_calculator = BBoxCalculator()
        self.exporter = DatasetExporter(output_dir=output_dir)

        self.mode = "RGB"
        self.frame_idx = 0

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode in {"RGB", "MASK", "DEPTH"}:
            self.mode = mode

    def draw(self, projection: np.ndarray, view: np.ndarray):
        if self.mode == "RGB":
            self.rgb_renderer.render(self.scene, projection, view)
        elif self.mode == "MASK":
            self.mask_renderer.render(self.scene, projection, view)
        elif self.mode == "DEPTH":
            self.depth_renderer.render(self.scene, projection, view)

    def _render_pass(self, renderer, projection: np.ndarray, view: np.ndarray):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        renderer.render(self.scene, projection, view)
        GL.glFinish()

    def export_current_frame(self, camera_manager):
        old_mode = self.mode
        camera_data = {}

        dyn = self.scene.get_dynamic_entities()

        for cam in camera_manager.get_all_cameras():
            w, h = cam.resolution
            projection = cam.projection_matrix()
            view = cam.view_matrix()
            
            GL.glViewport(0, 0, w, h)

            # 1) RGB pass
            GL.glClearColor(0.18, 0.20, 0.24, 1.0) # normal sky clear color
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.rgb_renderer.render(self.scene, projection, view)
            GL.glFinish()
            rgb = self.exporter.read_rgb_buffer(w, h)

            # 2) Mask pass (MaskRenderer clears its own color for Sky)
            self.mask_renderer.render(self.scene, projection, view)
            GL.glFinish()
            mask = self.exporter.read_rgb_buffer(w, h)

            # 3) Depth pass
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.depth_renderer.render(self.scene, projection, view)
            GL.glFinish()
            depth_gray = self.exporter.read_depth_buffer_as_gray(w, h)

            # 4) CPU bbox from dynamic entities
            bboxes = self.bbox_calculator.calculate(dyn, projection, view, w, h)

            camera_data[cam.name] = {
                "rgb": rgb,
                "mask": mask,
                "depth_gray": depth_gray,
                "bboxes": bboxes,
                "intrinsics": cam.get_intrinsics(),
                "extrinsics": cam.get_extrinsics(),
                "width": w,
                "height": h
            }

        # 5) Write dataset files
        self.exporter.save_multi_view(
            frame_idx=self.frame_idx,
            camera_data=camera_data
        )

        self.frame_idx += 1
        self.mode = old_mode
