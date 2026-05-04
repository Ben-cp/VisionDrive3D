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
        self.dimmed = False

    def set_dimmed(self, dimmed: bool):
        self.dimmed = bool(dimmed)

    def toggle_dimmed(self) -> bool:
        self.dimmed = not self.dimmed
        return self.dimmed

    def render(self, scene, projection: np.ndarray, view: np.ndarray):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        if self.dimmed:
            light_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            ambient_strength = 0.10
            specular_strength = 0.08
        else:
            light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            ambient_strength = 0.28
            specular_strength = 0.35

        uma.upload_uniform_vector3fv(light_color, "light_color")
        uma.upload_uniform_vector3fv(np.array([0.0, 8.0, 5.0], dtype=np.float32), "light_pos")
        uma.upload_uniform_scalar1f(64.0, "shininess")
        uma.upload_uniform_scalar1f(ambient_strength, "ambient_strength")
        uma.upload_uniform_scalar1f(specular_strength, "specular_strength")

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
        prev_clear = GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE)
        GL.glClearColor(0.0, 1.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        for ent in scene.entities:
            uma.upload_uniform_vector3fv(ent.instance_color, "instance_color")
            ent.mesh.draw(projection, view, ent.world_matrix(), self.shader)
            
        GL.glClearColor(*prev_clear)


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

    def __init__(self, scene, base_dir: str, scene_overlay=None, output_dir: str = "outputs", near: float = 0.1, far: float = 150.0):
        self.scene = scene
        self.rgb_renderer = RGBRenderer(base_dir)
        self.mask_renderer = MaskRenderer(base_dir)
        self.depth_renderer = DepthRenderer(base_dir, near=near, far=far)
        self.scene_overlay = scene_overlay
        self.bbox_calculator = BBoxCalculator()
        self.exporter = DatasetExporter(output_dir=output_dir)

        self.mode = "RGB"
        self.frame_idx = 0

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode in {"RGB", "MASK", "DEPTH"}:
            self.mode = mode

    def toggle_rgb_dimmed(self) -> bool:
        return self.rgb_renderer.toggle_dimmed()

    def is_rgb_dimmed(self) -> bool:
        return self.rgb_renderer.dimmed

    @staticmethod
    def _reset_texture_state(shader=None):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        if shader is not None:
            GL.glUseProgram(shader.render_idx)
            loc_has_texture = GL.glGetUniformLocation(shader.render_idx, "has_texture")
            if loc_has_texture != -1:
                GL.glUniform1i(loc_has_texture, 0)

    def draw(self, projection: np.ndarray, view: np.ndarray):
        self._reset_texture_state()

        if self.mode == "RGB":
            self.rgb_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.rgb_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(
                    self.rgb_renderer.shader,
                    projection,
                    view,
                    is_rgb=True,
                    rgb_dimmed=self.is_rgb_dimmed(),
                )
            self._reset_texture_state(self.rgb_renderer.shader)

        elif self.mode == "MASK":
            self._reset_texture_state(self.mask_renderer.shader)
            self.mask_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.mask_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(self.mask_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.mask_renderer.shader)

        elif self.mode == "DEPTH":
            self._reset_texture_state(self.depth_renderer.shader)
            self.depth_renderer.render(self.scene, projection, view)
            self._reset_texture_state(self.depth_renderer.shader)
            if self.scene_overlay:
                self.scene_overlay.render(self.depth_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.depth_renderer.shader)

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
            self._reset_texture_state(self.rgb_renderer.shader)
            self.rgb_renderer.render(self.scene, projection, view)
            if self.scene_overlay:
                self.scene_overlay.render(
                    self.rgb_renderer.shader,
                    projection,
                    view,
                    is_rgb=True,
                    rgb_dimmed=self.is_rgb_dimmed(),
                )
            self._reset_texture_state(self.rgb_renderer.shader)
            GL.glFinish()
            rgb = self.exporter.read_rgb_buffer(w, h)

            # 2) Mask pass (MaskRenderer clears its own color for Sky)
            self._reset_texture_state(self.mask_renderer.shader)
            self.mask_renderer.render(self.scene, projection, view)
            if self.scene_overlay:
                self.scene_overlay.render(self.mask_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.mask_renderer.shader)
            GL.glFinish()
            mask = self.exporter.read_rgb_buffer(w, h)

            # 3) Depth pass
            GL.glClearColor(1.0, 1.0, 1.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._reset_texture_state(self.depth_renderer.shader)
            self.depth_renderer.render(self.scene, projection, view)
            if self.scene_overlay:
                self.scene_overlay.render(self.depth_renderer.shader, projection, view, is_rgb=False)
            self._reset_texture_state(self.depth_renderer.shader)
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

        # Store paths for active camera hook access
        active_cam_name = camera_manager.get_active_camera().name
        stem = f"{self.frame_idx:06d}"
        pfx = f"{active_cam_name}_{stem}"
        self.last_exported_rgb_path = os.path.join(self.exporter.rgb_dir, f"{pfx}.png")
        self.last_exported_mask_path = os.path.join(self.exporter.mask_dir, f"{pfx}.png")
        self.last_exported_depth_path = os.path.join(self.exporter.depth_dir, f"{pfx}.png")

        self.frame_idx += 1
        self.mode = old_mode
