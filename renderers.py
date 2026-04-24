# renderers.py
import os
from dataclasses import dataclass, field
from typing import List

import OpenGL.GL as GL
import numpy as np

from libs.shader import Shader
from libs.buffer import UManager
from annotations import BBoxCalculator, DatasetExporter


@dataclass
class PointLight:
    position: np.ndarray
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.95, 0.78], dtype=np.float32))
    intensity: float = 2.4
    radius: float = 16.0


@dataclass
class LightingProfile:
    is_dark_mode: bool = False
    day_clear_color: tuple[float, float, float, float] = (0.18, 0.20, 0.24, 1.0)
    night_clear_color: tuple[float, float, float, float] = (0.02, 0.03, 0.08, 1.0)
    directional_light_world: np.ndarray = field(default_factory=lambda: np.array([18.0, 28.0, 10.0], dtype=np.float32))
    directional_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.98, 0.94], dtype=np.float32))
    day_ambient_strength: float = 0.34
    night_ambient_strength: float = 0.02
    day_specular_strength: float = 0.36
    night_specular_strength: float = 0.18
    day_directional_intensity: float = 1.0
    night_directional_intensity: float = 0.32
    day_emissive_scale: float = 0.05
    night_emissive_scale: float = 2.8
    point_lights: List[PointLight] = field(default_factory=list)

    def clear_color(self) -> tuple[float, float, float, float]:
        return self.night_clear_color if self.is_dark_mode else self.day_clear_color

    def ambient_strength(self) -> float:
        return self.night_ambient_strength if self.is_dark_mode else self.day_ambient_strength

    def specular_strength(self) -> float:
        return self.night_specular_strength if self.is_dark_mode else self.day_specular_strength

    def directional_intensity(self) -> float:
        return self.night_directional_intensity if self.is_dark_mode else self.day_directional_intensity

    def emissive_scale(self) -> float:
        return self.night_emissive_scale if self.is_dark_mode else self.day_emissive_scale

    def toggle_dark_mode(self) -> bool:
        self.is_dark_mode = not self.is_dark_mode
        return self.is_dark_mode


class RGBRenderer:
    MAX_POINT_LIGHTS = 16

    def __init__(self, base_dir: str):
        self.shader = Shader(
            os.path.join(base_dir, "shaders", "phong.vert"),
            os.path.join(base_dir, "shaders", "phong.frag"),
        )

    @staticmethod
    def _to_view_space(view: np.ndarray, world_position: np.ndarray) -> np.ndarray:
        pos4 = np.array([world_position[0], world_position[1], world_position[2], 1.0], dtype=np.float32)
        return (view @ pos4)[:3].astype(np.float32)

    def _upload_point_lights(self, profile: LightingProfile, view: np.ndarray):
        active = profile.point_lights[: self.MAX_POINT_LIGHTS]
        shader_id = self.shader.render_idx

        num_loc = GL.glGetUniformLocation(shader_id, "num_point_lights")
        if num_loc != -1:
            GL.glUniform1i(num_loc, len(active))

        for idx, light in enumerate(active):
            p_view = self._to_view_space(view, np.asarray(light.position, dtype=np.float32))
            color = np.asarray(light.color, dtype=np.float32)

            loc_pos = GL.glGetUniformLocation(shader_id, f"point_light_pos[{idx}]")
            loc_col = GL.glGetUniformLocation(shader_id, f"point_light_color[{idx}]")
            loc_intensity = GL.glGetUniformLocation(shader_id, f"point_light_intensity[{idx}]")
            loc_radius = GL.glGetUniformLocation(shader_id, f"point_light_range[{idx}]")

            if loc_pos != -1:
                GL.glUniform3fv(loc_pos, 1, p_view)
            if loc_col != -1:
                GL.glUniform3fv(loc_col, 1, color)
            if loc_intensity != -1:
                GL.glUniform1f(loc_intensity, float(light.intensity))
            if loc_radius != -1:
                GL.glUniform1f(loc_radius, float(max(0.01, light.radius)))

    def render(self, scene, projection: np.ndarray, view: np.ndarray, lighting_profile: LightingProfile):
        GL.glUseProgram(self.shader.render_idx)
        uma = UManager(self.shader)

        light_view = self._to_view_space(view, lighting_profile.directional_light_world)
        light_color = np.asarray(lighting_profile.directional_color, dtype=np.float32) * float(lighting_profile.directional_intensity())

        uma.upload_uniform_vector3fv(light_color, "light_color")
        uma.upload_uniform_vector3fv(light_view, "light_pos")
        uma.upload_uniform_scalar1f(64.0, "shininess")
        uma.upload_uniform_scalar1f(float(lighting_profile.ambient_strength()), "ambient_strength")
        uma.upload_uniform_scalar1f(float(lighting_profile.specular_strength()), "specular_strength")
        uma.upload_uniform_scalar1f(float(lighting_profile.emissive_scale()), "emissive_global_scale")

        self._upload_point_lights(lighting_profile, view)

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
        self.lighting = LightingProfile()
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
            self.rgb_renderer.render(self.scene, projection, view, self.lighting)
        elif self.mode == "MASK":
            self.mask_renderer.render(self.scene, projection, view)
        elif self.mode == "DEPTH":
            self.depth_renderer.render(self.scene, projection, view)

    def toggle_dark_mode(self) -> bool:
        return self.lighting.toggle_dark_mode()

    def set_dark_mode(self, enabled: bool):
        self.lighting.is_dark_mode = bool(enabled)

    def get_clear_color(self) -> tuple[float, float, float, float]:
        return self.lighting.clear_color()

    def _render_pass(self, renderer, projection: np.ndarray, view: np.ndarray):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if renderer is self.rgb_renderer:
            renderer.render(self.scene, projection, view, self.lighting)
        else:
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
            cc = self.get_clear_color()
            GL.glClearColor(cc[0], cc[1], cc[2], cc[3])
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.rgb_renderer.render(self.scene, projection, view, self.lighting)
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
