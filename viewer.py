# import OpenGL.GL as GL
# import glfw
# from itertools import cycle
# import os
# import sys
# import math
# import numpy as np

# # Add parent directory to path to import libs
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, parent_dir)

# from libs import transform as T
# from mesh import Mesh


# class Viewer:
#     """GLFW viewer window, with classic initialization & graphics loop."""

#     def __init__(self, width=800, height=800):
#         self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

#         glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
#         glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
#         glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
#         glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
#         glfw.window_hint(glfw.RESIZABLE, False)
#         glfw.window_hint(glfw.DEPTH_BITS, 16)
#         glfw.window_hint(glfw.DOUBLEBUFFER, True)
#         self.win = glfw.create_window(width, height, "Viewer", None, None)

#         glfw.make_context_current(self.win)

#         self.mouse = (0.0, 0.0)
#         self.last_frame_time = glfw.get_time()

#         # Free camera state (first-person style)
#         self.camera_pos = np.array([0.0, 1.2, 6.0], dtype=np.float32)
#         self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
#         self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
#         self.yaw = -90.0
#         self.pitch = 0.0
#         self.move_speed = 5.0
#         self.mouse_sensitivity = 0.12

#         glfw.set_key_callback(self.win, self.on_key)
#         glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
#         glfw.set_scroll_callback(self.win, self.on_scroll)

#         print(
#             "OpenGL",
#             GL.glGetString(GL.GL_VERSION).decode() + ", GLSL",
#             GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() + ", Renderer",
#             GL.glGetString(GL.GL_RENDERER).decode(),
#         )

#         GL.glClearColor(0.5, 0.5, 0.5, 0.1)
#         GL.glEnable(GL.GL_DEPTH_TEST)
#         GL.glDepthFunc(GL.GL_LESS)

#         self.drawables = []

#     def add(self, *drawables):
#         self.drawables.extend(drawables)

#     def run(self):
#         while not glfw.window_should_close(self.win):
#             now = glfw.get_time()
#             delta_time = max(0.0, now - self.last_frame_time)
#             self.last_frame_time = now
#             self._update_free_camera(delta_time)

#             GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

#             win_size = glfw.get_window_size(self.win)
#             view = T.lookat(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
#             projection = T.perspective(35, win_size[0] / max(1, win_size[1]), 0.1, 100.0)

#             for drawable in self.drawables:
#                 drawable.draw(projection, view, None)

#             glfw.swap_buffers(self.win)
#             glfw.poll_events()

#     def on_key(self, _win, key, _scancode, action, _mods):
#         if action == glfw.PRESS or action == glfw.REPEAT:
#             if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
#                 glfw.set_window_should_close(self.win, True)
#             if key == glfw.KEY_F:
#                 GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

#     def on_mouse_move(self, win, xpos, ypos):
#         # Hold left mouse button to look around.
#         if not glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
#             self.mouse = (xpos, ypos)
#             return
        
#         # Nếu đây là lần đầu nhấn button, hãy set position không tính offset
#         if self.mouse == (0.0, 0.0):
#             self.mouse = (xpos, ypos)
#             return

#         old_x, old_y = self.mouse
#         self.mouse = (xpos, ypos)

#         x_offset = (xpos - old_x) * self.mouse_sensitivity
#         y_offset = (old_y - ypos) * self.mouse_sensitivity

#         self.yaw += x_offset
#         self.pitch += y_offset
#         self.pitch = max(-89.0, min(89.0, self.pitch))
#         self._update_camera_front()

#     def on_scroll(self, win, _deltax, deltay):
#         del win, _deltax
#         self.move_speed = float(np.clip(self.move_speed + deltay * 0.3, 1.0, 20.0))

#     def _update_camera_front(self):
#         yaw_rad = math.radians(self.yaw)
#         pitch_rad = math.radians(self.pitch)
#         front = np.array(
#             [
#                 math.cos(yaw_rad) * math.cos(pitch_rad),
#                 math.sin(pitch_rad),
#                 math.sin(yaw_rad) * math.cos(pitch_rad),
#             ],
#             dtype=np.float32,
#         )
#         self.camera_front = front / np.linalg.norm(front)

#     def _update_free_camera(self, delta_time: float):
#         velocity = self.move_speed * delta_time
#         right = np.cross(self.camera_front, self.camera_up)
#         right /= np.linalg.norm(right)

#         if glfw.get_key(self.win, glfw.KEY_W) == glfw.PRESS:
#             self.camera_pos += self.camera_front * velocity
#         if glfw.get_key(self.win, glfw.KEY_S) == glfw.PRESS:
#             self.camera_pos -= self.camera_front * velocity
#         if glfw.get_key(self.win, glfw.KEY_A) == glfw.PRESS:
#             self.camera_pos -= right * velocity
#         if glfw.get_key(self.win, glfw.KEY_D) == glfw.PRESS:
#             self.camera_pos += right * velocity
#         if glfw.get_key(self.win, glfw.KEY_E) == glfw.PRESS:
#             self.camera_pos += self.camera_up * velocity
#         if glfw.get_key(self.win, glfw.KEY_C) == glfw.PRESS:
#             self.camera_pos -= self.camera_up * velocity


# def main():
#     viewer = Viewer()
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     # NOTE: Put `car.obj` in the same folder as this file: `BTL2/car.obj`
#     car = Mesh(os.path.join(base_dir, "assets/car.obj")).setup()
#     viewer.add(car)

#     viewer.run()
    
    
    
    
#     ##### code dưới đây mô tả cơ bản quá trình sẽ render tuần tự 
#     ##### từng chức năng dựa trên các assets có sẵn và chưa tự động hoá quá trình tạo synthetic dataset
#     # viewer = Viewer(width=800, height=800)
#     # base_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # car_mesh = Mesh(os.path.join(base_dir, "assets/car.obj")).setup()
    
#     # from entity import Scene, Entity
#     # scene = Scene()
    
#     # car1 = Entity(mesh=car_mesh, class_id=0, instance_color=(1.0, 0.0, 0.0))
#     # car1.set_transform(position=(0, 0, 0), scale=(1, 1, 1))
    
#     # car2 = Entity(mesh=car_mesh, class_id=0, instance_color=(0.0, 1.0, 0.0))
#     # car2.set_transform(positionn=(3.0, 0, -2.0), scale=(1, 1, 1))
    
#     # scene.add_entity(car1)
#     # scene.add_entity(car2)
    
#     # from renderers import RendererManager
#     # render_manager = RendererManager(scene)
    
#     # def custom_key_callback(win, key, scancode, action, mods):
#     #     if action == glfw.PRESS:
#     #         if key == glfw.KEY_1:
#     #             render_manager.set_mode("RGB")
#     #         elif key == glfw.KEY_2:
#     #             render_manager.set_mode("DEPTH")
#     #         elif key == glfw.KEY_3:
#     #             render_manager.set_mode("MASK")
#     #         elif key == glfw.KEY_4:
#     #             render_manager.toggle_bbox_visualization()
#     #         elif key == glfw.KEY_P:
#     #             render_manager.export_current_frame()
#     #     viewer.on_key(win, key, scancode, action, mods)
        
#     # glfw.set_key_callback(viewer.win, custom_key_callback)
    
#     # viewer.add(render_manager)
#     # viewer.run()


# if __name__ == "__main__":
#     glfw.init()
#     main()
#     glfw.terminate()

# viewer.py
import os
import subprocess
import tempfile
import textwrap
from itertools import cycle
from pathlib import Path

import glfw
import numpy as np
import OpenGL.GL as GL

from mesh import Mesh
from entity import Scene, Entity
from camera_suite import Camera, CameraPresetFactory, CameraManager
from renderers import RenderManager, PointLight
from car import Car


def instance_color_from_id(idx: int) -> tuple[float, float, float]:
    """
    Visually distinct colors.
    Dynamic Cars are Class 5 (Red). We still want to distinguish instances,
    so we can vary the Blue channel slightly based on Instance ID.
    Red base = 1.0, Green = 0.0.
    """
    b = float((idx * 60) % 255) / 255.0
    return (1.0, 0.0, b)


class ViewerApp:
    def __init__(self, width: int = 1280, height: int = 720):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.window = glfw.create_window(width, height, "VisionDrive3D - Synthetic Generator", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Lock cursor for FPS fly camera
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.18, 0.20, 0.24, 1.0)

        self.fill_modes = cycle([GL.GL_FILL, GL.GL_LINE, GL.GL_POINT])

        self.scene = Scene()
        
        # 1. Setup CameraManager and the default free camera
        self.camera_manager = CameraManager()
        self.free_camera = Camera(name="free_cam", resolution=(1280, 720), near=0.1, far=150.0, is_free_cam=True)
        self.free_camera.position[:] = np.array([0.0, 2.2, 12.0], dtype=np.float32)
        self.camera_manager.add_camera(self.free_camera)
        
        # We also create an ego vehicle for dataset cameras
        self.ego_vehicle = Car(
            car_folder=os.path.join("assets", "car0"),
            name="ego_vehicle",
            class_id=4, # Ego vehicle class is 4
            instance_color=(0.0, 0.0, 1.0), # Blue
        )
        self.scene.add_entity(self.ego_vehicle)
        self.ego_vehicle.position[:] = np.array([0, 0.05, 0], dtype=np.float32)

        # 2. Add nuScenes cameras into manager
        nuscenes_cams = CameraPresetFactory.create_nuscenes_surround(self.ego_vehicle)
        self.camera_manager.add_cameras(nuscenes_cams)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.render_manager = RenderManager(
            scene=self.scene,
            base_dir=base_dir,
            output_dir=os.path.join(base_dir, "outputs"),
            near=0.1,
            far=150.0,
        )

        self.last_time = glfw.get_time()
        self.last_mouse_pos = None

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)

        self._load_scene_assets(base_dir)

    @staticmethod
    def _default_lanes_config() -> list[dict]:
        return [
            {"x_center": -3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
            {"x_center": -1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
        ]

    @staticmethod
    def _resolve_blender_executable() -> str | None:
        candidates = [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return candidate
            except Exception:
                continue
        return None

    def _convert_to_glb_with_blender(self, source_path: str) -> str | None:
        blender_exec = self._resolve_blender_executable()
        if blender_exec is None:
            return None

        cache_dir = os.path.join(tempfile.gettempdir(), "visiondrive3d_cache", "road_junction")
        os.makedirs(cache_dir, exist_ok=True)

        src = Path(source_path)
        output_glb = os.path.join(cache_dir, f"{src.stem}.glb")
        script_path = os.path.join(cache_dir, f"export_{src.stem}.py")

        if os.path.exists(output_glb):
            try:
                if os.path.getmtime(output_glb) >= os.path.getmtime(source_path):
                    return output_glb
            except OSError:
                pass

        script = textwrap.dedent(
            """
            import bpy
            import os
            import sys

            argv = sys.argv
            if "--" in argv:
                argv = argv[argv.index("--") + 1 :]
            if len(argv) < 2:
                raise RuntimeError("Expected source and target path")

            source_path = argv[0]
            target_path = argv[1]
            ext = os.path.splitext(source_path)[1].lower()

            bpy.ops.wm.read_factory_settings(use_empty=True)

            if ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=source_path)
            elif ext == ".blend":
                bpy.ops.wm.open_mainfile(filepath=source_path)
            else:
                raise RuntimeError(f"Unsupported source format for blender conversion: {ext}")

            bpy.ops.export_scene.gltf(
                filepath=target_path,
                export_format="GLB",
                export_apply=True,
                export_texcoords=True,
                export_normals=True,
                export_materials="EXPORT",
                export_cameras=False,
                export_lights=False,
            )
            """
        )

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
        except OSError:
            return None

        try:
            result = subprocess.run(
                [blender_exec, "-b", "-P", script_path, "--", source_path, output_glb],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=180,
            )
            if result.returncode == 0 and os.path.exists(output_glb):
                return output_glb
        except Exception:
            return None

        return None

    def _resolve_master_road_geometry(self, base_dir: str) -> str | None:
        road_dir = os.path.join(base_dir, "assets", "road_junction")
        if not os.path.isdir(road_dir):
            return None

        entries = os.listdir(road_dir)
        priority_ext = [".obj", ".glb", ".fbx", ".blend"]

        for ext in priority_ext:
            matches = sorted([os.path.join(road_dir, e) for e in entries if e.lower().endswith(ext)])
            for path in matches:
                if ext in {".obj", ".glb"}:
                    return path

                converted = self._convert_to_glb_with_blender(path)
                if converted is not None:
                    return converted

        return None

    def _load_master_road_entity(self, base_dir: str) -> Entity | None:
        road_geom = self._resolve_master_road_geometry(base_dir)
        if road_geom is None:
            print("[road_junction] No supported geometry found (.obj/.glb/.fbx/.blend).")
            return None

        mesh = Mesh(road_geom).setup()

        # Optional MTL override to preserve road_junction material mapping if provided.
        road_dir = os.path.join(base_dir, "assets", "road_junction")
        mtl_candidates = sorted([os.path.join(road_dir, f) for f in os.listdir(road_dir) if f.lower().endswith(".mtl")])
        if mtl_candidates:
            mesh.apply_mtl_overrides(mtl_candidates[0])

        road = Entity(
            name="road_junction_master",
            mesh=mesh,
            class_id=-1,
            instance_color=(-1.0, 0.0, 0.0),
            is_dynamic=False,
        )
        road.position[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        road.scale[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add_entity(road)
        print(f"[road_junction] Loaded master geometry: {road_geom}")
        return road

    def _spawn_scene_houses_around_road(self, road_entity: Entity | None):
        houses_mesh = Mesh(os.path.join("assets", "scene", "Street environment_V01.obj")).setup()

        default_offsets = [
            (-52.0, 0.0, -32.0),
            (52.0, 0.0, -32.0),
            (-52.0, 0.0, 32.0),
            (52.0, 0.0, 32.0),
        ]
        offsets = default_offsets

        if road_entity is not None:
            aabb_min, aabb_max = road_entity.mesh.compute_local_aabb()
            width = float(max(16.0, aabb_max[0] - aabb_min[0]))
            depth = float(max(16.0, aabb_max[2] - aabb_min[2]))
            dx = 0.75 * width + 18.0
            dz = 0.75 * depth + 14.0
            offsets = [(-dx, 0.0, -dz), (dx, 0.0, -dz), (-dx, 0.0, dz), (dx, 0.0, dz)]

        for idx, (x, y, z) in enumerate(offsets):
            houses = Entity(
                name=f"scene_houses_{idx:02d}",
                mesh=houses_mesh,
                class_id=-1,
                instance_color=(-1.0, 0.0, 0.0),
                is_dynamic=False,
            )
            houses.position[:] = np.array([x, y, z], dtype=np.float32)
            houses.scale[:] = np.array([0.85, 0.85, 0.85], dtype=np.float32)
            self.scene.add_entity(houses)

    def _configure_lighting_from_road_junction(self, road_entity: Entity | None):
        if road_entity is None:
            self.render_manager.lighting.point_lights = []
            return

        aabb_min, aabb_max = road_entity.mesh.compute_local_aabb()
        cx = 0.5 * float(aabb_min[0] + aabb_max[0])
        cz = 0.5 * float(aabb_min[2] + aabb_max[2])
        span_x = float(max(20.0, aabb_max[0] - aabb_min[0]))
        span_z = float(max(20.0, aabb_max[2] - aabb_min[2]))
        lamp_y = float(max(6.0, aabb_max[1] * 0.8))

        # Auto-insert night lamps around the intersection when no dedicated night rig exists.
        lamp_positions = [
            np.array([cx - 0.35 * span_x, lamp_y, cz - 0.28 * span_z], dtype=np.float32),
            np.array([cx + 0.35 * span_x, lamp_y, cz - 0.28 * span_z], dtype=np.float32),
            np.array([cx - 0.35 * span_x, lamp_y, cz + 0.28 * span_z], dtype=np.float32),
            np.array([cx + 0.35 * span_x, lamp_y, cz + 0.28 * span_z], dtype=np.float32),
            np.array([cx - 0.12 * span_x, lamp_y + 0.4, cz], dtype=np.float32),
            np.array([cx + 0.12 * span_x, lamp_y + 0.4, cz], dtype=np.float32),
        ]
        self.render_manager.lighting.point_lights = [
            PointLight(position=pos, intensity=2.9, radius=max(span_x, span_z) * 0.45)
            for pos in lamp_positions
        ]

    def _load_scene_assets(self, base_dir: str):
        # 1) Master road junction from assets/road_junction (FBX/BLEND/OBJ/GLB + optional MTL override)
        road_entity = self._load_master_road_entity(base_dir)

        # 2) Secondary house assets from assets/scene (keep existing OBJ+MTL texture pipeline untouched)
        self._spawn_scene_houses_around_road(road_entity)
        self._configure_lighting_from_road_junction(road_entity)

        # 3) Dynamic cars (composed from assets/car0/*.glb)
        num_cars = 5
        for i in range(num_cars):
            car = Car(
                car_folder=os.path.join("assets", "car0"),
                name=f"car_{i:02d}",
                class_id=5,  # dynamic cars class is 5
                instance_color=instance_color_from_id(i + 1),
            )
            self.scene.add_entity(car)

        self.scene.spawn_cars_on_lanes(
            lanes_config=self._default_lanes_config(),
            num_cars=num_cars,
            scale_range=(0.9, 1.15),
            sat_padding=0.20,
            max_retry_per_car=120,
        )

    def _on_key(self, _win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
            return

        if key == glfw.KEY_F:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            return

        if key == glfw.KEY_1:
            self.render_manager.set_mode("RGB")
            return

        if key == glfw.KEY_2:
            self.render_manager.set_mode("MASK")
            return

        if key == glfw.KEY_3:
            self.render_manager.set_mode("DEPTH")
            return

        if key == glfw.KEY_T:
            is_dark = self.render_manager.toggle_dark_mode()
            mode_name = "Dark" if is_dark else "Day"
            print(f"Lighting mode: {mode_name}")
            return

        if key == glfw.KEY_R:
            self.scene.spawn_cars_on_lanes(
                lanes_config=self._default_lanes_config(),
                num_cars=len(self.scene.get_dynamic_entities()),
                scale_range=(0.9, 1.15),
                sat_padding=0.20,
                max_retry_per_car=120,
            )
            print("Cars respawned on lanes.")
            return

        if key == glfw.KEY_TAB:
            self.camera_manager.switch_next()
            print(f"Switched to camera: {self.camera_manager.get_active_camera().name}")
            return
            
        if key == glfw.KEY_G:
            self._run_auto_generate_hook(num_frames=500)
            return

        if key == glfw.KEY_P:
            self.render_manager.export_current_frame(self.camera_manager)
            print(f"Exported frame index: {self.render_manager.frame_idx - 1:06d}")
            return

    def _run_auto_generate_hook(self, num_frames=5):
        print(f"--- Starting automated dataset generation for {num_frames} frames ---")
        for i in range(num_frames):
            # randomizes scene layout
            self.scene.spawn_cars_on_lanes(
                lanes_config=self._default_lanes_config(),
                num_cars=len(self.scene.get_dynamic_entities()),
                scale_range=(0.9, 1.15),
                sat_padding=0.20,
                max_retry_per_car=120,
            )
            
            # Optionally randomize ego location
            tx = float(np.random.uniform(-3.0, 3.0))
            tz = float(np.random.uniform(-30.0, 5.0))
            self.ego_vehicle.position[:] = np.array([tx, 0.05, tz], dtype=np.float32)
            
            # We don't render to screen, just export the camera bounds
            self.render_manager.export_current_frame(self.camera_manager)
            print(f"Generated frame {self.render_manager.frame_idx - 1:06d}")
        print("--- Automated generation complete ---")

    def _on_mouse_move(self, _win, xpos, ypos):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (xpos, ypos)
            return

        dx = xpos - self.last_mouse_pos[0]
        dy = ypos - self.last_mouse_pos[1]
        self.last_mouse_pos = (xpos, ypos)

        # Mouse look for fly navigation
        active_cam = self.camera_manager.get_active_camera()
        active_cam.process_mouse_delta(dx, dy)

    def run(self):
        print("Controls: WASD + E/Q(C) move | Mouse look | TAB switch cam | R respawn | P export | G gen dataset | 1/2/3 mode | T day/night | F fill")

        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            dt = float(max(1e-6, now - self.last_time))
            self.last_time = now

            glfw.poll_events()

            active_cam = self.camera_manager.get_active_camera()
            active_cam.process_keyboard(self.window, dt)
            active_cam.update_ground_alignment(self.scene, dt)

            w, h = glfw.get_framebuffer_size(self.window)
            GL.glViewport(0, 0, w, h)
            clear_color = self.render_manager.get_clear_color()
            GL.glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3])
            if self.render_manager.mode != "MASK":
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Re-read active camera resolution if necessary.
            # But the viewport requires current window w, h for screen viewing
            # so we only use camera.projection_matrix if the camera has one,
            # wait, if camera has fixed resolution, it will stretch. That's fine for viewport.
            projection = active_cam.projection_matrix()
            # If our active_cam is free_cam which needs aspect, wait, we removed aspect argument
            # Let's see, we stored resolution statically in Camera, but free_cam can update it?
            # Actually our free_cam uses self.resolution inside projection_matrix
            # Let's dynamically update free_cam resolution to window size so it doesn't stretch
            if active_cam.is_free_cam:
                active_cam.resolution = (w, h)
                projection = active_cam.projection_matrix()

            self.scene.update(dt)
            view = active_cam.view_matrix()

            self.render_manager.draw(projection, view)

            glfw.swap_buffers(self.window)

        glfw.terminate()


def main():
    app = ViewerApp(width=1280, height=720)
    app.run()


if __name__ == "__main__":
    main()