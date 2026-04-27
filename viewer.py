import os
import random
from itertools import cycle

import glfw
import numpy as np
import OpenGL.GL as GL

from mesh import Mesh
from entity import Scene, Entity
from camera_suite import Camera, CameraPresetFactory, CameraManager
from renderers import RenderManager
from car import Car
from scene_overlay import SceneOverlay
from traffic import TrafficManager
from src.dataset.dataset_manager import DatasetManager


def instance_color_from_id(idx: int) -> tuple[float, float, float]:
    """
    Generate distinct RGB colors using bit manipulation.
    Works well for many instances.
    """
    idx = idx * 2654435761  # Knuth hash (helps distribution)

    r = (idx & 0xFF) / 255.0
    g = ((idx >> 8) & 0xFF) / 255.0
    b = ((idx >> 16) & 0xFF) / 255.0

    return (r, g, b)


class ViewerApp:
    def __init__(self, width: int = 1280, height: int = 720, headless: bool = False):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.headless = headless

        if headless:
            self._init_offscreen(width, height)
        else:
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

        # 2. Create nuScenes cameras WITHOUT a parent for now
        #    They will be re-parented to a traffic car once the pool is ready.
        self._nuscenes_cameras = CameraPresetFactory.create_nuscenes_surround(None)
        self.camera_manager.add_cameras(self._nuscenes_cameras)
        self._camera_host_car = None  # the traffic car cameras are attached to

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.scene_overlay = SceneOverlay()
        self.render_manager = RenderManager(
            scene=self.scene,
            base_dir=base_dir,
            scene_overlay=self.scene_overlay,
            output_dir=os.path.join(base_dir, "outputs"),
            near=0.1,
            far=150.0,
        )
        
        self.last_time = glfw.get_time()
        self.last_mouse_pos = None

        # Traffic manager (wired after scene assets load)
        self.traffic_manager = TrafficManager()

        if not self.headless:
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)

        self._load_scene_assets(base_dir)

        self.scene_id = 0
        self.dataset_manager = DatasetManager(
            root="./output_dataset", validate=True
        )

    # ------------------------------------------------------------------
    # Camera ↔ traffic car attachment
    # ------------------------------------------------------------------
    def _attach_cameras_to_car(self, car):
        """Re-parent all nuScenes cameras to the given traffic car."""
        self._camera_host_car = car
        for cam in self._nuscenes_cameras:
            car.add_child(cam)
        print(f"Cameras attached to: {car.name}")

    def _ensure_camera_host(self):
        """
        If the current host car is gone / finished / inactive,
        pick a random active traffic car and re-attach cameras.
        """
        host = self._camera_host_car
        need_new = (
            host is None
            or not host.is_active
            or host.route_finished
        )
        if not need_new:
            return

        active = [c for c in self.traffic_manager.cars
                  if c.is_active and not c.route_finished]
        if not active:
            return  # no cars available yet

        new_host = random.choice(active)
        self._attach_cameras_to_car(new_host)

    @staticmethod
    def _default_lanes_config() -> list[dict]:
        return [
            {"x_center": -3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
            {"x_center": -1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, -1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 1.8, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.22, "safe_z": 3.0},
            {"x_center": 3.5, "z_min": -42.0, "z_max": 10.0, "direction": [0.0, 0.0, 1.0], "ground_y": 0.05, "x_jitter": 0.28, "safe_z": 3.2},
        ]

    def _load_scene_assets(self, base_dir: str):
        # 1) Static environment (do not randomize its transform)
        street_obj = os.path.join("assets", "scene", "Street environment_V01.obj")
        street_mesh = Mesh(street_obj).setup()
        street = Entity(
            name="street_environment",
            mesh=street_mesh,
            class_id=-1,
            instance_color=(-1.0, 0.0, 0.0), # Flag for ground/house dynamic shader
            is_dynamic=False,
        )
        street.position[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        street.scale[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.add_entity(street)

        # 2) Dynamic cars – Object Pool (5 cars, random car0/car1/car2)
        car_folders = ["assets/car0", "assets/car1", "assets/car2"]
        num_cars = 5
        for i in range(num_cars):
            folder = random.choice(car_folders)
            car = Car(
                car_folder=folder,
                name=f"car_{i:02d}",
                class_id=5,  # dynamic traffic cars
                instance_color=instance_color_from_id(i + 1),
            )
            self.scene.add_entity(car)

        # 3) Initialize traffic manager with the car pool
        traffic_cars = self.scene.get_traffic_cars()
        self.traffic_manager.register_cars(traffic_cars)

        # 4) Attach nuScenes cameras to a random traffic car
        self._ensure_camera_host()

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

        if key == glfw.KEY_R:
            # Reset all traffic routes
            traffic_cars = self.scene.get_traffic_cars()
            self.traffic_manager.register_cars(traffic_cars)
            self._ensure_camera_host()
            print("Traffic routes reset.")
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

            # Re-attach cameras to a random active car for this frame
            self._ensure_camera_host()

            # We don't render to screen, just export the camera bounds
            self.render_manager.export_current_frame(self.camera_manager)
            print(f"Generated frame {self.render_manager.frame_idx - 1:06d}")

            # --- Dataset export hook ---
            frame_idx = self.render_manager.frame_idx - 1
            paths = self.dataset_manager.begin_export(self.scene_id)

            active_cam = self.camera_manager.get_active_camera()
            camera_params = {
                "position": active_cam.position.tolist(),
                "target":   active_cam.target.tolist() if hasattr(active_cam, 'target') else (active_cam.position + active_cam.front).tolist(),
                "up":       active_cam.up.tolist() if hasattr(active_cam, 'up') else [0,1,0],
                "fov_deg":  active_cam.fov if hasattr(active_cam, 'fov') else 45.0,
                "near":     active_cam.near,
                "far":      active_cam.far,
                "intrinsic_matrix":  active_cam.get_intrinsics().tolist(),
                "extrinsic_matrix":  active_cam.get_extrinsics().tolist(),
            }

            objects = []
            for entity in self.scene.entities:
                objects.append({
                    "instance_id":    getattr(entity, "instance_id", id(entity)),
                    "class_name":     getattr(entity, "class_name", type(entity).__name__),
                    "position_world": entity.position.tolist(),
                    "rotation_euler": getattr(entity, "rotation", [0,0,0]),
                    "scale":          getattr(entity, "scale", [1,1,1]),
                    "bbox_2d":        getattr(entity, "bbox_2d", [0,0,0,0]),
                    "visible":        getattr(entity, "visible", True),
                    "occlusion_ratio":getattr(entity, "occlusion_ratio", 0.0),
                })

            render_config = {
                "resolution": list(active_cam.resolution),
                "lighting":   "directional",
                "num_lights": len(getattr(self.scene, "lights", [])),
                "background": "road_scene",
            }

            # Read back files already written by export_current_frame
            import cv2, numpy as np
            from pathlib import Path

            existing_rgb   = cv2.imread(str(self.render_manager.last_exported_rgb_path))
            existing_depth = np.load(str(self.render_manager.last_exported_depth_path)) \
                             if Path(self.render_manager.last_exported_depth_path).suffix == ".npy" \
                             else cv2.imread(str(self.render_manager.last_exported_depth_path), cv2.IMREAD_GRAYSCALE)
            existing_mask  = cv2.imread(str(self.render_manager.last_exported_mask_path), cv2.IMREAD_GRAYSCALE)

            self.dataset_manager.finish_scene(
                scene_id=self.scene_id,
                camera_params=camera_params,
                objects=objects,
                render_config=render_config,
                rgb_image=existing_rgb,
                depth_map=existing_depth,
                seg_mask=existing_mask,
            )
            self.scene_id += 1
            # --- End dataset export hook ---
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

    def _init_offscreen(self, width: int, height: int):
        """
        Initialize an OpenGL offscreen context using GLFW with a hidden window.
        Falls back to OSMesa if GLFW headless hint is unavailable.
        """
        # Tell GLFW not to show the window at all
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, "offscreen", None, None)
        if not self.window:
            raise RuntimeError(
                "GLFW offscreen context creation failed. "
                "On Linux servers, try: export DISPLAY=:99 or install osmesa."
            )
        glfw.make_context_current(self.window)

        # Create FBO + texture + depth renderbuffer
        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        self._color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                        width, height, 0,
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                                  GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self._color_tex, 0)

        self._depth_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._depth_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8,
                                 width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                                     GL.GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL.GL_RENDERBUFFER, self._depth_rbo)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete, status: {hex(status)}")

        self._offscreen_width  = width
        self._offscreen_height = height

    def _destroy_offscreen(self):
        GL.glDeleteFramebuffers(1, [self._fbo])
        GL.glDeleteTextures(1, [self._color_tex])
        GL.glDeleteRenderbuffers(1, [self._depth_rbo])
        glfw.destroy_window(self.window)
        glfw.terminate()

    def run_headless(self, num_frames: int, output: str):
        """
        Run dataset generation fully offscreen. No window is shown.
        Renders num_frames scenes and exports them via DatasetManager.
        """
        w = self._offscreen_width
        h = self._offscreen_height

        self.dataset_manager = DatasetManager(root=output, validate=True)

        for i in range(num_frames):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
            GL.glViewport(0, 0, w, h)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            # Randomize scene for variety (reuse existing scene logic)
            self.scene.spawn_cars_on_lanes(
                lanes_config=self._default_lanes_config(),
                num_cars=len(self.scene.get_dynamic_entities()),
                scale_range=(0.9, 1.15),
                sat_padding=0.20,
                max_retry_per_car=120,
            )
            self._ensure_camera_host()
            self.traffic_manager.update(0.016)
            self.scene.update(0.016)

            active_cam = self.camera_manager.get_active_camera()
            # To get ground alignment properly if needed
            active_cam.update_ground_alignment(self.scene, 0.016)
            
            projection = active_cam.get_projection_matrix(w / h) if hasattr(active_cam, 'get_projection_matrix') else active_cam.projection_matrix()
            view       = active_cam.get_view_matrix() if hasattr(active_cam, 'get_view_matrix') else active_cam.view_matrix()

            self.render_manager.draw(projection, view)

            # Read pixels directly from FBO — no window swap needed
            GL.glFlush()
            raw = GL.glReadPixels(0, 0, w, h,
                                  GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            import numpy as np
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            rgb = np.flipud(rgb)           # OpenGL origin is bottom-left

            # Read depth buffer
            raw_d = GL.glReadPixels(0, 0, w, h,
                                    GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
            depth = np.frombuffer(raw_d, dtype=np.float32).reshape(h, w)
            depth = np.flipud(depth)

            # Linearize depth from NDC [0,1] to world units
            near  = active_cam.near
            far   = active_cam.far
            depth = (2.0 * near * far) / (far + near - (depth * 2.0 - 1.0) * (far - near))

            # Trigger mask render pass before reading the mask buffer
            self.render_manager._reset_texture_state(self.render_manager.mask_renderer.shader)
            self.render_manager.mask_renderer.render(self.scene, projection, view)
            if self.render_manager.scene_overlay:
                self.render_manager.scene_overlay.render(self.render_manager.mask_renderer.shader, projection, view, is_rgb=False)
            self.render_manager._reset_texture_state(self.render_manager.mask_renderer.shader)
            GL.glFinish()

            if hasattr(self.render_manager, 'exporter') and hasattr(self.render_manager.exporter, 'read_rgb_buffer'):
                seg_mask = self.render_manager.exporter.read_rgb_buffer(w, h)
            elif hasattr(self.render_manager, 'get_segmentation_mask'):
                seg_mask = self.render_manager.get_segmentation_mask(w, h)
            else:
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                seg_mask = (depth_norm * 255).astype(np.uint8)

            def project_entity_bbox(entity, projection, view, w, h):
                """
                Project the 8 corners of the entity's axis-aligned bounding box
                into screen space and return the 2D screen-space bbox that
                tightly contains all visible corners.
                """
                import numpy as np

                # --- Get entity transform components ---
                pos   = np.array(getattr(entity, "position", [0,0,0])[:3],
                                 dtype=np.float32)
                scale = np.array(getattr(entity, "scale", [1,1,1]),
                                 dtype=np.float32)
                if np.isscalar(scale):
                    scale = np.array([scale, scale, scale], dtype=np.float32)
                scale = np.atleast_1d(scale)
                if scale.shape[0] == 1:
                    scale = np.array([scale[0], scale[0], scale[0]])

                # --- Estimate half-extents from entity type ---
                # Try to read from entity.mesh_extents or entity.bbox_local if
                # they exist. Otherwise use class-name defaults.
                local_extents = getattr(entity, "mesh_extents", None)
                if local_extents is None:
                    local_extents = getattr(entity, "bbox_local", None)

                if local_extents is not None:
                    # entity exposes its own local AABB half-extents [hx, hy, hz]
                    hx, hy, hz = np.array(local_extents, dtype=np.float32) * scale
                else:
                    # Fallback: class-name based defaults in world units (metres)
                    class_name = getattr(entity, "class_name",
                                         type(entity).__name__).lower()
                    defaults = {
                        "car":          (2.0, 0.8, 1.0),   # half: width, height, depth
                        "truck":        (2.5, 1.2, 1.5),
                        "pedestrian":   (0.4, 0.9, 0.4),
                        "trafficlight": (0.2, 0.6, 0.2),
                        "trafficsign":  (0.3, 0.3, 0.1),
                        "entity":       (0.3, 0.3, 0.3),
                    }
                    # Fuzzy match: check if any key is a substring of class_name
                    hw, hh, hd = (1.0, 0.5, 0.5)  # generic fallback
                    for key, dims in defaults.items():
                        if key in class_name:
                            hw, hh, hd = dims
                            break
                    hx = hw * float(scale[0])
                    hy = hh * float(scale[1])
                    hz = hd * float(scale[2])

                # --- Build 8 corners of the AABB in world space ---
                corners_local = np.array([
                    [-hx, -hy, -hz],
                    [ hx, -hy, -hz],
                    [-hx,  hy, -hz],
                    [ hx,  hy, -hz],
                    [-hx, -hy,  hz],
                    [ hx, -hy,  hz],
                    [-hx,  hy,  hz],
                    [ hx,  hy,  hz],
                ], dtype=np.float32)

                # Apply rotation if available (Y-axis rotation only, from euler degrees)
                rotation = getattr(entity, "rotation_euler",
                           getattr(entity, "rotation", [0,0,0]))
                if rotation is not None:
                    ry = float(np.array(rotation).flat[1] if
                               len(np.atleast_1d(rotation)) > 1 else 0)
                    ry_rad = np.deg2rad(ry)
                    cos_r, sin_r = np.cos(ry_rad), np.sin(ry_rad)
                    Ry = np.array([
                        [ cos_r, 0, sin_r],
                        [     0, 1,     0],
                        [-sin_r, 0, cos_r],
                    ], dtype=np.float32)
                    corners_local = (Ry @ corners_local.T).T

                corners_world = corners_local + pos  # translate to world space

                # --- Project each corner through view→projection ---
                PV = np.array(projection, dtype=np.float32) @ \
                     np.array(view,       dtype=np.float32)

                screen_xs, screen_ys = [], []
                for corner in corners_world:
                    clip = PV @ np.array([*corner, 1.0], dtype=np.float32)
                    if clip[3] <= 0:
                        continue                       # behind camera — skip corner
                    ndc = clip[:3] / clip[3]
                    sx = (ndc[0] * 0.5 + 0.5) * w
                    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * h
                    screen_xs.append(sx)
                    screen_ys.append(sy)

                if not screen_xs:
                    return [0, 0, 0, 0]              # all corners behind camera

                x0 = max(0, int(min(screen_xs)))
                y0 = max(0, int(min(screen_ys)))
                x1 = min(w, int(max(screen_xs)))
                y1 = min(h, int(max(screen_ys)))

                if x1 - x0 < 4 or y1 - y0 < 4:
                    return [0, 0, 0, 0]              # degenerate box

                return [x0, y0, x1, y1]

            entity_bboxes = {}
            for entity in self.scene.entities:
                try:
                    entity_bboxes[id(entity)] = project_entity_bbox(entity, projection, view, w, h)
                except Exception:
                    entity_bboxes[id(entity)] = [0, 0, 0, 0]

            # Build camera and object dicts (same adapter code as viewer.py hook)
            camera_params = {
                "position":         active_cam.position.tolist(),
                "target":           active_cam.target.tolist() if hasattr(active_cam, 'target') else (active_cam.position + active_cam.front).tolist(),
                "up":               active_cam.up.tolist() if hasattr(active_cam, 'up') else [0,1,0],
                "fov_deg":          active_cam.fov if hasattr(active_cam, 'fov') else 45.0,
                "near":             near,
                "far":              far,
                "intrinsic_matrix": active_cam.get_intrinsics().tolist() if hasattr(active_cam, 'get_intrinsics') else np.eye(3).tolist(),
                "extrinsic_matrix": active_cam.get_extrinsics().tolist() if hasattr(active_cam, 'get_extrinsics') else np.eye(4).tolist(),
            }
            objects = []
            for entity in self.scene.entities:
                objects.append({
                    "instance_id":     getattr(entity, "instance_id", id(entity)),
                    "class_name":      getattr(entity, "class_name", type(entity).__name__),
                    "position_world":  entity.position.tolist(),
                    "rotation_euler":  getattr(entity, "rotation", [0, 0, 0]),
                    "scale":           getattr(entity, "scale", [1, 1, 1]),
                    "bbox_2d":         entity_bboxes.get(id(entity), [0, 0, 0, 0]),
                    "visible":         getattr(entity, "visible", True),
                    "occlusion_ratio": getattr(entity, "occlusion_ratio", 0.0),
                })
            render_config = {
                "resolution": [w, h],
                "lighting":   "directional",
                "num_lights": len(getattr(self.scene, "lights", [])),
                "background": "road_scene",
            }

            paths = self.dataset_manager.begin_export(i)
            
            # Save copies to dataset paths (DatasetManager may not write files
            # directly — ensure RGB and depth are saved to the canonical paths)
            import cv2
            cv2.imwrite(str(paths["image"]), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(str(paths["depth"]), depth)
            cv2.imwrite(str(paths["mask"]), seg_mask)

            self.dataset_manager.finish_scene(
                scene_id=i,
                camera_params=camera_params,
                objects=objects,
                render_config=render_config,
                rgb_image=rgb,
                depth_map=depth,
                seg_mask=seg_mask,
            )

            print(f"[headless] scene {i+1}/{num_frames} exported → {paths['image']}")

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self._destroy_offscreen()
        print(f"[headless] done. Dataset written to: {output}")

    def run(self):
        print("Controls: WASD + E/Q(C) move | Mouse look | TAB switch cam | R respawn | P export | G gen dataset | 1/2/3 mode | F fill")

        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            dt = float(max(1e-6, now - self.last_time))
            self.last_time = now

            glfw.poll_events()

            # Auto-switch camera host if current car is finished/inactive
            self._ensure_camera_host()

            active_cam = self.camera_manager.get_active_camera()
            active_cam.process_keyboard(self.window, dt)
            active_cam.update_ground_alignment(self.scene, dt)

            w, h = glfw.get_framebuffer_size(self.window)
            GL.glViewport(0, 0, w, h)
            # Default clearing for RGB interactive viewing
            GL.glClearColor(0.18, 0.20, 0.24, 1.0)
            if self.render_manager.mode != "MASK":
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            projection = active_cam.projection_matrix()
            if active_cam.is_free_cam:
                active_cam.resolution = (w, h)
                projection = active_cam.projection_matrix()

            # Traffic AI: collision avoidance + route recycling
            self.traffic_manager.update(dt)
            self.scene.update(dt)
            view = active_cam.view_matrix()

            self.render_manager.draw(projection, view)

            glfw.swap_buffers(self.window)

        glfw.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionDrive3D")
    parser.add_argument("--headless",    action="store_true", help="Run offscreen, no window")
    parser.add_argument("--frames",      type=int, default=10, help="Number of scenes to generate (headless only)")
    parser.add_argument("--output",      type=str, default="./output_dataset", help="Dataset output directory (headless only)")
    parser.add_argument("--width",       type=int, default=1280)
    parser.add_argument("--height",      type=int, default=720)
    args = parser.parse_args()

    if not glfw.init():
        raise RuntimeError("GLFW failed to initialize")

    if args.headless:
        app = ViewerApp(width=args.width, height=args.height, headless=True)
        app.run_headless(num_frames=args.frames, output=args.output)
    else:
        app = ViewerApp(width=args.width, height=args.height, headless=False)
        app.run()