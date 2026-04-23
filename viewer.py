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
from traffic import TrafficManager


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

        # 2. Create nuScenes cameras WITHOUT a parent for now
        #    They will be re-parented to a traffic car once the pool is ready.
        self._nuscenes_cameras = CameraPresetFactory.create_nuscenes_surround(None)
        self.camera_manager.add_cameras(self._nuscenes_cameras)
        self._camera_host_car = None  # the traffic car cameras are attached to

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

        # Traffic manager (wired after scene assets load)
        self.traffic_manager = TrafficManager()

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)

        self._load_scene_assets(base_dir)

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


def main():
    app = ViewerApp(width=1280, height=720)
    app.run()


if __name__ == "__main__":
    main()