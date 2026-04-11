import OpenGL.GL as GL
import glfw
from itertools import cycle
import os
import sys
import math
import numpy as np

# Add parent directory to path to import libs
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from libs import transform as T
from mesh import Mesh


class Viewer:
    """GLFW viewer window, with classic initialization & graphics loop."""

    def __init__(self, width=800, height=800):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        self.win = glfw.create_window(width, height, "Viewer", None, None)

        glfw.make_context_current(self.win)

        self.mouse = (0.0, 0.0)
        self.last_frame_time = glfw.get_time()

        # Free camera state (first-person style)
        self.camera_pos = np.array([0.0, 1.2, 6.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.move_speed = 5.0
        self.mouse_sensitivity = 0.12

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        print(
            "OpenGL",
            GL.glGetString(GL.GL_VERSION).decode() + ", GLSL",
            GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() + ", Renderer",
            GL.glGetString(GL.GL_RENDERER).decode(),
        )

        GL.glClearColor(0.5, 0.5, 0.5, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        self.drawables = []

    def add(self, *drawables):
        self.drawables.extend(drawables)

    def run(self):
        while not glfw.window_should_close(self.win):
            now = glfw.get_time()
            delta_time = max(0.0, now - self.last_frame_time)
            self.last_frame_time = now
            self._update_free_camera(delta_time)

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)
            view = T.lookat(self.camera_pos, self.camera_pos + self.camera_front, self.camera_up)
            projection = T.perspective(35, win_size[0] / max(1, win_size[1]), 0.1, 100.0)

            for drawable in self.drawables:
                drawable.draw(projection, view, None)

            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_F:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

    def on_mouse_move(self, win, xpos, ypos):
        # Hold left mouse button to look around.
        if not glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.mouse = (xpos, ypos)
            return

        old_x, old_y = self.mouse
        self.mouse = (xpos, ypos)

        x_offset = (xpos - old_x) * self.mouse_sensitivity
        y_offset = (old_y - ypos) * self.mouse_sensitivity

        self.yaw += x_offset
        self.pitch += y_offset
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self._update_camera_front()

    def on_scroll(self, win, _deltax, deltay):
        del win, _deltax
        self.move_speed = float(np.clip(self.move_speed + deltay * 0.3, 1.0, 20.0))

    def _update_camera_front(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        front = np.array(
            [
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad),
            ],
            dtype=np.float32,
        )
        self.camera_front = front / np.linalg.norm(front)

    def _update_free_camera(self, delta_time: float):
        velocity = self.move_speed * delta_time
        right = np.cross(self.camera_front, self.camera_up)
        right /= np.linalg.norm(right)

        if glfw.get_key(self.win, glfw.KEY_W) == glfw.PRESS:
            self.camera_pos += self.camera_front * velocity
        if glfw.get_key(self.win, glfw.KEY_S) == glfw.PRESS:
            self.camera_pos -= self.camera_front * velocity
        if glfw.get_key(self.win, glfw.KEY_A) == glfw.PRESS:
            self.camera_pos -= right * velocity
        if glfw.get_key(self.win, glfw.KEY_D) == glfw.PRESS:
            self.camera_pos += right * velocity
        if glfw.get_key(self.win, glfw.KEY_E) == glfw.PRESS:
            self.camera_pos += self.camera_up * velocity
        if glfw.get_key(self.win, glfw.KEY_C) == glfw.PRESS:
            self.camera_pos -= self.camera_up * velocity


def main():
    # viewer = Viewer()
    # base_dir = os.path.dirname(os.path.abspath(__file__))

    # # NOTE: Put `car.obj` in the same folder as this file: `BTL2/car.obj`
    # car = Mesh(os.path.join(base_dir, "assets/car.obj")).setup()
    # viewer.add(car)

    # viewer.run()
    
    
    
    
    ##### code dưới đây mô tả cơ bản quá trình sẽ render tuần tự 
    ##### từng chức năng dựa trên các assets có sẵn và chưa tự động hoá quá trình tạo synthetic dataset
    viewer = Viewer(width=800, height=800)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    car_mesh = Mesh(os.path.join(base_dir, "assets/car.obj")).setup()
    
    from entity import Scene, Entity
    scene = Scene()
    
    car1 = Entity(mesh=car_mesh, class_id=0, instance_color=(1.0, 0.0, 0.0))
    car1.set_transform(position=(0, 0, 0), scale=(1, 1, 1))
    
    car2 = Entity(mesh=car_mesh, class_id=0, instance_color=(0.0, 1.0, 0.0))
    car2.set_transform(positionn=(3.0, 0, -2.0), scale=(1, 1, 1))
    
    scene.add_entity(car1)
    scene.add_entity(car2)
    
    from renderers import RendererManager
    render_manager = RendererManager(scene)
    
    def custom_key_callback(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                render_manager.set_mode("RGB")
            elif key == glfw.KEY_2:
                render_manager.set_mode("DEPTH")
            elif key == glfw.KEY_3:
                render_manager.set_mode("MASK")
            elif key == glfw.KEY_4:
                render_manager.toggle_bbox_visualization()
            elif key == glfw.KEY_P:
                render_manager.export_current_frame()
        viewer.on_key(win, key, scancode, action, mods)
        
    glfw.set_key_callback(viewer.win, custom_key_callback)
    
    viewer.add(render_manager)
    viewer.run()


if __name__ == "__main__":
    glfw.init()
    main()
    glfw.terminate()

