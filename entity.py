# entity.py
import math
from typing import List, Optional, Tuple, Iterable

import glfw
import numpy as np

from libs import transform as T


def _rotation_matrix_xyz(deg_xyz: np.ndarray) -> np.ndarray:
    """
    Build Euler rotation matrix with order Rx * Ry * Rz (degrees input).
    """
    rx, ry, rz = np.radians(deg_xyz.astype(np.float32))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([
        [1, 0, 0, 0],
        [0, cx, -sx, 0],
        [0, sx, cx, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    ry_m = np.array([
        [cy, 0, sy, 0],
        [0, 1, 0, 0],
        [-sy, 0, cy, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rz_m = np.array([
        [cz, -sz, 0, 0],
        [sz, cz, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    return rx_m @ ry_m @ rz_m


def _sat_aabb_overlap_xz(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    """
    SAT for axis-aligned boxes in 2D XZ plane:
    Non-overlap exists if there is a separating axis on X or Z.
    """
    separated_x = a_max[0] <= b_min[0] or a_min[0] >= b_max[0]
    separated_z = a_max[1] <= b_min[1] or a_min[1] >= b_max[1]
    return not (separated_x or separated_z)


class Node:
    """
    Scene graph node with local TRS and hierarchical world matrix.
    """

    def __init__(self, name: str = "node", parent: Optional["Node"] = None):
        self.name = name
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # euler deg
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child: "Node"):
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Node"):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def local_matrix(self) -> np.ndarray:
        t = T.translate(self.position)
        r = _rotation_matrix_xyz(self.rotation)
        s = T.scale(self.scale)
        return t @ r @ s

    def world_matrix(self) -> np.ndarray:
        local = self.local_matrix()
        if self.parent is None:
            return local
        return self.parent.world_matrix() @ local


class Entity(Node):
    """
    Renderable object wrapping Mesh + annotation metadata.
    """

    def __init__(
        self,
        name: str,
        mesh,
        class_id: int,
        instance_color: Tuple[float, float, float],
        is_dynamic: bool = True,
        parent: Optional[Node] = None,
    ):
        super().__init__(name=name, parent=parent)
        self.mesh = mesh
        self.class_id = int(class_id)
        self.instance_color = np.asarray(instance_color, dtype=np.float32)
        self.is_dynamic = bool(is_dynamic)

        local_min, local_max = self.mesh.compute_local_aabb()
        self.local_aabb_min = local_min.astype(np.float32)
        self.local_aabb_max = local_max.astype(np.float32)
        self.local_size = (self.local_aabb_max - self.local_aabb_min).astype(np.float32)

        # Distance from origin to lowest local Y, used to place object on ground.
        self.bottom_y_offset = float(-self.local_aabb_min[1])

    def place_on_surface(self, x: float, z: float, ground_y: float):
        """
        Ensure mesh bottom touches road plane:
        y = ground_y + bottom_offset * scale_y
        """
        self.position[0] = float(x)
        self.position[2] = float(z)
        self.position[1] = float(ground_y) + self.bottom_y_offset * float(self.scale[1])

    def world_aabb_xz(self, padding: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate world AABB on XZ (axis-aligned) for SAT overlap checks.
        """
        half_x = 0.5 * float(self.local_size[0] * self.scale[0]) + padding
        half_z = 0.5 * float(self.local_size[2] * self.scale[2]) + padding
        center_x = float(self.position[0])
        center_z = float(self.position[2])

        mn = np.array([center_x - half_x, center_z - half_z], dtype=np.float32)
        mx = np.array([center_x + half_x, center_z + half_z], dtype=np.float32)
        return mn, mx


class Scene:
    """
    Holds entities and domain randomization logic.
    """

    def __init__(self):
        self.entities: List[Entity] = []

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def get_dynamic_entities(self) -> List[Entity]:
        return [e for e in self.entities if e.is_dynamic and e.class_id >= 0]

    def get_static_entities(self) -> List[Entity]:
        return [e for e in self.entities if not e.is_dynamic]

    def raycast_ground_height(self, x: float, z: float, default_y: float = 0.0) -> float:
        """
        Lightweight raycast approximation:
        project a vertical ray and find the top Y of static entities whose XZ AABB contains (x, z).
        """
        hit_y = default_y
        found = False

        for ent in self.get_static_entities():
            local_min = ent.local_aabb_min * ent.scale
            local_max = ent.local_aabb_max * ent.scale
            center = ent.position

            x_min = float(center[0] + local_min[0])
            x_max = float(center[0] + local_max[0])
            z_min = float(center[2] + local_min[2])
            z_max = float(center[2] + local_max[2])

            if x_min <= x <= x_max and z_min <= z <= z_max:
                y_top = float(center[1] + local_max[1])
                if not found or y_top > hit_y:
                    hit_y = y_top
                    found = True

        return hit_y

    def randomize_domain(
        self,
        ground_y: float = 0.0,
        x_range: Tuple[float, float] = (-8.0, 8.0),
        z_range: Tuple[float, float] = (-35.0, 8.0),
        scale_range: Tuple[float, float] = (0.85, 1.20),
        yaw_range: Tuple[float, float] = (0.0, 360.0),
        sat_padding: float = 0.15,
        max_retry: int = 80,
    ):
        """
        Domain randomization for dynamic vehicles.
        Use SAT overlap test on 2D AABB in XZ plane to avoid spawning collisions.
        """
        placed: List[Tuple[np.ndarray, np.ndarray]] = []  # (aabb_min_xz, aabb_max_xz)

        for ent in self.get_dynamic_entities():
            placed_ok = False

            # Random object variation
            uniform_scale = float(np.random.uniform(scale_range[0], scale_range[1]))
            ent.scale[:] = np.array([uniform_scale, uniform_scale, uniform_scale], dtype=np.float32)
            ent.rotation[1] = float(np.random.uniform(yaw_range[0], yaw_range[1]))

            for _ in range(max_retry):
                tx = float(np.random.uniform(x_range[0], x_range[1]))
                tz = float(np.random.uniform(z_range[0], z_range[1]))
                gy = self.raycast_ground_height(tx, tz, default_y=ground_y)
                ent.place_on_surface(tx, tz, gy)

                mn, mx = ent.world_aabb_xz(padding=sat_padding)

                overlap = False
                for pmn, pmx in placed:
                    if _sat_aabb_overlap_xz(mn, mx, pmn, pmx):
                        overlap = True
                        break

                if not overlap:
                    placed.append((mn, mx))
                    placed_ok = True
                    break

            if not placed_ok:
                # Keep object but move out of camera frustum if no slot found.
                ent.position[:] = np.array([9999.0, -9999.0, 9999.0], dtype=np.float32)


class Camera(Node):
    """
    Fly camera (WASD + mouse look) + ground-follow with raycast + lerp.
    """

    def __init__(
        self,
        fov_deg: float = 60.0,
        near: float = 0.1,
        far: float = 150.0,
    ):
        super().__init__(name="camera")
        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)

        self.yaw = -90.0
        self.pitch = 0.0

        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.move_speed = 12.0
        self.mouse_sensitivity = 0.10

        self.eye_height = 1.7
        self.ground_follow_lerp = 9.0

    def _update_front(self):
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        front = np.array(
            [
                math.cos(yaw_r) * math.cos(pitch_r),
                math.sin(pitch_r),
                math.sin(yaw_r) * math.cos(pitch_r),
            ],
            dtype=np.float32,
        )
        n = np.linalg.norm(front)
        if n > 1e-8:
            self.front = front / n

    def process_mouse_delta(self, dx: float, dy: float):
        self.yaw += float(dx) * self.mouse_sensitivity
        self.pitch -= float(dy) * self.mouse_sensitivity
        self.pitch = float(np.clip(self.pitch, -89.0, 89.0))
        self._update_front()

    def process_keyboard(self, window, dt: float):
        v = self.move_speed * float(dt)
        right = np.cross(self.front, self.up_ref)
        rnorm = np.linalg.norm(right)
        if rnorm > 1e-8:
            right = right / rnorm

        # WASD in XZ plane, vertical handled by ground alignment.
        forward_flat = np.array([self.front[0], 0.0, self.front[2]], dtype=np.float32)
        fnorm = np.linalg.norm(forward_flat)
        if fnorm > 1e-8:
            forward_flat /= fnorm

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += forward_flat * v
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= forward_flat * v
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= right * v
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += right * v

    def update_ground_alignment(self, scene: Scene, dt: float):
        """
        Raycast + lerp:
        - sample ground height under current camera XZ
        - lerp current Y toward (ground + eye_height)
        """
        ground_y = scene.raycast_ground_height(
            float(self.position[0]),
            float(self.position[2]),
            default_y=0.0,
        )
        target_y = ground_y + self.eye_height
        alpha = float(np.clip(dt * self.ground_follow_lerp, 0.0, 1.0))
        self.position[1] = float(self.position[1] + alpha * (target_y - self.position[1]))

    def view_matrix(self) -> np.ndarray:
        eye = self.position
        target = self.position + self.front
        return T.lookat(eye, target, self.up_ref)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return T.perspective(self.fov_deg, max(1e-6, aspect), self.near, self.far)