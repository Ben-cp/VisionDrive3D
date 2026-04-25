import json
import math
import os
import pathlib
import random
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from entity import Entity, _rotation_matrix_xyz
from libs.shader import Shader
from libs.buffer import UManager
from mesh import Mesh


@dataclass
class ModelData:
    path: str
    mesh: Mesh
    aabb_min: np.ndarray
    aabb_max: np.ndarray


class _CarMeshProxy:
    def __init__(self, car: "Car"):
        self._car = car

    def compute_local_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._car._compute_car_local_aabb()

    def draw(self, projection: np.ndarray, view: np.ndarray, model: np.ndarray, shader):
        self._car._draw_with_shader(view, projection, shader, model_override=model)


class Car(Entity):
    """
    Car made from 5 separate .glb files with correct wheel offsets.
    Now works with Y-up engine coordinate system.
    """

    _WHEEL_FILE = {
        "fl": "wheel_fl.glb",
        "fr": "wheel_fr.glb",
        "bl": "wheel_bl.glb",
        "br": "wheel_br.glb",
    }

    def __init__(
        self,
        car_folder: str = "assets/car0",
        position: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        class_id: int = 0,
        instance_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        name: str = "car",
    ):
        self.body: Optional[ModelData] = None
        self.wheels: Dict[str, Dict] = {}
        self._body_root_offset = np.zeros(3, dtype=np.float32)
        self.wheel_spin_angle = {k: 0.0 for k in self._WHEEL_FILE}
        self.steering_angle = 0.0
        self.speed = 0.0
        self.target_speed = float(random.uniform(3.0, 5.0))
        self.wheel_radius = 0.35

        # --- Waypoint / pool state ---
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wp_idx: int = 0
        self.is_active: bool = False
        self.route_finished: bool = False
        self._yaw_lerp_factor: float = 5.0   # lerp speed for yaw smoothing
        self._steer_lerp_factor: float = 8.0 # lerp speed for steering smoothing
        self._lookahead_wps: int = 3         # how many waypoints ahead to look

        self._mesh_proxy = _CarMeshProxy(self)
        self._standalone_shader: Optional[Shader] = None

        super().__init__(
            name=name,
            mesh=self._mesh_proxy,
            class_id=class_id,
            instance_color=instance_color,
            is_dynamic=True,
            parent=None,
        )

        if position is not None:
            self.position[:] = np.asarray(position, dtype=np.float32)
        if rotation is not None:
            self.rotation[:] = np.asarray(rotation, dtype=np.float32)
        if scale is not None:
            self.scale[:] = np.asarray(scale, dtype=np.float32)

        self.load_models(car_folder)
        self._refresh_local_bounds()

    # ------------------------------------------------------------------
    # Waypoint-following update
    # ------------------------------------------------------------------
    def update(self, dt: float):
        dt = float(max(0.0, min(dt, 0.1)))

        if not self.is_active or not self.waypoints:
            return

        # Already consumed all waypoints?
        if self.current_wp_idx >= len(self.waypoints):
            self.route_finished = True
            return

        current_yaw = float(self.rotation[1])

        # --- Current target waypoint ---
        wx, wz = self.waypoints[self.current_wp_idx]
        px, pz = float(self.position[0]), float(self.position[2])
        dx, dz = wx - px, wz - pz
        dist = math.hypot(dx, dz)

        # Calculate forward vector to see if waypoint is behind the car
        fwd_x = math.sin(math.radians(current_yaw))
        fwd_z = math.cos(math.radians(current_yaw))
        dot_product = dx * fwd_x + dz * fwd_z

        # Advance waypoint if close enough OR if the car has passed it
        if dist < 2.5 or dot_product < 0.0:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.route_finished = True
                return
            wx, wz = self.waypoints[self.current_wp_idx]
            dx, dz = wx - px, wz - pz
            dist = math.hypot(dx, dz)

        # --- Lookahead: pick a point several waypoints ahead for smoother steering ---
        lookahead_idx = min(
            self.current_wp_idx + self._lookahead_wps,
            len(self.waypoints) - 1,
        )
        lx, lz = self.waypoints[lookahead_idx]
        ldx, ldz = lx - px, lz - pz
        ldist = math.hypot(ldx, ldz)

        # --- Smooth yaw toward lookahead waypoint ---
        if ldist > 0.01:
            target_yaw = math.degrees(math.atan2(ldx, ldz))  # atan2(sin, cos)
        else:
            target_yaw = float(self.rotation[1])

        current_yaw = float(self.rotation[1])

        # Shortest-arc delta
        delta_yaw = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0

        # Exponential lerp for smooth yaw rotation
        lerp_t = min(1.0, self._yaw_lerp_factor * dt)
        new_yaw = current_yaw + delta_yaw * lerp_t
        self.rotation[1] = float(new_yaw % 360.0)

        # --- Smooth steering angle (visual front-wheel offset) ---
        target_steering = float(max(-35.0, min(35.0, delta_yaw * 1.5)))
        steer_lerp = min(1.0, self._steer_lerp_factor * dt)
        self.steering_angle += (target_steering - self.steering_angle) * steer_lerp

        # --- Move forward ---
        distance = self.speed * dt
        forward = self._movement_forward_world()
        self.position += forward * distance

        # --- Spin wheels ---
        if self.wheel_radius > 1e-4:
            delta_angle_rad = distance / float(self.wheel_radius)
            delta_angle_deg = math.degrees(delta_angle_rad)
            for k in self.wheel_spin_angle:
                self.wheel_spin_angle[k] += delta_angle_deg

    # ------------------------------------------------------------------
    # Object pool helpers
    # ------------------------------------------------------------------
    def reset_for_pool(
        self,
        x: float,
        z: float,
        yaw: float,
        waypoints: List[Tuple[float, float]],
        target_speed: float = 5.0,
    ):
        """Teleport & reset this car for reuse from the object pool."""
        self.place_on_surface(x, z, 0.25)  # ground_y = 0.05
        self.rotation[1] = float(yaw)
        self.waypoints = list(waypoints)
        self.current_wp_idx = 0
        self.target_speed = float(target_speed)
        self.speed = float(target_speed * 0.5)  # start at half speed
        self.is_active = True
        self.route_finished = False
        self.steering_angle = 0.0
        for k in self.wheel_spin_angle:
            self.wheel_spin_angle[k] = 0.0

    def future_aabb_xz(self, lookahead: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return an AABB in XZ shifted `lookahead` meters ahead along the
        car's current forward vector.  Used for forward-collision radar.
        """
        fwd = self._movement_forward_world()
        offset_x = float(fwd[0]) * lookahead
        offset_z = float(fwd[2]) * lookahead

        mn, mx = self.world_aabb_xz()
        mn_shifted = mn + np.array([offset_x, offset_z], dtype=np.float32)
        mx_shifted = mx + np.array([offset_x, offset_z], dtype=np.float32)
        return mn_shifted, mx_shifted

    def load_models(self, folder: str):
        folder_abs = folder if os.path.isabs(folder) else os.path.join(os.path.dirname(__file__), folder)
        wheel_offsets = self._load_wheel_offsets(folder_abs)
        body_path = os.path.join(folder_abs, "body_car.glb")
        self.body = self._load_model_data(body_path)
        self._body_root_offset = self._read_body_root_offset(body_path)

        self.wheels = {}
        identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        unit_s = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        for key, filename in self._WHEEL_FILE.items():
            glb_path = os.path.join(folder_abs, filename)
            offset = np.asarray(wheel_offsets[key], dtype=np.float32)
            # Mesh loader bakes GLB node TRS into vertices; do not re-apply node0 TRS here.
            self.wheels[key] = {
                "model": self._load_model_data(glb_path),
                "hub_offset": offset.astype(np.float32).copy(),
                # Chỉ dịch tới hub (AABB); quay spin/steer dùng tâm riêng từng bánh trong _wheel_local_transform.
                "local_matrix": self._compose_trs(
                    translation=offset,
                    quaternion_xyzw=identity_q,
                    scale=unit_s,
                ),
            }

    def _load_wheel_offsets(self, folder_abs: str) -> Dict[str, np.ndarray]:
        offsets_path = os.path.join(folder_abs, "wheel_offsets.py")
        if not os.path.exists(offsets_path):
            raise FileNotFoundError(f"Missing wheel offsets file: {offsets_path}")

        namespace = {"np": np, "__builtins__": __builtins__}
        with open(offsets_path, "r", encoding="utf-8") as f:
            source = f.read()
        exec(compile(source, offsets_path, "exec"), namespace, namespace)

        raw_offsets = namespace.get("_WHEEL_OFFSETS")
        if not isinstance(raw_offsets, dict):
            raise ValueError(f"Invalid _WHEEL_OFFSETS in {offsets_path}: expected dict")

        wheel_offsets: Dict[str, np.ndarray] = {}
        for key in self._WHEEL_FILE:
            if key not in raw_offsets:
                raise ValueError(f"Missing wheel offset '{key}' in {offsets_path}")
            wheel_offsets[key] = np.asarray(raw_offsets[key], dtype=np.float32)
        return wheel_offsets

    def render(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        light_position: np.ndarray = np.array([8.0, 10.0, 6.0], dtype=np.float32),
    ):
        """
        Standalone rendering path (without RenderManager).
        In the existing app pipeline, RenderManager calls ent.mesh.draw(...)
        so this method is optional there.
        """
        if self._standalone_shader is None:
            self._standalone_shader = Shader(self.default_vertex_shader(), self.default_fragment_shader())

        shader = self._standalone_shader
        uma = UManager(shader)
        uma.upload_uniform_vector3fv(np.asarray(light_position, dtype=np.float32), "light_pos")
        uma.upload_uniform_vector3fv(np.array([1.0, 1.0, 1.0], dtype=np.float32), "light_color")
        uma.upload_uniform_vector3fv(self._extract_camera_pos(view_matrix), "view_pos")
        uma.upload_uniform_scalar1f(0.25, "ambient_strength")
        uma.upload_uniform_scalar1f(0.35, "specular_strength")
        uma.upload_uniform_scalar1f(64.0, "shininess")

        self._draw_with_shader(view_matrix, projection_matrix, shader, model_override=None)

    def move(self, delta):
        """Simple movement - deprecated, use waypoint system."""
        _ = delta

    def follow_trajectory(self, points, speed, dt):
        """Path following - deprecated, use waypoint system."""
        _ = (points, speed, dt)

    # ----------------------------
    # Draw internals
    # ----------------------------
    def _draw_with_shader(
        self,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        shader,
        model_override: Optional[np.ndarray] = None,
    ):
        model_car = model_override if model_override is not None else self.world_matrix()

        self.body.mesh.draw(projection_matrix, view_matrix, model_car, shader)

        for key in ("fl", "fr", "bl", "br"):
            wheel_info = self.wheels[key]
            local_wheel = self._wheel_local_transform(key)
            model_wheel = model_car @ local_wheel
            wheel_info["model"].mesh.draw(projection_matrix, view_matrix, model_wheel, shader)

    # ----------------------------
    # Wheel transforms
    # ----------------------------
    # def _wheel_local_transform(self, wheel_key: str) -> np.ndarray:
    #     """
    #     Biến đổi wheel trong hệ thân xe:
    #     - Spin (lăn): quanh trục qua **tâm mesh bánh** (gốc geometry), trong hệ thân xe.
    #     - Steer (trước): quay Ry quanh **trục Y đi qua hub** của đúng bánh đó (không phải gốc xe).
    #     - Cuối: đưa bánh tới hub_offset.
    #     Chuỗi: T(hub) @ Ry @ R_spin  ≡  (T(hub)@Ry@T(-hub)) @ T(hub) @ R_spin
    #     """
    #     hub = self.wheels[wheel_key]["hub_offset"]
    #     spin_deg = float(self.wheel_spin_angle.get(wheel_key, 0.0))
    #     axis_body = self._rolling_axis_body(wheel_key)
    #     spin = self._axis_angle_to_matrix4(axis_body, spin_deg)
    #     th = self._translation_matrix4(hub)
    #     tn  = self._translation_matrix4(-np.asarray(hub, dtype=np.float32))
    #     if wheel_key in ("fl", "fr"):
    #         steer_m = self._rotation_y_at_point(hub, float(self.steering_angle))
    #     else:
    #         steer_m = np.eye(4, dtype=np.float32)
    #     return steer_m @ th @ spin @ tn
    def _wheel_local_transform(self, wheel_key: str) -> np.ndarray:
        wheel_data = self.wheels[wheel_key]
        hub = wheel_data["hub_offset"]
        model_data = wheel_data["model"]

        # ← ĐÂY: tâm thực của mesh bánh trong body-space (sau khi bake GLB)
        mesh_center = ((model_data.aabb_min + model_data.aabb_max) * 0.5).astype(np.float32)

        spin_deg = float(self.wheel_spin_angle.get(wheel_key, 0.0))
        axis_body = self._rolling_axis_body(wheel_key)

        # Spin quanh mesh_center thay vì origin
        tc  = self._translation_matrix4(mesh_center)
        tnc = self._translation_matrix4(-mesh_center)
        spin_at_center = tc @ self._axis_angle_to_matrix4(axis_body, spin_deg) @ tnc

        # Steer (bánh trước) cũng quanh mesh_center
        if wheel_key in ("fl", "fr"):
            steer_m = self._rotation_y_at_point(mesh_center, float(self.steering_angle))
        else:
            steer_m = np.eye(4, dtype=np.float32)

        # hub_offset giữ nguyên vai trò fine-tuning nhỏ
        th = self._translation_matrix4(hub)
        return th @ steer_m @ spin_at_center

    @staticmethod
    def _translation_matrix4(t: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
        return m

    def _rotation_y_at_point(self, point: np.ndarray, angle_deg: float) -> np.ndarray:
        """Quay quanh trục song song trục dọc +Y đi qua điểm `point` trong hệ thân xe."""
        th = self._translation_matrix4(point)
        tn = self._translation_matrix4(-np.asarray(point, dtype=np.float32))
        ry = self._rotation_y(angle_deg)
        return th @ ry @ tn

    def _body_rotation_3x3(self) -> np.ndarray:
        return _rotation_matrix_xyz(self.rotation.astype(np.float32))[:3, :3].astype(np.float32)

    def _movement_forward_world(self) -> np.ndarray:
        """Vector tiến theo mặt đất (XZ), khớp spawn lane (yaw = rotation[1])."""
        yaw_rad = math.radians(float(self.rotation[1]))
        return np.array(
            [math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
            dtype=np.float32,
        )

    def _wheel_forward_world(self, wheel_key: str) -> np.ndarray:
        """
        Hướng lăn trên mặt phẳng ngang cho từng bánh: sau có body +Z;
        trước (fl/fr) thêm góc lái quanh Y trong hệ thân xe rồi đưa ra world.
        """
        if wheel_key in ("fl", "fr"):
            sr = math.radians(float(self.steering_angle))
            fwd_b = np.array([math.sin(sr), 0.0, math.cos(sr)], dtype=np.float64)
        else:
            fwd_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R = self._body_rotation_3x3().astype(np.float64)
        fwd_w = (R @ fwd_b).astype(np.float64)
        fwd_w[1] = 0.0
        n = float(np.linalg.norm(fwd_w))
        if n < 1e-8:
            return self._movement_forward_world()
        fwd_w /= n
        return fwd_w.astype(np.float32)

    def _rolling_axis_body(self, wheel_key: str) -> np.ndarray:
        """Trục bánh trong hệ thân xe: R^T * normalize(cross(fwd_w, up_w))."""
        fwd_w = self._wheel_forward_world(wheel_key)
        up_w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # axle_w = np.cross(fwd_w, up_w)
        axle_w = np.cross(up_w, fwd_w)
        n = float(np.linalg.norm(axle_w))
        if n < 1e-8:
            axle_b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        else:
            axle_w = (axle_w / n).astype(np.float32)
            R = self._body_rotation_3x3()
            axle_b = (R.T @ axle_w.reshape(3, 1)).reshape(3).astype(np.float32)
            nb = float(np.linalg.norm(axle_b))
            if nb < 1e-8:
                axle_b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            else:
                axle_b /= nb
        return axle_b

    @staticmethod
    def _axis_angle_to_matrix4(axis_unit: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rodrigues: quay angle_deg (độ) quanh trục đơn vị (x,y,z)."""
        x, y, z = [float(v) for v in np.asarray(axis_unit, dtype=np.float64).reshape(3)]
        n = math.hypot(math.hypot(x, y), z)
        if n < 1e-12:
            return np.eye(4, dtype=np.float32)
        x, y, z = x / n, y / n, z / n
        rad = math.radians(float(angle_deg))
        c = math.cos(rad)
        s = math.sin(rad)
        t = 1.0 - c
        R = np.eye(4, dtype=np.float32)
        R[0, 0] = t * x * x + c
        R[0, 1] = t * x * y - s * z
        R[0, 2] = t * x * z + s * y
        R[1, 0] = t * x * y + s * z
        R[1, 1] = t * y * y + c
        R[1, 2] = t * y * z - s * x
        R[2, 0] = t * x * z - s * y
        R[2, 1] = t * y * z + s * x
        R[2, 2] = t * z * z + c
        return R

    @staticmethod
    def _rotation_y(angle_deg: float) -> np.ndarray:
        return _rotation_matrix_xyz(np.array([0.0, angle_deg, 0.0], dtype=np.float32))

    # ----------------------------
    # Model loading / matrix helpers
    # ----------------------------
    def _load_model_data(self, glb_path: str) -> ModelData:
        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"Missing car part: {glb_path}")
        mesh = Mesh(glb_path).setup()
        aabb_min, aabb_max = mesh.compute_local_aabb()
        return ModelData(
            path=glb_path,
            mesh=mesh,
            aabb_min=aabb_min.astype(np.float32),
            aabb_max=aabb_max.astype(np.float32),
        )

    @staticmethod
    def _read_glb_json(glb_path: str) -> dict:
        data = pathlib.Path(glb_path).read_bytes()
        if len(data) < 20:
            raise ValueError(f"Invalid GLB file: {glb_path}")
        chunk0_len = struct.unpack_from("<I", data, 12)[0]
        return json.loads(data[20:20 + chunk0_len])

    @classmethod
    def _read_glb_node0_trs(cls, glb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read translation, rotation(xyzw), scale from first node of GLB JSON chunk.
        GLTF stores transforms in Y-up space, so no Blender->engine conversion is needed.
        """
        gltf = cls._read_glb_json(glb_path)
        nodes = gltf.get("nodes", [])
        node = nodes[0] if nodes else {}

        t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        return t, r, s

    @classmethod
    def _read_body_root_offset(cls, glb_path: str) -> np.ndarray:
        """
        Extract body root offset from Sketchfab_model if available.
        Kept for diagnostics/inspection; mesh loader already applies node transforms.
        """
        gltf = cls._read_glb_json(glb_path)
        for node in gltf.get("nodes", []):
            if node.get("name", "") == "Sketchfab_model":
                return np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _quat_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in q_xyzw]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )
        return mat

    @classmethod
    def _compose_trs(cls, translation: np.ndarray, quaternion_xyzw: np.ndarray, scale: np.ndarray) -> np.ndarray:
        t = np.eye(4, dtype=np.float32)
        t[:3, 3] = np.asarray(translation, dtype=np.float32)
        r = cls._quat_xyzw_to_matrix(np.asarray(quaternion_xyzw, dtype=np.float32))
        s = np.eye(4, dtype=np.float32)
        sc = np.asarray(scale, dtype=np.float32)
        s[0, 0], s[1, 1], s[2, 2] = sc[0], sc[1], sc[2]
        return t @ r @ s

    @staticmethod
    def _extract_camera_pos(view: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(view.astype(np.float32))
        return inv[:3, 3].astype(np.float32)

    # ----------------------------
    # Bounds integration for Entity APIs
    # ----------------------------
    def _compute_car_local_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        mins = []
        maxs = []
        if self.body is not None:
            mins.append(self.body.aabb_min)
            maxs.append(self.body.aabb_max)

        for key in ("fl", "fr", "bl", "br"):
            wheel = self.wheels.get(key)
            if wheel is None:
                continue
            model = wheel["model"]
            local_m = wheel["local_matrix"]
            wmin, wmax = self._transform_aabb(model.aabb_min, model.aabb_max, local_m)
            mins.append(wmin)
            maxs.append(wmax)

        if not mins:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero
        return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)

    @staticmethod
    def _transform_aabb(aabb_min: np.ndarray, aabb_max: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        corners = np.array(
            [
                [aabb_min[0], aabb_min[1], aabb_min[2], 1.0],
                [aabb_max[0], aabb_min[1], aabb_min[2], 1.0],
                [aabb_min[0], aabb_max[1], aabb_min[2], 1.0],
                [aabb_max[0], aabb_max[1], aabb_min[2], 1.0],
                [aabb_min[0], aabb_min[1], aabb_max[2], 1.0],
                [aabb_max[0], aabb_min[1], aabb_max[2], 1.0],
                [aabb_min[0], aabb_max[1], aabb_max[2], 1.0],
                [aabb_max[0], aabb_max[1], aabb_max[2], 1.0],
            ],
            dtype=np.float32,
        )
        tr = (matrix @ corners.T).T[:, :3]
        return np.min(tr, axis=0), np.max(tr, axis=0)

    def _refresh_local_bounds(self):
        mn, mx = self._compute_car_local_aabb()
        self.local_aabb_min = mn.astype(np.float32)
        self.local_aabb_max = mx.astype(np.float32)
        self.local_size = (self.local_aabb_max - self.local_aabb_min).astype(np.float32)
        self.bottom_y_offset = float(-self.local_aabb_min[1])   # Y-up → keep [1]
