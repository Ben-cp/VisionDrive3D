# traffic.py
"""
Waypoint-based traffic routing system with Bezier turning,
object pooling, lane-following, and intersection control.
"""
import math
import random
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Intersection and road geometry constants
# ---------------------------------------------------------------------------
INTERSECTION = {
    "center_x": -1.85905,
    "center_z": -2.155,
    "x_min": -11.1,
    "x_max": 14.8,
    "z_min": -10.8,
    "z_max": 15.1,
}

ROAD_END_TOP_Z = 48.4
ROAD_END_BOTTOM_Z = -47.8
ROAD_END_LEFT_X = -47.1
ROAD_END_RIGHT_X = 46.9

# Right-hand traffic: drive on the right side of the road.
_VERT_ROAD_CX = INTERSECTION["center_x"]
_HORIZ_ROAD_CZ = INTERSECTION["center_z"]
_LANE_OFFSET_X = 4.6
_LANE_OFFSET_Z = 4.6

LANE_SOUTHBOUND_X = _VERT_ROAD_CX + _LANE_OFFSET_X
LANE_NORTHBOUND_X = _VERT_ROAD_CX - _LANE_OFFSET_X
LANE_EASTBOUND_Z = _HORIZ_ROAD_CZ + _LANE_OFFSET_Z
LANE_WESTBOUND_Z = _HORIZ_ROAD_CZ - _LANE_OFFSET_Z

GROUND_Y = 0.05

SPAWN_POINTS = [
    {
        "label": "top",
        "x": LANE_SOUTHBOUND_X,
        "z": ROAD_END_TOP_Z,
        "yaw": 180.0,
        "dir": (0.0, -1.0),
    },
    {
        "label": "bottom",
        "x": LANE_NORTHBOUND_X,
        "z": ROAD_END_BOTTOM_Z,
        "yaw": 0.0,
        "dir": (0.0, 1.0),
    },
    {
        "label": "left",
        "x": ROAD_END_LEFT_X,
        "z": LANE_EASTBOUND_Z,
        "yaw": 90.0,
        "dir": (1.0, 0.0),
    },
    {
        "label": "right",
        "x": ROAD_END_RIGHT_X,
        "z": LANE_WESTBOUND_Z,
        "yaw": -90.0,
        "dir": (-1.0, 0.0),
    },
]

EXIT_STRAIGHT = {
    "top": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z},
    "bottom": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z},
    "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z},
    "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z},
}

_TURN_MAP = {
    "top": {
        "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
        "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
    },
    "bottom": {
        "right": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
        "left": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
    },
    "left": {
        "right": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
        "left": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
    },
    "right": {
        "right": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
        "left": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
    },
}


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------
def _bezier_quadratic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    num_points: int = 12,
) -> List[Tuple[float, float]]:
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        z = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, z))
    return pts


def _bezier_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    num_points: int = 16,
) -> List[Tuple[float, float]]:
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = (
            u ** 3 * p0[0]
            + 3 * u ** 2 * t * p1[0]
            + 3 * u * t ** 2 * p2[0]
            + t ** 3 * p3[0]
        )
        z = (
            u ** 3 * p0[1]
            + 3 * u ** 2 * t * p1[1]
            + 3 * u * t ** 2 * p2[1]
            + t ** 3 * p3[1]
        )
        pts.append((x, z))
    return pts


# ---------------------------------------------------------------------------
# Intersection helpers
# ---------------------------------------------------------------------------
def is_inside_intersection(x: float, z: float) -> bool:
    return (
        INTERSECTION["x_min"] <= x <= INTERSECTION["x_max"]
        and INTERSECTION["z_min"] <= z <= INTERSECTION["z_max"]
    )


def _approach_margin() -> float:
    return 5.0


def is_approaching_intersection(x: float, z: float, dx: float, dz: float) -> bool:
    if is_inside_intersection(x, z):
        return False
    margin = _approach_margin()
    fx, fz = x + dx * margin, z + dz * margin
    return is_inside_intersection(fx, fz)


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------
def _intersection_entry_point(
    spawn_label: str,
    spawn_x: float,
    spawn_z: float,
) -> Tuple[float, float]:
    if spawn_label == "top":
        return (spawn_x, INTERSECTION["z_max"])
    if spawn_label == "bottom":
        return (spawn_x, INTERSECTION["z_min"])
    if spawn_label == "left":
        return (INTERSECTION["x_min"], spawn_z)
    if spawn_label == "right":
        return (INTERSECTION["x_max"], spawn_z)
    return (spawn_x, spawn_z)


def _intersection_exit_point(
    exit_label: str,
    exit_x: float,
    exit_z: float,
) -> Tuple[float, float]:
    if exit_label == "bottom":
        return (exit_x, INTERSECTION["z_min"])
    if exit_label == "top":
        return (exit_x, INTERSECTION["z_max"])
    if exit_label == "right":
        return (INTERSECTION["x_max"], exit_z)
    if exit_label == "left":
        return (INTERSECTION["x_min"], exit_z)
    return (exit_x, exit_z)


def generate_route(spawn_idx: int, maneuver: str = "straight") -> List[Tuple[float, float]]:
    """
    Generate a list of (x, z) waypoints for a car spawning at `spawn_idx`.
    """
    sp = SPAWN_POINTS[spawn_idx]
    label = sp["label"]
    sx, sz = sp["x"], sp["z"]

    waypoints: List[Tuple[float, float]] = []

    entry = _intersection_entry_point(label, sx, sz)
    num_approach = 3
    for i in range(1, num_approach + 1):
        t = i / (num_approach + 1)
        wx = sx + t * (entry[0] - sx)
        wz = sz + t * (entry[1] - sz)
        waypoints.append((wx, wz))
    waypoints.append(entry)

    if maneuver == "straight":
        exit_cfg = EXIT_STRAIGHT[label]
        ex, ez = exit_cfg["x"], exit_cfg["z"]
        exit_labels = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left",
        }
        exit_lbl = exit_labels[label]
        exit_pt = _intersection_exit_point(exit_lbl, ex, ez)

        if label in ("top", "bottom"):
            cross_zs = sorted(
                [
                    z
                    for z in (LANE_WESTBOUND_Z, LANE_EASTBOUND_Z)
                    if INTERSECTION["z_min"] < z < INTERSECTION["z_max"]
                ],
                reverse=(label == "top"),
            )
            for cz in cross_zs:
                waypoints.append((entry[0], cz))
        else:
            cross_xs = sorted(
                [
                    x
                    for x in (LANE_NORTHBOUND_X, LANE_SOUTHBOUND_X)
                    if INTERSECTION["x_min"] < x < INTERSECTION["x_max"]
                ],
                reverse=(label == "right"),
            )
            for cx in cross_xs:
                waypoints.append((cx, entry[1]))

        waypoints.append(exit_pt)

        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (ex - exit_pt[0])
            wz = exit_pt[1] + t * (ez - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((ex, ez))

    elif maneuver in ("right", "left"):
        turn_cfg = _TURN_MAP[label][maneuver]
        exit_road_x, exit_road_z = turn_cfg["x"], turn_cfg["z"]
        exit_lbl = turn_cfg["label"]
        exit_pt = _intersection_exit_point(exit_lbl, exit_road_x, exit_road_z)

        if label in ("top", "bottom"):
            corner = (entry[0], exit_pt[1])
        else:
            corner = (exit_pt[0], entry[1])

        if maneuver == "right":
            curve_pts = _bezier_quadratic(entry, corner, exit_pt, num_points=12)
        else:
            cx, cz = INTERSECTION["center_x"], INTERSECTION["center_z"]
            ctrl1 = (
                entry[0] + 0.35 * (corner[0] - entry[0]) + 0.1 * (cx - entry[0]),
                entry[1] + 0.35 * (corner[1] - entry[1]) + 0.1 * (cz - entry[1]),
            )
            ctrl2 = (
                exit_pt[0] + 0.35 * (corner[0] - exit_pt[0]) + 0.1 * (cx - exit_pt[0]),
                exit_pt[1] + 0.35 * (corner[1] - exit_pt[1]) + 0.1 * (cz - exit_pt[1]),
            )
            curve_pts = _bezier_cubic(entry, ctrl1, ctrl2, exit_pt, num_points=14)

        waypoints.extend(curve_pts[1:])

        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (exit_road_x - exit_pt[0])
            wz = exit_pt[1] + t * (exit_road_z - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((exit_road_x, exit_road_z))

    return waypoints


def random_maneuver() -> str:
    r = random.random()
    if r < 0.50:
        return "straight"
    if r < 0.75:
        return "right"
    return "left"


# ---------------------------------------------------------------------------
# Traffic manager
# ---------------------------------------------------------------------------
_BRAKE_DECEL = 12.0
_ACCEL = 8.0
_MIN_SPEED = 0.0

_FOLLOW_REACTION_TIME = 0.9
_FOLLOW_MIN_GAP = 2.5
_FOLLOW_GAIN = 1.35
_LANE_MATCH_TOL = 2.25
_LANE_EXTRA_WIDTH = 1.0

_STAGGER_DISTANCE = 3.0
_SPAWN_CLEAR_RADIUS = 8.0
_SPAWN_LANE_CLEARANCE = 18.0

_INTERSECTION_APPROACH_MARGIN = 8.0
_STOP_QUEUE_DISTANCE = 12.0
_STOP_LINE_BUFFER = 1.5

_APPROACH_ORDER = {"top": 0, "right": 1, "bottom": 2, "left": 3}
_MANEUVER_PRIORITY = {"right": 0, "straight": 1, "left": 2}
_SPAWN_LANE_IDS = {
    "top": "southbound",
    "bottom": "northbound",
    "left": "eastbound",
    "right": "westbound",
}


class TrafficManager:
    """
    Manages the car pool: assigns routes, handles respawning,
    and runs traffic control each frame.
    """

    def __init__(self):
        self.cars: list = []
        self._car_lookup: dict = {}
        self._spawn_timers: dict = {}
        self._pending_respawns: dict = {}
        self._car_entry_labels: dict = {}
        self._car_maneuvers: dict = {}
        self._wait_started_at: dict = {}
        self._intersection_owner: Optional[str] = None
        self._intersection_owner_entered: bool = False
        self._sim_time: float = 0.0

    def register_cars(self, cars: list):
        """Register the car pool and create a clean traffic state."""
        self.cars = list(cars)
        self._car_lookup = {car.name: car for car in self.cars}
        self._spawn_timers.clear()
        self._pending_respawns.clear()
        self._car_entry_labels.clear()
        self._car_maneuvers.clear()
        self._wait_started_at.clear()
        self._intersection_owner = None
        self._intersection_owner_entered = False
        self._sim_time = 0.0

        n_spawns = len(SPAWN_POINTS)
        for idx, car in enumerate(self.cars):
            spawn_idx = idx % n_spawns
            self._spawn_car(car, spawn_idx=spawn_idx, stagger_offset=idx)

    def _spawn_car(self, car, spawn_idx: int = -1, stagger_offset: int = 0):
        """
        Teleport a car to a spawn point and assign a new route.
        If no spawn point is clear, leave it in the pending respawn queue.
        """
        if spawn_idx < 0:
            spawn_idx = self._pick_clear_spawn()
        if spawn_idx is None:
            self._queue_for_respawn(car)
            return

        sp = SPAWN_POINTS[spawn_idx]
        maneuver = random_maneuver()
        waypoints = generate_route(spawn_idx, maneuver)
        target_speed = random.uniform(3.0, 5.0)

        dir_x, dir_z = sp["dir"]
        offset = _STAGGER_DISTANCE * stagger_offset + random.uniform(0.0, 4.0)
        spawn_x = sp["x"] - dir_x * offset
        spawn_z = sp["z"] - dir_z * offset

        car.reset_for_pool(
            x=spawn_x,
            z=spawn_z,
            yaw=sp["yaw"],
            waypoints=waypoints,
            target_speed=target_speed,
        )
        self._pending_respawns.pop(car.name, None)
        self._spawn_timers[car.name] = 0.0
        self._wait_started_at.pop(car.name, None)
        self._car_entry_labels[car.name] = sp["label"]
        self._car_maneuvers[car.name] = maneuver

    def _queue_for_respawn(self, car, preferred_spawn_idx: Optional[int] = None):
        car.is_active = False
        car.route_finished = False
        car.speed = 0.0
        self._pending_respawns[car.name] = preferred_spawn_idx
        self._wait_started_at.pop(car.name, None)
        if car.name == self._intersection_owner:
            self._intersection_owner = None
            self._intersection_owner_entered = False

    def _pick_clear_spawn(self) -> Optional[int]:
        indices = list(range(len(SPAWN_POINTS)))
        random.shuffle(indices)
        for idx in indices:
            if self._is_spawn_point_clear(idx):
                return idx
        return None

    def _is_spawn_point_clear(self, spawn_idx: int) -> bool:
        sp = SPAWN_POINTS[spawn_idx]
        sx, sz = sp["x"], sp["z"]
        dir_x, dir_z = sp["dir"]
        lane_id = _SPAWN_LANE_IDS[sp["label"]]

        for car in self.cars:
            if not car.is_active or car.route_finished:
                continue

            cx = float(car.position[0])
            cz = float(car.position[2])
            if math.hypot(cx - sx, cz - sz) < _SPAWN_CLEAR_RADIUS:
                return False

            if self._lane_id_for_car(car) != lane_id:
                continue

            dx = cx - sx
            dz = cz - sz
            longitudinal = dx * dir_x + dz * dir_z
            lateral = abs(dx * (-dir_z) + dz * dir_x)
            if -2.0 <= longitudinal <= _SPAWN_LANE_CLEARANCE and lateral <= _LANE_MATCH_TOL:
                return False
        return True

    def _process_pending_respawns(self):
        for car_name, preferred_idx in list(self._pending_respawns.items()):
            car = self._car_lookup.get(car_name)
            if car is None:
                self._pending_respawns.pop(car_name, None)
                continue

            spawn_idx = preferred_idx
            if spawn_idx is not None and not self._is_spawn_point_clear(spawn_idx):
                spawn_idx = None
            if spawn_idx is None:
                spawn_idx = self._pick_clear_spawn()
            if spawn_idx is None:
                continue
            self._spawn_car(car, spawn_idx=spawn_idx, stagger_offset=0)

    @staticmethod
    def _apply_speed_toward(car, desired_speed: float, dt: float):
        desired_speed = float(max(_MIN_SPEED, min(car.target_speed, desired_speed)))
        if car.speed > desired_speed:
            car.speed = max(desired_speed, car.speed - _BRAKE_DECEL * dt)
        else:
            car.speed = min(desired_speed, car.speed + _ACCEL * dt)

    @staticmethod
    def _car_pos_xz(car) -> Tuple[float, float]:
        return float(car.position[0]), float(car.position[2])

    @staticmethod
    def _distance_to_stop_line_for_label(label: str, x: float, z: float) -> float:
        if label == "top":
            return max(0.0, z - INTERSECTION["z_max"])
        if label == "bottom":
            return max(0.0, INTERSECTION["z_min"] - z)
        if label == "left":
            return max(0.0, INTERSECTION["x_min"] - x)
        if label == "right":
            return max(0.0, x - INTERSECTION["x_max"])
        return float("inf")

    def _distance_to_stop_line(self, car) -> float:
        x, z = self._car_pos_xz(car)
        return self._distance_to_stop_line_for_label(
            self._car_entry_labels.get(car.name, ""),
            x,
            z,
        )

    def _lane_id_for_car(self, car) -> Optional[str]:
        x, z = self._car_pos_xz(car)
        fwd = car._movement_forward_world()
        fx = float(fwd[0])
        fz = float(fwd[2])

        if abs(fz) >= abs(fx):
            if fz > 0.4 and abs(x - LANE_NORTHBOUND_X) <= _LANE_MATCH_TOL:
                return "northbound"
            if fz < -0.4 and abs(x - LANE_SOUTHBOUND_X) <= _LANE_MATCH_TOL:
                return "southbound"
        else:
            if fx > 0.4 and abs(z - LANE_EASTBOUND_Z) <= _LANE_MATCH_TOL:
                return "eastbound"
            if fx < -0.4 and abs(z - LANE_WESTBOUND_Z) <= _LANE_MATCH_TOL:
                return "westbound"
        return None

    @staticmethod
    def _car_half_extent_for_lane(car, lane_id: str) -> Tuple[float, float]:
        vertical = lane_id in ("northbound", "southbound")
        half_length = 0.5 * float(
            (car.local_size[2] if vertical else car.local_size[0])
            * (car.scale[2] if vertical else car.scale[0])
        )
        half_width = 0.5 * float(
            (car.local_size[0] if vertical else car.local_size[2])
            * (car.scale[0] if vertical else car.scale[2])
        )
        return half_length, half_width

    def _find_lead_car(self, car_a, active_cars: list):
        lane_id = self._lane_id_for_car(car_a)
        if lane_id is None:
            return None, None

        ax, az = self._car_pos_xz(car_a)
        if is_inside_intersection(ax, az):
            return None, None

        fwd_a = car_a._movement_forward_world()
        fx = float(fwd_a[0])
        fz = float(fwd_a[2])
        perp_x = -fz
        perp_z = fx
        half_len_a, half_width_a = self._car_half_extent_for_lane(car_a, lane_id)

        best_car = None
        best_gap = float("inf")
        for car_b in active_cars:
            if car_b is car_a or self._lane_id_for_car(car_b) != lane_id:
                continue

            bx, bz = self._car_pos_xz(car_b)
            dx = bx - ax
            dz = bz - az
            longitudinal = dx * fx + dz * fz
            if longitudinal <= 0.0:
                continue

            _, half_width_b = self._car_half_extent_for_lane(car_b, lane_id)
            lateral = abs(dx * perp_x + dz * perp_z)
            if lateral > half_width_a + half_width_b + _LANE_EXTRA_WIDTH:
                continue

            half_len_b, _ = self._car_half_extent_for_lane(car_b, lane_id)
            gap = longitudinal - half_len_a - half_len_b
            if gap < best_gap:
                best_gap = gap
                best_car = car_b

        if best_car is None:
            return None, None
        return best_car, max(0.0, best_gap)

    @staticmethod
    def _following_safe_gap(speed: float, lead_speed: float) -> float:
        closing_speed = max(0.0, speed - lead_speed)
        braking_margin = (closing_speed * closing_speed) / max(1e-3, 2.0 * _BRAKE_DECEL)
        return _FOLLOW_MIN_GAP + _FOLLOW_REACTION_TIME * speed + braking_margin

    def _car_following_speed_cap(self, car, active_cars: list) -> Optional[float]:
        lead_car, gap = self._find_lead_car(car, active_cars)
        if lead_car is None or gap is None:
            return None

        lead_speed = float(max(0.0, lead_car.speed))
        safe_gap = self._following_safe_gap(float(car.speed), lead_speed)
        speed_cap = lead_speed + (gap - safe_gap) * _FOLLOW_GAIN
        return max(_MIN_SPEED, min(car.target_speed, speed_cap))

    def _is_waiting_for_intersection(self, car) -> bool:
        x, z = self._car_pos_xz(car)
        if is_inside_intersection(x, z):
            return False

        label = self._car_entry_labels.get(car.name)
        if not label:
            return False

        fwd = car._movement_forward_world()
        approaching = is_approaching_intersection(x, z, float(fwd[0]), float(fwd[2]))
        return approaching and self._distance_to_stop_line(car) <= _STOP_QUEUE_DISTANCE

    def _select_waiting_heads(self, active_cars: list) -> dict:
        heads = {}
        for car in active_cars:
            if not self._is_waiting_for_intersection(car):
                self._wait_started_at.pop(car.name, None)
                continue

            self._wait_started_at.setdefault(car.name, self._sim_time)
            label = self._car_entry_labels.get(car.name)
            stop_dist = self._distance_to_stop_line(car)
            current = heads.get(label)
            if current is None:
                heads[label] = car
                continue

            current_dist = self._distance_to_stop_line(current)
            if stop_dist < current_dist - 1e-3:
                heads[label] = car
            elif abs(stop_dist - current_dist) <= 1e-3:
                if self._wait_started_at[car.name] < self._wait_started_at[current.name]:
                    heads[label] = car
        return heads

    def _refresh_intersection_owner(self, active_cars: list):
        inside_cars = [
            car
            for car in active_cars
            if is_inside_intersection(float(car.position[0]), float(car.position[2]))
        ]

        if self._intersection_owner is not None:
            owner = self._car_lookup.get(self._intersection_owner)
            if owner is None or not owner.is_active or owner.route_finished:
                self._intersection_owner = None
                self._intersection_owner_entered = False
            else:
                owner_inside = is_inside_intersection(
                    float(owner.position[0]),
                    float(owner.position[2]),
                )
                if owner_inside:
                    self._intersection_owner_entered = True
                elif self._intersection_owner_entered:
                    self._intersection_owner = None
                    self._intersection_owner_entered = False

        if self._intersection_owner is None and inside_cars:
            chosen = min(
                inside_cars,
                key=lambda car: self._distance_to_stop_line_for_label(
                    self._car_entry_labels.get(car.name, ""),
                    float(car.position[0]),
                    float(car.position[2]),
                ),
            )
            self._intersection_owner = chosen.name
            self._intersection_owner_entered = True
            return

        if self._intersection_owner is not None:
            return

        heads = self._select_waiting_heads(active_cars)
        if not heads:
            return

        winner = min(
            heads.values(),
            key=lambda car: (
                self._wait_started_at.get(car.name, self._sim_time),
                _MANEUVER_PRIORITY.get(self._car_maneuvers.get(car.name, "straight"), 1),
                _APPROACH_ORDER.get(self._car_entry_labels.get(car.name, ""), 99),
                car.name,
            ),
        )
        self._intersection_owner = winner.name
        self._intersection_owner_entered = False

    def _intersection_speed_cap(self, car) -> Optional[float]:
        x, z = self._car_pos_xz(car)
        if is_inside_intersection(x, z):
            return None
        if not self._is_waiting_for_intersection(car):
            return None
        if self._intersection_owner == car.name:
            return None

        stop_dist = self._distance_to_stop_line(car)
        brake_distance = max(0.0, stop_dist - _STOP_LINE_BUFFER)
        if brake_distance <= 0.0:
            return 0.0
        return math.sqrt(max(0.0, 2.0 * _BRAKE_DECEL * brake_distance))

    def update(self, dt: float):
        """
        Called every frame before scene.update().
        1. Queue finished cars for respawn.
        2. Respawn only when a spawn lane is clear.
        3. Update following and intersection speed limits.
        """
        dt = float(max(0.0, min(dt, 0.1)))
        self._sim_time += dt

        for car in self.cars:
            if car.name in self._spawn_timers:
                self._spawn_timers[car.name] += dt

        for car in self.cars:
            if car.is_active and car.route_finished:
                self._queue_for_respawn(car)

        self._process_pending_respawns()

        active_cars = [c for c in self.cars if c.is_active and not c.route_finished]
        self._refresh_intersection_owner(active_cars)

        for car in active_cars:
            desired_speed = float(car.target_speed)

            following_cap = self._car_following_speed_cap(car, active_cars)
            if following_cap is not None:
                desired_speed = min(desired_speed, following_cap)

            intersection_cap = self._intersection_speed_cap(car)
            if intersection_cap is not None:
                desired_speed = min(desired_speed, intersection_cap)

            self._apply_speed_toward(car, desired_speed, dt)
