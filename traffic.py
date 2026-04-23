# traffic.py
"""
Waypoint-based traffic routing system with Bezier curve turning,
SAT collision avoidance, and object pooling.
"""
import math
import random
from typing import List, Tuple, Optional

import numpy as np

from entity import _sat_aabb_overlap_xz


# ---------------------------------------------------------------------------
# Intersection & road geometry constants
# ---------------------------------------------------------------------------
INTERSECTION = {
    "center_x": -1.9,
    "center_z": 1.55,
    "x_min": -14.9,
    "x_max": 11.1,
    "z_min": -11.4,
    "z_max": 14.5,
}

ROAD_END_TOP_Z = 46.5
ROAD_END_BOTTOM_Z = -46.5
ROAD_END_LEFT_X = -47.9
ROAD_END_RIGHT_X = 48.0

# Right-hand traffic (Vietnam): drive on the RIGHT side of the road.
# Vertical road: center X ≈ -1.9, half-road ≈ 13.0 → each direction ≈ 6.5
# Horizontal road: center Z ≈ 1.55, half-road ≈ 13.0
_VERT_ROAD_CX = INTERSECTION["center_x"]
_HORIZ_ROAD_CZ = INTERSECTION["center_z"]
_LANE_OFFSET = 4.6  # offset from center to the right-lane center

# Lane center coordinates for right-hand traffic
LANE_SOUTHBOUND_X = _VERT_ROAD_CX + _LANE_OFFSET      # ≈ -6.5  (heading Z-, right side is X-)
LANE_NORTHBOUND_X = _VERT_ROAD_CX - _LANE_OFFSET      # ≈ +2.7  (heading Z+, right side is X+)
LANE_EASTBOUND_Z = _HORIZ_ROAD_CZ + _LANE_OFFSET       # ≈ -3.05 (heading X+)
LANE_WESTBOUND_Z = _HORIZ_ROAD_CZ - _LANE_OFFSET       # ≈ +6.15 (heading X-)

GROUND_Y = 0.05

# Spawn configurations: position (X, Z), yaw (degrees), direction label
# yaw convention: 0° = +Z, 90° = +X, 180° = -Z, 270°/-90° = -X
SPAWN_POINTS = [
    {   # From Top, heading South (Z-)
        "label": "top",
        "x": LANE_SOUTHBOUND_X,
        "z": ROAD_END_TOP_Z,
        "yaw": 180.0,
        "dir": (0.0, -1.0),  # (dx, dz) unit direction
    },
    {   # From Bottom, heading North (Z+)
        "label": "bottom",
        "x": LANE_NORTHBOUND_X,
        "z": ROAD_END_BOTTOM_Z,
        "yaw": 0.0,
        "dir": (0.0, 1.0),
    },
    {   # From Left, heading East (X+)
        "label": "left",
        "x": ROAD_END_LEFT_X,
        "z": LANE_EASTBOUND_Z,
        "yaw": 90.0,
        "dir": (1.0, 0.0),
    },
    {   # From Right, heading West (X-)
        "label": "right",
        "x": ROAD_END_RIGHT_X,
        "z": LANE_WESTBOUND_Z,
        "yaw": -90.0,
        "dir": (-1.0, 0.0),
    },
]

# Exit points: where a car exits when going straight through
EXIT_STRAIGHT = {
    "top": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z},
    "bottom": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z},
    "left": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z},
    "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z},
}

# Turn mappings: from_label → {right: exit_cfg, left: exit_cfg}
# Right-hand traffic: right turn is the short/easy turn, left turn crosses traffic.
_TURN_MAP = {
    "top": {  # heading south
        "right": {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
        "left":  {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
    },
    "bottom": {  # heading north
        "right": {"x": ROAD_END_RIGHT_X, "z": LANE_EASTBOUND_Z, "label": "right"},
        "left":  {"x": ROAD_END_LEFT_X, "z": LANE_WESTBOUND_Z, "label": "left"},
    },
    "left": {  # heading east
        "right": {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
        "left":  {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
    },
    "right": {  # heading west
        "right": {"x": LANE_NORTHBOUND_X, "z": ROAD_END_TOP_Z, "label": "top"},
        "left":  {"x": LANE_SOUTHBOUND_X, "z": ROAD_END_BOTTOM_Z, "label": "bottom"},
    },
}


# ---------------------------------------------------------------------------
# Bézier curve helpers
# ---------------------------------------------------------------------------

def _bezier_quadratic(p0: Tuple[float, float],
                      p1: Tuple[float, float],
                      p2: Tuple[float, float],
                      num_points: int = 12) -> List[Tuple[float, float]]:
    """Quadratic Bézier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2."""
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        z = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, z))
    return pts


def _bezier_cubic(p0: Tuple[float, float],
                  p1: Tuple[float, float],
                  p2: Tuple[float, float],
                  p3: Tuple[float, float],
                  num_points: int = 16) -> List[Tuple[float, float]]:
    """Cubic Bézier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3."""
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        u = 1.0 - t
        x = (u**3 * p0[0] + 3 * u**2 * t * p1[0]
             + 3 * u * t**2 * p2[0] + t**3 * p3[0])
        z = (u**3 * p0[1] + 3 * u**2 * t * p1[1]
             + 3 * u * t**2 * p2[1] + t**3 * p3[1])
        pts.append((x, z))
    return pts


# ---------------------------------------------------------------------------
# Intersection helpers
# ---------------------------------------------------------------------------

def is_inside_intersection(x: float, z: float) -> bool:
    """Check if a world XZ position is inside the intersection bounding box."""
    return (INTERSECTION["x_min"] <= x <= INTERSECTION["x_max"] and
            INTERSECTION["z_min"] <= z <= INTERSECTION["z_max"])


def _approach_margin() -> float:
    """Distance before intersection edge to start yielding."""
    return 5.0


def is_approaching_intersection(x: float, z: float, dx: float, dz: float) -> bool:
    """Check if a car just outside the intersection is heading toward it."""
    if is_inside_intersection(x, z):
        return False
    # Project a few meters ahead
    margin = _approach_margin()
    fx, fz = x + dx * margin, z + dz * margin
    return is_inside_intersection(fx, fz)


# ---------------------------------------------------------------------------
# Route generation
# ---------------------------------------------------------------------------

def _intersection_entry_point(spawn_label: str, spawn_x: float, spawn_z: float) -> Tuple[float, float]:
    """Point where the car reaches the intersection perimeter along its lane."""
    if spawn_label == "top":
        return (spawn_x, INTERSECTION["z_max"])
    elif spawn_label == "bottom":
        return (spawn_x, INTERSECTION["z_min"])
    elif spawn_label == "left":
        return (INTERSECTION["x_min"], spawn_z)
    elif spawn_label == "right":
        return (INTERSECTION["x_max"], spawn_z)
    return (spawn_x, spawn_z)


def _intersection_exit_point(exit_label: str, exit_x: float, exit_z: float) -> Tuple[float, float]:
    """Point where the car exits the intersection before heading to road end."""
    if exit_label == "bottom":
        return (exit_x, INTERSECTION["z_min"])
    elif exit_label == "top":
        return (exit_x, INTERSECTION["z_max"])
    elif exit_label == "right":
        return (INTERSECTION["x_max"], exit_z)
    elif exit_label == "left":
        return (INTERSECTION["x_min"], exit_z)
    return (exit_x, exit_z)


def generate_route(spawn_idx: int, maneuver: str = "straight") -> List[Tuple[float, float]]:
    """
    Generate a list of (x, z) waypoints for a car spawning at `spawn_idx`.

    maneuver: "straight", "right", "left"

    Returns waypoints from spawn → approach → through intersection → exit → road end.
    """
    sp = SPAWN_POINTS[spawn_idx]
    label = sp["label"]
    sx, sz = sp["x"], sp["z"]

    waypoints: List[Tuple[float, float]] = []

    # ---- Approach waypoints (from spawn to intersection edge) ----
    entry = _intersection_entry_point(label, sx, sz)
    # Add 2-3 intermediate approach waypoints along the straight lane
    num_approach = 3
    for i in range(1, num_approach + 1):
        t = i / (num_approach + 1)
        wx = sx + t * (entry[0] - sx)
        wz = sz + t * (entry[1] - sz)
        waypoints.append((wx, wz))
    waypoints.append(entry)

    # if maneuver == "straight":
    #     # Straight through intersection
    #     exit_cfg = EXIT_STRAIGHT[label]
    #     ex, ez = exit_cfg["x"], exit_cfg["z"]

    #     # Determine exit label for the exit point calculation
    #     exit_labels = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    #     exit_lbl = exit_labels[label]
    #     exit_pt = _intersection_exit_point(exit_lbl, ex, ez)

    #     # A couple of interior waypoints through the intersection
    #     mid_x = (entry[0] + exit_pt[0]) * 0.5
    #     mid_z = (entry[1] + exit_pt[1]) * 0.5
    #     waypoints.append((mid_x, mid_z))
    #     waypoints.append(exit_pt)

    #     # Post-intersection straight to road end
    #     num_exit = 3
    #     for i in range(1, num_exit + 1):
    #         t = i / (num_exit + 1)
    #         wx = exit_pt[0] + t * (ex - exit_pt[0])
    #         wz = exit_pt[1] + t * (ez - exit_pt[1])
    #         waypoints.append((wx, wz))
    #     waypoints.append((ex, ez))
    if maneuver == "straight":
        exit_cfg = EXIT_STRAIGHT[label]
        ex, ez = exit_cfg["x"], exit_cfg["z"]
        exit_labels = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
        exit_lbl = exit_labels[label]
        exit_pt = _intersection_exit_point(exit_lbl, ex, ez)

        # Thay mid bằng crossing waypoints chính xác tại giao điểm các lane
        # → Northbound/Southbound chia sẻ waypoints với Eastbound/Westbound
        if label in ("top", "bottom"):
            # Xe đi dọc: add waypoints tại Z của 2 lane ngang
            cross_zs = sorted(
                [z for z in (LANE_WESTBOUND_Z, LANE_EASTBOUND_Z)
                if INTERSECTION["z_min"] < z < INTERSECTION["z_max"]],
                reverse=(label == "top"),   # southbound → Z giảm dần
            )
            for cz in cross_zs:
                waypoints.append((entry[0], cz))
        else:
            # Xe đi ngang: add waypoints tại X của 2 lane dọc
            cross_xs = sorted(
                [x for x in (LANE_NORTHBOUND_X, LANE_SOUTHBOUND_X)
                if INTERSECTION["x_min"] < x < INTERSECTION["x_max"]],
                reverse=(label == "right"),  # westbound → X giảm dần
            )
            for cx in cross_xs:
                waypoints.append((cx, entry[1]))   # entry[1] = Z cố định của lane ngang

        waypoints.append(exit_pt)

        # Post-intersection (giữ nguyên)
        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (ex - exit_pt[0])
            wz = exit_pt[1] + t * (ez - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((ex, ez))

    elif maneuver in ("right", "left"):
        # Turning through intersection using Bezier curve
        turn_cfg = _TURN_MAP[label][maneuver]
        exit_road_x, exit_road_z = turn_cfg["x"], turn_cfg["z"]
        exit_lbl = turn_cfg["label"]
        exit_pt = _intersection_exit_point(exit_lbl, exit_road_x, exit_road_z)

        # Geometric corner: intersection of entry forward line and exit backward line
        if label in ("top", "bottom"):
            # Entry is vertical, exit is horizontal
            corner = (entry[0], exit_pt[1])  # match entry X, exit Z
        else:
            # Entry is horizontal, exit is vertical
            corner = (exit_pt[0], entry[1])  # match exit X, entry Z

        if maneuver == "right":
            # RIGHT TURN: tight quadratic Bezier, control point at geometric corner
            # No center-pull → stays in the near-side lane
            curve_pts = _bezier_quadratic(entry, corner, exit_pt, num_points=12)
        else:
            # LEFT TURN: wide cubic Bezier, pull control points past the center
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

        # Skip the first point (already added as entry)
        waypoints.extend(curve_pts[1:])

        # Post-intersection straight to road end
        num_exit = 3
        for i in range(1, num_exit + 1):
            t = i / (num_exit + 1)
            wx = exit_pt[0] + t * (exit_road_x - exit_pt[0])
            wz = exit_pt[1] + t * (exit_road_z - exit_pt[1])
            waypoints.append((wx, wz))
        waypoints.append((exit_road_x, exit_road_z))

    return waypoints


def random_maneuver() -> str:
    """Pick a random maneuver with weighted probability."""
    r = random.random()
    if r < 0.50:
        return "straight"
    elif r < 0.75:
        return "right"
    else:
        return "left"


# ---------------------------------------------------------------------------
# Traffic Manager (Object Pool + Collision Avoidance)
# ---------------------------------------------------------------------------

# Tuning constants
_BRAKE_DECEL = 12.0       # units/s² when braking
_ACCEL = 8.0              # units/s² when accelerating to target
_RADAR_LOOKAHEAD = 2.5    # meters ahead for forward collision check
_MIN_SPEED = 0.0
_TRAJ_LOOKAHEAD = 7       # how many future waypoints to compare
_TRAJ_CONFLICT_DIST = 3.5 # meters; below this → trajectory conflict


_SPAWN_IMMUNITY_TIME = 3.0    # seconds after spawn where radar is ignored
_STAGGER_DISTANCE = 3.0      # meters between cars on the same lane at init
_SPAWN_CLEAR_RADIUS = 6.0     # meters; don't spawn if another car is this close

_DEADLOCK_TIMEOUT = 3.5    # giây đứng yên → tự giải deadlock
_TRAJ_PRIORITY_MARGIN = 0.8  # epsilon tránh equal-distance deadlock
_INTERSECTION_APPROACH_MARGIN = 8.0

class TrafficManager:
    """
    Manages the car pool: assigns routes, handles respawning,
    runs collision avoidance each frame.
    """

    def __init__(self):
        self.cars: list = []           # references to Car entities
        self._spawn_cooldowns: dict = {}  # car_name -> cooldown timer
        self._spawn_timers: dict = {}     # car_name -> seconds since spawn
        self._stopped_timers: dict = {}

    def register_cars(self, cars: list):
        """Register the pool of Car entities to manage, staggering initial spawns."""
        self.cars = list(cars)
        n_spawns = len(SPAWN_POINTS)
        for idx, car in enumerate(self.cars):
            # Round-robin across spawn points so no two cars share one
            spawn_idx = idx % n_spawns
            self._spawn_car(car, spawn_idx=spawn_idx, stagger_offset=idx)

    def _spawn_car(self, car, spawn_idx: int = -1, stagger_offset: int = 0):
        """
        Teleport a car to a spawn point and assign a new route.
        - Uses round-robin spawn_idx if given, else picks a random clear one.
        - Offsets the car backwards along the lane to stagger traffic.
        """
        if spawn_idx < 0:
            spawn_idx = self._pick_clear_spawn()

        sp = SPAWN_POINTS[spawn_idx]
        maneuver = random_maneuver()
        waypoints = generate_route(spawn_idx, maneuver)
        target_speed = random.uniform(3.0, 5.0)

        # Stagger: shift the spawn position backwards along the lane direction
        # so cars on the same lane don't overlap
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
        self._spawn_timers[car.name] = 0.0  # reset immunity timer

    def _pick_clear_spawn(self) -> int:
        """Pick a spawn point that has no active car nearby."""
        active_positions = []
        for c in self.cars:
            if c.is_active and not c.route_finished:
                active_positions.append((float(c.position[0]), float(c.position[2])))

        # Try each spawn point, prefer one with no nearby car
        indices = list(range(len(SPAWN_POINTS)))
        random.shuffle(indices)
        for idx in indices:
            sp = SPAWN_POINTS[idx]
            sx, sz = sp["x"], sp["z"]
            clear = True
            for ax, az in active_positions:
                if math.hypot(ax - sx, az - sz) < _SPAWN_CLEAR_RADIUS:
                    clear = False
                    break
            if clear:
                return idx
        # Fallback: pick random and rely on stagger offset
        return random.randint(0, len(SPAWN_POINTS) - 1)

    def _is_immune(self, car) -> bool:
        """True if the car was recently spawned and should ignore radar blocks."""
        return self._spawn_timers.get(car.name, 0.0) < _SPAWN_IMMUNITY_TIME

    @staticmethod
    def _upcoming_waypoints(car, lookahead: int = _TRAJ_LOOKAHEAD):
        """Return list of (x,z) for the car's next `lookahead` waypoints."""
        start = car.current_wp_idx
        end = min(start + lookahead, len(car.waypoints))
        return car.waypoints[start:end]

    @staticmethod
    def _path_distance_to_wp(car, wp_idx_in_slice: int) -> float:
        """
        Approximate path-distance from the car's current position to a
        waypoint at `wp_idx_in_slice` steps ahead (within the upcoming slice).
        """
        if wp_idx_in_slice <= 0:
            return 0.0
        cx, cz = float(car.position[0]), float(car.position[2])
        start = car.current_wp_idx
        total = 0.0
        px, pz = cx, cz
        for k in range(wp_idx_in_slice + 1):
            idx = start + k
            if idx >= len(car.waypoints):
                break
            nx, nz = car.waypoints[idx]
            total += math.hypot(nx - px, nz - pz)
            px, pz = nx, nz
        return total

    def update(self, dt: float):
        """
        Called every frame BEFORE scene.update().
        1. Recycle finished cars.
        2. Run collision avoidance (radar + trajectory intersection).
        3. Let cars advance (they call their own update in scene.update).
        """
        dt = float(max(0.0, min(dt, 0.1)))  # clamp dt

        # --- Tick spawn immunity timers ---
        for car in self.cars:
            if car.name in self._spawn_timers:
                self._spawn_timers[car.name] += dt

        # --- 1. Recycle finished cars ---
        for car in self.cars:
            if not car.is_active:
                continue
            if car.route_finished:
                self._spawn_car(car)

        # --- 2. Collision avoidance ---
        active_cars = [c for c in self.cars if c.is_active]
        upcoming = [self._upcoming_waypoints(c) for c in active_cars]

        for i, car_a in enumerate(active_cars):
            blocked = False

            if self._is_immune(car_a):
                car_a.speed = min(car_a.target_speed, car_a.speed + _ACCEL * dt)
                self._stopped_timers[car_a.name] = 0.0
                continue

            car_a_x = float(car_a.position[0])
            car_a_z = float(car_a.position[2])
            fwd_a = car_a._movement_forward_world()

            # Zone check: chỉ chạy trajectory check khi xe đang trong / sắp vào intersection
            proj_x = car_a_x + float(fwd_a[0]) * _INTERSECTION_APPROACH_MARGIN
            proj_z = car_a_z + float(fwd_a[2]) * _INTERSECTION_APPROACH_MARGIN
            car_a_near_intersection = (
                is_inside_intersection(car_a_x, car_a_z)
                or is_inside_intersection(proj_x, proj_z)
            )

            # 2a) Radar — same-direction only (đường thẳng / following distance)
            fut_min, fut_max = car_a.future_aabb_xz(_RADAR_LOOKAHEAD)
            for j, car_b in enumerate(active_cars):
                if i == j:
                    continue
                if self._is_immune(car_b):
                    continue
                fwd_b = car_b._movement_forward_world()
                dot = float(fwd_a[0]) * float(fwd_b[0]) + float(fwd_a[2]) * float(fwd_b[2])
                if dot <= 0.3:
                    continue   # khác hướng → trajectory xử lý
                b_min, b_max = car_b.world_aabb_xz()
                if _sat_aabb_overlap_xz(fut_min, fut_max, b_min, b_max):
                    blocked = True
                    break

            # 2b) Trajectory — cross-direction, CHỈ khi gần intersection
            #     Giải quyết cả deadlock và pass-through
            if not blocked and car_a_near_intersection:
                wps_a = upcoming[i]
                for j, car_b in enumerate(active_cars):
                    if i == j:
                        continue
                    if self._is_immune(car_b):
                        continue
                    fwd_b = car_b._movement_forward_world()
                    dot = float(fwd_a[0]) * float(fwd_b[0]) + float(fwd_a[2]) * float(fwd_b[2])
                    if dot > 0.3:
                        continue   # cùng hướng → radar xử lý

                    wps_b = upcoming[j]
                    conflict_found = False
                    best_ai = best_bi = -1
                    best_dist = float('inf')
                    for ai, (ax, az) in enumerate(wps_a):
                        for bi, (bx, bz) in enumerate(wps_b):
                            d = math.hypot(ax - bx, az - bz)
                            if d < _TRAJ_CONFLICT_DIST and d < best_dist:
                                best_dist = d
                                best_ai, best_bi = ai, bi
                                conflict_found = True

                    if conflict_found:
                        dist_a = self._path_distance_to_wp(car_a, best_ai)
                        dist_b = self._path_distance_to_wp(car_b, best_bi)
                        if dist_a > dist_b + _TRAJ_PRIORITY_MARGIN:
                            # car_a rõ ràng xa hơn → yield
                            blocked = True
                            break
                        elif abs(dist_a - dist_b) <= _TRAJ_PRIORITY_MARGIN:
                            # Khoảng cách bằng nhau → tiebreaker deterministic
                            # index cao hơn yield (nhất quán, không flip-flop)
                            if i > j:
                                blocked = True
                                break

            # 2c) Deadlock timeout safety net
            if car_a.speed < 0.1:
                self._stopped_timers[car_a.name] = self._stopped_timers.get(car_a.name, 0.0) + dt
            else:
                self._stopped_timers[car_a.name] = 0.0

            if self._stopped_timers.get(car_a.name, 0.0) > _DEADLOCK_TIMEOUT:
                blocked = False
                self._stopped_timers[car_a.name] = 0.0

            # 2d) Apply speed
            if blocked:
                car_a.speed = max(_MIN_SPEED, car_a.speed - _BRAKE_DECEL * dt)
            else:
                car_a.speed = min(car_a.target_speed, car_a.speed + _ACCEL * dt)