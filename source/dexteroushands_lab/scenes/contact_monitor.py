"""Per-body PhysX contact classification and episode-local histories.

Isaac Sim 6.0.1's multi-environment ContactSensor filter view matches two
internal entries for the official hand bodies. Exact, per-environment native
RigidContactView paths were GPU-probed and retain one-to-many force matrices.
Source: https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/contact_sensor.html
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


ENV_PATH = re.compile(r"^/World/envs/env_(\d+)/(LeftHand|RightHand|Bottle|Cap)(?:/([^/]+))?$")
GROUND_PATH = "/World/ground/GroundPlane/CollisionPlane"


def classify_contact_pair(source_path: str, partner_path: str) -> str | None:
    """Return an anatomical/physical category only for an exact valid pair."""
    source = ENV_PATH.fullmatch(source_path)
    partner = ENV_PATH.fullmatch(partner_path)
    if partner_path == GROUND_PATH and source:
        return {"Bottle": "bottle_ground", "Cap": "cap_ground"}.get(source.group(2))
    if not source or not partner or source.group(1) != partner.group(1):
        return None
    roles = {source.group(2), partner.group(2)}
    for hand, obj, name in (
        ("LeftHand", "Bottle", "left_bottle"),
        ("RightHand", "Cap", "right_cap"),
        ("LeftHand", "Cap", "left_cap"),
        ("RightHand", "Bottle", "right_bottle"),
    ):
        if roles == {hand, obj}:
            return name
    if roles == {"LeftHand", "RightHand"}:
        return "hand_hand"
    return None


@dataclass(frozen=True)
class ContactMonitorConfig:
    detection_force_n: float = 0.01  # numerical event floor; never a grasp criterion
    max_contact_data_count: int = 128
    include_hand_hand: bool = True

    def __post_init__(self):
        if self.detection_force_n < 0 or self.max_contact_data_count < 1:
            raise ValueError("Contact detection floor must be nonnegative and capacity positive")


class ContactHistory:
    """Count contact duration and forces without sharing state across environments."""

    def __init__(self, num_envs: int, config: ContactMonitorConfig):
        self.config = config
        self.num_envs = num_envs
        self.reset(range(num_envs))

    def reset(self, env_ids) -> None:
        if not hasattr(self, "_data"):
            self._data = [{} for _ in range(self.num_envs)]
            self._steps = [0] * self.num_envs
            self._elapsed = [0.0] * self.num_envs
            self._fresh = [False] * self.num_envs
            self._touched = [set() for _ in range(self.num_envs)]
        for env in env_ids:
            self._data[env] = {}
            self._steps[env] = 0
            self._elapsed[env] = 0.0
            self._fresh[env] = False
            self._touched[env] = set()

    def record(self, env: int, category: str, source_body: str, partner_body: str,
               normal_force_n: float, dt: float, *, position=None, friction_force_n=None,
               separation_m=None) -> None:
        if not math.isfinite(normal_force_n) or normal_force_n < 0:
            raise ValueError("Contact force must be finite and nonnegative")
        if normal_force_n < self.config.detection_force_n:
            return
        data = self._data[env].setdefault(category, {
            "active_steps": 0, "active_time_s": 0.0,
            "peak_force_n": 0.0, "sum_force_n": 0.0,
            "force_samples": 0, "bodies": {},
        })
        self._touched[env].add(category)
        data["peak_force_n"] = max(data["peak_force_n"], normal_force_n)
        data["sum_force_n"] += normal_force_n
        data["force_samples"] += 1
        key = f"{source_body}|{partner_body}"
        detail = data["bodies"].setdefault(key, {
            "source_body": source_body, "partner_body": partner_body,
            "peak_force_n": 0.0, "samples": 0, "latest_position_w": None,
            "latest_friction_force_n": None, "minimum_separation_m": None,
        })
        detail["peak_force_n"] = max(detail["peak_force_n"], normal_force_n)
        detail["samples"] += 1
        if position is not None and all(math.isfinite(x) for x in position):
            detail["latest_position_w"] = tuple(float(x) for x in position)
        if friction_force_n is not None and math.isfinite(friction_force_n):
            detail["latest_friction_force_n"] = float(friction_force_n)
        if separation_m is not None and math.isfinite(separation_m):
            current = detail["minimum_separation_m"]
            detail["minimum_separation_m"] = (float(separation_m) if current is None
                                               else min(current, float(separation_m)))

    def finish_step(self, env: int, dt: float) -> None:
        if dt <= 0:
            raise ValueError("Physics dt must be positive")
        self._steps[env] += 1
        self._elapsed[env] += dt
        for category in self._touched[env]:
            self._data[env][category]["active_steps"] += 1
            self._data[env][category]["active_time_s"] += dt
        self._touched[env].clear()
        self._fresh[env] = True

    def is_fresh(self, env: int) -> bool:
        return self._fresh[env]

    def summary(self, env: int) -> dict:
        steps = self._steps[env]
        result = {}
        for category, entry in self._data[env].items():
            result[category] = {
                "contact_ratio": entry["active_steps"] / steps if steps else 0.0,
                "active_steps": entry["active_steps"],
                "active_time_s": entry["active_time_s"],
                "elapsed_time_s": self._elapsed[env],
                "peak_force_n": entry["peak_force_n"],
                "mean_force_n": entry["sum_force_n"] / entry["force_samples"],
                "bodies": entry["bodies"],
            }
        return result


class BimanualContactMonitor:
    """Read exact per-environment PhysX one-to-many contact views.

    Construct after ``sim.reset()``. Call ``update_after_physics_step`` only
    after the simulation and scene have advanced, never directly after reset.
    ``force_matrix`` is normal force in the pinned Isaac Sim 6.0.1 backend.
    """

    def __init__(self, sim_view, num_envs: int, left_spec, right_spec,
                 config: ContactMonitorConfig | None = None):
        self.config = config or ContactMonitorConfig()
        self.history = ContactHistory(num_envs, self.config)
        self._views = []
        for env in range(num_envs):
            root = f"/World/envs/env_{env}"
            hand_paths = [f"{root}/LeftHand/{name}" for name in left_spec.body_names]
            hand_paths += [f"{root}/RightHand/{name}" for name in right_spec.body_names]
            for obj in ("Bottle", "Cap"):
                self._add_view(sim_view, env, f"{root}/{obj}", hand_paths + [GROUND_PATH])
            if self.config.include_hand_hand:
                right_paths = [f"{root}/RightHand/{name}" for name in right_spec.body_names]
                for name in left_spec.body_names:
                    self._add_view(sim_view, env, f"{root}/LeftHand/{name}", right_paths)

    def _add_view(self, sim_view, env: int, source: str, filters: list[str]) -> None:
        view = sim_view.create_rigid_contact_view(
            source, filter_patterns=filters,
            max_contact_data_count=self.config.max_contact_data_count,
        )
        if view is None or list(view.sensor_paths) != [source]:
            raise RuntimeError(f"PhysX cannot create one-body contact view for {source}")
        actual = list(view.filter_paths[0])
        if actual != filters:
            raise RuntimeError(f"PhysX contact filter order mismatch at {source}: {actual}")
        self._views.append((env, source, tuple(filters), view))

    def reset(self, env_ids) -> None:
        self.history.reset(env_ids)

    def update_after_physics_step(self, dt: float) -> None:
        import warp as wp

        for env, source, filters, view in self._views:
            forces = wp.to_torch(view.get_contact_force_matrix(dt=dt)).reshape(-1, 3)
            if forces.shape[0] != len(filters):
                raise RuntimeError(f"PhysX contact force shape changed for {source}")
            strengths = forces.norm(dim=1)
            active = (strengths >= self.config.detection_force_n).nonzero().flatten().tolist()
            positions = None
            friction = None
            if active:
                try:
                    _, points, _, distances, counts, starts = view.get_contact_data(dt=dt)
                    positions = (wp.to_torch(points), wp.to_torch(distances).reshape(-1),
                                 wp.to_torch(counts).reshape(-1),
                                 wp.to_torch(starts).reshape(-1))
                    fvec, _, fcounts, fstarts = view.get_friction_data(dt=dt)
                    friction = (wp.to_torch(fvec), wp.to_torch(fcounts).reshape(-1),
                                wp.to_torch(fstarts).reshape(-1))
                except (AttributeError, NotImplementedError):
                    pass
            for index in active:
                partner = filters[index]
                category = classify_contact_pair(source, partner)
                if category is None:
                    continue
                point = None
                separation = None
                if positions is not None:
                    pbuf, distances, counts, starts = positions
                    count = int(counts[index].item())
                    if count:
                        start = int(starts[index].item())
                        point = tuple(pbuf[start:start + count].float().mean(dim=0).tolist())
                        separation = float(distances[start:start + count].float().min().item())
                friction_n = None
                if friction is not None:
                    fbuf, counts, starts = friction
                    count = int(counts[index].item())
                    if count:
                        start = int(starts[index].item())
                        friction_n = float(fbuf[start:start + count].float().sum(dim=0).norm().item())
                self.history.record(
                    env, category, source.rsplit("/", 1)[-1], partner.rsplit("/", 1)[-1],
                    float(strengths[index].item()), dt, position=point,
                    friction_force_n=friction_n, separation_m=separation,
                )
        for env in range(self.history.num_envs):
            self.history.finish_step(env, dt)

    def summary(self, env: int) -> dict:
        if not self.history.is_fresh(env):
            raise RuntimeError("Contact data invalid after reset until a new physics step")
        return self.history.summary(env)
