"""Metric engineering example for an uncalibrated plastic bottle and cap."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BottleConfig:
    source_profile: str = "engineering_example"
    bottle_radius: float = 0.035
    bottle_height: float = 0.20
    neck_outer_radius: float = 0.012
    neck_height: float = 0.040
    cap_outer_radius: float = 0.017
    cap_height: float = 0.025
    visual_radial_segments: int = 64
    thread_pitch: float = 0.003
    thread_turns: float = 3.0
    allow_detachment: bool = False
    detachment_angle_margin: float = 0.06
    detachment_overlap_fraction: float = 0.05
    detachment_max_constraint_error: float = 0.0005
    thread_handedness: str = "right"
    thread_clearance: float = 0.0005
    thread_height: float = 0.001
    initial_engagement: float = 0.009
    cap_top_clearance: float = 0.002
    cap_mass: float = 0.006
    bottle_mass: float = 0.030
    static_friction: float = 0.6
    dynamic_friction: float = 0.4
    breakaway_torque: float = 0.020
    loose_breakaway_torque: float = 0.004
    breakaway_stiffness: float = 0.2
    breakaway_damping: float = 0.003
    slip_angle_ratio: float = 1.5
    slip_speed: float = 0.1
    restick_speed: float = 0.05
    running_resistance_torque: float = 0.006
    loose_running_resistance_torque: float = 0.001
    engagement_resistance_exponent: float = 1.5
    max_constraint_force: float = 8.0
    max_constraint_torque: float = 0.08
    lateral_stiffness: float = 10.0
    lateral_damping: float = 0.1
    axial_stiffness: float = 500.0
    axial_damping: float = 3.5
    end_stop_stiffness: float = 0.5
    end_stop_damping: float = 0.02
    tilt_stiffness: float = 0.001
    tilt_damping: float = 0.0001
    friction_velocity_scale: float = 20.0
    max_body_linear_velocity: float = 5.0
    max_body_angular_velocity_deg: float = 1080.0
    body_angular_damping: float = 1.0

    def __post_init__(self):
        numeric = (
            self.bottle_radius, self.bottle_height, self.neck_outer_radius,
            self.neck_height, self.cap_outer_radius, self.cap_height,
            self.thread_pitch, self.thread_clearance, self.thread_height,
            self.cap_mass, self.bottle_mass, self.max_constraint_force,
            self.max_constraint_torque, self.friction_velocity_scale,
            self.axial_stiffness, self.axial_damping,
            self.breakaway_stiffness,
            self.breakaway_damping,
            self.slip_angle_ratio, self.slip_speed, self.restick_speed,
            self.end_stop_stiffness, self.end_stop_damping,
            self.engagement_resistance_exponent,
            self.detachment_angle_margin, self.detachment_overlap_fraction,
            self.detachment_max_constraint_error,
        )
        if any(not math.isfinite(x) or x <= 0 for x in numeric):
            raise ValueError("Bottle dimensions, mass, pitch, spring parameters and limits must be finite and positive")
        if (not isinstance(self.thread_turns, (int, float))
                or not math.isfinite(self.thread_turns) or not 0 < self.thread_turns <= 10):
            raise ValueError("thread_turns must be finite and in (0, 10]")
        if not isinstance(self.visual_radial_segments, int) or not 24 <= self.visual_radial_segments <= 256:
            raise ValueError("visual_radial_segments must be an integer in [24, 256]")
        if self.thread_handedness not in ("right", "left"):
            raise ValueError("thread_handedness must be right or left")
        if not isinstance(self.allow_detachment, bool):
            raise ValueError("allow_detachment must be bool")
        if self.detachment_overlap_fraction >= 1 or self.detachment_angle_margin >= 2 * math.pi:
            raise ValueError("detachment tolerances exceed one turn or full engagement")
        if self.cap_inner_radius >= self.cap_outer_radius:
            raise ValueError("cap wall thickness must be positive")
        if self.neck_outer_radius >= self.bottle_radius:
            raise ValueError("neck must be narrower than bottle")
        if not (0 <= self.initial_engagement < min(self.neck_height, self.cap_height - self.cap_top_clearance)):
            raise ValueError("initial_engagement would intersect bottle shoulder or neck top")
        if self.dynamic_friction > self.static_friction or self.dynamic_friction < 0:
            raise ValueError("friction coefficients are invalid")
        if self.breakaway_torque < 0 or self.running_resistance_torque < 0:
            raise ValueError("resistance torques must be nonnegative")
        if not (0 <= self.loose_breakaway_torque <= self.breakaway_torque):
            raise ValueError("loose breakaway torque must be within tight breakaway limit")
        if not (0 <= self.loose_running_resistance_torque <= self.running_resistance_torque):
            raise ValueError("loose running torque must be within tight running limit")

    @property
    def thread_sign(self) -> int:
        return 1 if self.thread_handedness == "right" else -1

    @property
    def cap_inner_radius(self) -> float:
        return self.neck_outer_radius + self.thread_height + self.thread_clearance

    @classmethod
    def pco1881_reference(cls) -> "BottleConfig":
        """Supplier-grounded 28 mm PCO 1881 example, not a measured user bottle.

        Thread data: BERICAP PCO 1881 brochure (650 degrees, 2.7 mm pitch).
        Closure: Delta El Nile DEN-CLO-PCO_1881 (29.6 mm top diameter,
        15.65 mm height, 2.34 g; 5-14 in-lb removal-torque guideline).
        Bottle body, running friction, fill mass and seal are not SKU-matched.
        """
        return cls(
            source_profile="pco1881_reference",
            bottle_radius=0.0325, bottle_height=0.190,
            neck_outer_radius=0.01247, neck_height=0.017,
            cap_outer_radius=0.0148, cap_height=0.01565,
            thread_pitch=0.0027, thread_turns=650.0 / 360.0,
            thread_height=0.00123, thread_clearance=0.0002,
            initial_engagement=0.013,
            cap_mass=0.00234, bottle_mass=0.520,
            breakaway_torque=0.90, loose_breakaway_torque=0.05,
            breakaway_stiffness=20.0, breakaway_damping=0.05,
            running_resistance_torque=0.12,
            loose_running_resistance_torque=0.02,
            max_constraint_torque=1.5, max_constraint_force=8.0,
            axial_damping=1.0,
        )

    @property
    def cap_initial_z(self) -> float:
        """Cap COM relative to bottle COM for the example upright initial pose."""
        return (
            self.bottle_height / 2 + self.neck_height + self.cap_height / 2
            + self.cap_top_clearance - self.initial_engagement
        )


def sample_helix(
    config: BottleConfig, *, segments_per_turn: int = 8, radius: float | None = None,
    start_z: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Low-resolution helical centreline; geometry tools may use capsules per edge."""
    if segments_per_turn < 4:
        raise ValueError("At least four segments per turn are needed")
    r = config.neck_outer_radius if radius is None else radius
    count = math.ceil(config.thread_turns * segments_per_turn)
    return tuple(
        (
            r * math.cos(2 * math.pi * config.thread_turns * i / count),
            r * math.sin(2 * math.pi * config.thread_turns * i / count),
            start_z + config.thread_sign * config.thread_pitch * config.thread_turns * i / count,
        )
        for i in range(count + 1)
    )
