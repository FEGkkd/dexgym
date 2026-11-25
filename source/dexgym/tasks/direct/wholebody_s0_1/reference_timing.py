"""The teleoperation position-rate envelope and source-local reference retiming."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from ..wholebody_s0.bag import digest
from ..wholebody_s0.prepare import write_json
from ..wholebody_s0.sync import runs
from . import DOF, DT

PERIOD_NS = round(DT * 1e9)
RATE_FIELD = "position_rate_rad_s"
RATE_SCHEMA = "teleop_position_rates_v2"
RATE_TOLERANCE = 1e-9
NEAR_LIMIT_RATIO = .9
DEXGYM_PROFILE = "dexgym_original"


def position_rates(names, metadata=None):
    values = None if metadata is None else metadata.get(RATE_FIELD)
    if values is None:
        return np.array([1.35 if name.startswith("Joint") else 5. for name in names])
    rate = np.asarray([np.inf if value is None else value for value in values], dtype=float)
    finite = np.asarray([value is not None for value in values])
    if (metadata["joint_names"] != list(names) or rate.shape != (DOF,)
            or not np.isfinite(rate[finite]).all() or (rate <= 0).any()
            or any(name.startswith("Joint") and not bounded for name, bounded in zip(names, finite, strict=True))):
        raise ValueError("Position rates must match MJCF joint order and all arm rates must be positive and finite")
    return rate


def load_contract(path, model):
    if path is None:
        return None
    rates = json.loads(path.read_text(encoding="utf-8"))
    if rates["schema"] not in (RATE_SCHEMA, "teleop_sim_position_rates_v1") or rates["model_sha256"] != digest(model):
        raise ValueError("The teleoperation speeds are bound to a different native model")
    position_rates(rates["joint_names"], rates)
    return rates


def resample(values, rotations, grid):
    indices = np.arange(len(values))
    lo = grid.astype(int)
    hi = np.minimum(lo + 1, indices[-1])
    weight = (grid - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    x = values[lo] + weight * (values[hi] - values[lo])
    if rotations is None:
        return x
    shape = rotations.shape[1:-2]
    r = [Slerp(indices, Rotation.from_matrix(a))(grid).as_matrix()
         for a in rotations.reshape(len(indices), -1, 3, 3).transpose(1, 0, 2, 3)]
    r = np.stack(r, axis=1).reshape(len(grid), *shape, 3, 3)
    integer = grid == lo
    r[integer] = rotations[lo[integer]]
    return x, r


def local_clock(rate_ratio, *, slow, include_near=False):
    """Allocate extra ticks only within windows containing a rate exceedance."""
    fast = rate_ratio > 1. + RATE_TOLERANCE
    count = len(rate_ratio)
    grid = np.arange(count + 1, dtype=float)
    window = fast.copy()
    if slow and fast.any():
        if include_near:
            near = rate_ratio >= NEAR_LIMIT_RATIO
            window[:-1] |= fast[1:] & near[:-1]
            window[1:] |= fast[:-1] & near[1:]
        parts, last = [], 0
        for a, b in runs(window):
            parts.append(grid[last:a])
            phase = np.r_[0., np.cumsum(np.maximum(1., rate_ratio[a:b]))]
            ticks = int(np.ceil(phase[-1] - RATE_TOLERANCE))
            parts.append(np.interp(np.linspace(0., phase[-1], ticks + 1)[:-1], phase, grid[a:b + 1]))
            last = b
        grid = np.concatenate((*parts, grid[last:]))
    else:
        window[:] = False
    return grid, {
        "original_intervals": count, "overspeed_intervals": int(fast.sum()),
        "retimed_original_intervals": int(window.sum()), "added_intervals": len(grid) - count - 1,
        "peak_original_rate_ratio": float(rate_ratio.max(initial=0.)),
    }


def retime_samples(q, goal, clock, slew, *, slow, include_near=False):
    ratio = (np.abs(np.diff(q, axis=0)) / slew).max(axis=-1)
    grid, metrics = local_clock(ratio, slow=slow, include_near=include_near)
    if metrics["added_intervals"]:
        wrist, rotation, *points = goal
        wrist, rotation = resample(wrist, rotation, grid)
        goal = (wrist, rotation, *(resample(value, None, grid) for value in points))
        q = resample(q, None, grid)
        offsets = np.interp(grid, np.arange(len(clock)), clock - clock[0])
        clock = clock[0] + np.rint(offsets).astype(np.int64)
    metrics["peak_reference_rate_ratio"] = float((np.abs(np.diff(q, axis=0)) / slew).max())
    if slow and np.any(np.abs(np.diff(q, axis=0)) > slew + 2e-6):
        raise ValueError("The local reference clock exceeds its permitted position rate")
    return q, goal, clock, metrics


def describe_rates(args):
    import mujoco

    from .targets import joint_names

    names = joint_names(mujoco.MjModel.from_xml_path(str(args.model)))
    report = {"schema": RATE_SCHEMA, "model_sha256": digest(args.model), "joint_names": names}
    if args.profile == DEXGYM_PROFILE:
        rate = position_rates(names)
        report.update(profile=DEXGYM_PROFILE, position_rate_rad_s=rate.tolist(), sources={},
                      control_period_s=DT, command_step_limit_rad=(rate * DT).tolist())
        write_json(args.output, report)
        return

    key = "arm.max_command_rate_rad_sec"

    def parameter_sets(path):
        return (node["ros__parameters"] for node in yaml.safe_load(path.read_text(encoding="utf-8")).values())

    base, = (params for params in parameter_sets(args.base) if key in params)
    overlay, = (params for params in parameter_sets(args.overlay) if key in params)
    driver, = (params for params in parameter_sets(args.arm_driver) if "max_joint_rate_deg_s" in params)
    arm = base | overlay
    command_rate = float(arm[key])
    computed_rate = np.deg2rad(arm["arm.analytic_joint_vmax_deg_s"])
    computed_rate *= arm["arm.analytic_sdk_velocity_ratio"] / 100 * arm["arm.analytic_command_rate_margin"]
    if not np.isclose(command_rate, computed_rate, rtol=1e-6, atol=1e-9):
        raise ValueError("The teleoperation arm command rate differs from its analytic settings")
    driver_rate = float(driver["max_joint_rate_deg_s"])
    effective_rate = min(command_rate, np.deg2rad(driver_rate))
    arm_names = {f"Joint{i}_{side}" for side in "LR" for i in range(1, 8)}
    if len(set(names) - arm_names) != 40 or not arm_names.issubset(names):
        raise ValueError("The native model must contain the 14 Tianji and 40 Wuji joints")
    feedforward = {params["side"]: float(params["control.velocity_max_rad_s"])
                   for params in parameter_sets(args.hand_driver)}
    if set(feedforward) != {"left", "right"} or not all(np.isfinite(v) and v > 0 for v in feedforward.values()):
        raise ValueError("The Wuji real driver must describe both independent velocity feedforward limits")
    report.update({
        "profile": "real",
        RATE_FIELD: [effective_rate if name in arm_names else None for name in names],
        "sources": {label: {"path": str(path.resolve()), "sha256": digest(path)}
                    for label, path in (("teleop_base", args.base), ("teleop_overlay", args.overlay),
                                        ("tianji_driver", args.arm_driver), ("wuji_driver", args.hand_driver))},
        "arm": {k.removeprefix("arm."): arm[k] for k in (
            key, "arm.control_rate_hz", "arm.max_raw_joint_step_rad", "arm.max_translation_rate_m_sec",
            "arm.max_rotation_rate_rad_sec", "arm.analytic_joint_vmax_deg_s",
            "arm.analytic_sdk_velocity_ratio", "arm.analytic_command_rate_margin")},
        "arm_base_command_rate_rad_s": float(base[key]),
        "arm_driver_max_deg_s": driver_rate,
        "hand": {"position_envelope": "joint_bounds", "velocity_feedforward_limit_rad_s": feedforward},
    })
    position_rates(names, report)
    write_json(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=(DEXGYM_PROFILE, "real"), default="real")
    for name in ("model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("base", "overlay", "arm-driver", "hand-driver"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.profile == "real" and not all((args.base, args.overlay, args.arm_driver, args.hand_driver)):
        parser.error("The real profile requires the base, overlay and both driver configurations")
    describe_rates(args)
