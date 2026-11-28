"""Compile pinned native hand acceptance and timeout methods for replay."""

import ctypes as ct
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
import yaml

from parity_wuji_probe import REVISION, checked_source, digest


PACKAGE = Path("src/whole_body_mujoco_driver")
BACKEND = PACKAGE / "src/mujoco_backend.cpp"
MODEL_NAMES = PACKAGE / "include/whole_body_mujoco_driver/model_joints.hpp"
CONFIG = PACKAGE / "config/sim.yaml"
DRIVER_KEYS = ("hand_command_timeout_sec", "hand_handoff_hold_sec", "hand_handoff_ramp_sec")
CLOCK_OFFSET_NS = 1_000_000_000


def bounded_piece(text, begin, end):
    if text.count(begin) != 1:
        raise ValueError(f"Pinned native driver source location changed: {begin}")
    return begin + text.split(begin, 1)[1].split(end, 1)[0]


def build_driver(teleop, build):
    sources = {p.as_posix(): checked_source(teleop, p) for p in (BACKEND, MODEL_NAMES, CONFIG)}
    native = sources[BACKEND.as_posix()].decode()
    accept = bounded_piece(native, "CommandResult MujocoBackend::accept_hand(", "\nvoid MujocoBackend::apply_gravity")
    hold = bounded_piece(native, "void MujocoBackend::hold_range_locked(", "\nbool MujocoBackend::all_finite_feedback")
    watchdog = bounded_piece(native, "if(hand_enabled_[side] && hand_home_mode_[side]==0 && steady_ns-hand_command_ns_[side]>",
                             "\n    }") + "\n}"
    files = {"native_hand_accept.inc": accept, "native_hand_hold.inc": hold, "native_hand_watchdog.inc": watchdog}
    config = yaml.safe_load(sources[CONFIG.as_posix()])
    params, = [v["ros__parameters"] for v in config.values() if "ros__parameters" in v]
    timing = tuple(float(params[name]) for name in DRIVER_KEYS)
    if timing != (.1, .5, 1.):
        raise ValueError("Pinned hand timing differs from the registered simulation profile")
    build.mkdir()
    for name, content in files.items():
        (build / name).write_text(content + "\n", encoding="utf-8")
    wrapper = Path(__file__).with_suffix(".cpp")
    library = build / "libnative_hand_driver_probe.so"
    command = ["g++", "-std=c++20", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
               "-I", str(build), "-I", str(teleop / PACKAGE / "include"), str(wrapper), "-o", str(library)]
    subprocess.run(command, check=True)
    return DriverLibrary(library, timing), {
        "native_revision": REVISION, "config": dict(zip(DRIVER_KEYS, timing, strict=True)),
        "source_sha256": {name: sha256(data).hexdigest() for name, data in sources.items()},
        "extracted_sha256": {name: digest(build / name) for name in files},
        "wrapper_sha256": digest(wrapper), "adapter_sha256": digest(Path(__file__)),
        "binary_sha256": digest(library), "compiler_command": command,
    }


class DriverLibrary:
    def __init__(self, path, timing):
        lib = self.lib = ct.CDLL(str(path))
        self.timing = timing
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        lib.s05_driver_alias.argtypes = [ct.c_int]
        lib.s05_driver_alias.restype = ct.c_char_p
        lib.s05_driver_create.argtypes = [doubles, doubles, ct.c_int64, ct.c_double, ct.c_double, ct.c_double]
        lib.s05_driver_create.restype = ct.c_void_p
        lib.s05_driver_destroy.argtypes = [ct.c_void_p]
        lib.s05_driver_destroy.restype = None
        lib.s05_driver_receive.argtypes = [ct.c_void_p, ct.c_int, ct.c_int64, ct.c_int64, ct.c_uint64, doubles]
        lib.s05_driver_receive.restype = ct.c_int
        lib.s05_driver_snapshot.argtypes = [ct.c_void_p, ct.c_int64, doubles, doubles, integers]
        lib.s05_driver_snapshot.restype = None
        self.hand_aliases = [lib.s05_driver_alias(i).decode() for i in range(14, 54)]

    def create(self, q, ctrl, cutoff):
        return Driver(self, q, ctrl, cutoff)


class Driver:
    def __init__(self, api, q, ctrl, cutoff):
        self.lib = api.lib
        seed = np.ascontiguousarray(q, dtype=np.float64)
        ranges = np.ascontiguousarray(ctrl, dtype=np.float64)
        if seed.shape != (54,) or ranges.shape != (54, 2) or not np.isfinite(seed).all() or not np.isfinite(ranges).all():
            raise ValueError("Native driver initialization requires finite signed 54-axis arrays")
        self.handle = self.lib.s05_driver_create(seed, ranges, cutoff + CLOCK_OFFSET_NS, *api.timing)

    def close(self):
        if self.handle:
            self.lib.s05_driver_destroy(self.handle)
            self.handle = None

    def receive(self, side, stamp, received, sequence, q):
        command = np.ascontiguousarray(q, dtype=np.float64)
        if command.shape != (20,) or not np.isfinite(command).all():
            raise ValueError("Published native hand output requires 20 finite axes")
        return bool(self.lib.s05_driver_receive(self.handle, side, stamp + CLOCK_OFFSET_NS,
                                                 received + CLOCK_OFFSET_NS, sequence, command))

    def snapshot(self, cutoff, measured):
        target = np.empty(54, dtype=np.float64)
        fault = np.empty(2, dtype=np.int32)
        state = np.ascontiguousarray(measured, dtype=np.float64)
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError("Native driver requires finite signed measured joint state")
        self.lib.s05_driver_snapshot(self.handle, cutoff + CLOCK_OFFSET_NS, state, target, fault)
        return target, fault.astype(bool)


def controls(api):
    """Run native seam positives and negatives without executing physics."""
    state = np.zeros(54)
    limits = np.repeat([[-1., 1.]], 54, axis=0)
    hand = np.full(20, 2.)
    driver = api.create(state, limits, 0)
    try:
        assert driver.receive(0, 500_000_000, 500_000_000, 1, hand)
        assert np.array_equal(driver.snapshot(500_000_000, state)[0], state)
        assert driver.receive(0, 750_000_000, 750_000_000, 2, hand)
        q, _ = driver.snapshot(750_000_000, state)
        assert np.all(q[14:34] == .15625) and not np.any(q[34:])
        assert not driver.receive(0, 750_000_000, 750_000_000, 2, np.zeros(20))
        assert not driver.receive(0, 750_000_000, 850_000_001, 3, hand)
        assert not driver.receive(0, 850_000_002, 850_000_001, 3, hand)
        measured = np.linspace(0, .2, 54)
        q, fault = driver.snapshot(850_000_001, measured)
        assert fault.all() and np.array_equal(q[14:34], measured[14:34])
        # Right never received a message and had already latched at the first snapshot.
        assert np.array_equal(q[34:], state[34:])
        return {"valid_hold": True, "smoothstep_and_clip": True, "independent_hands": True,
                "duplicate_rejected": True, "stale_rejected": True, "future_rejected": True,
                "timeout_latches_measured_once_per_side": True}
    finally:
        driver.close()
