"""Independent, signed P02 geometry. It never accepts nominal or measured joints."""
import numpy as np
from scipy.spatial.transform import Rotation

from ...p02.intent import CHAINS, hand_basis, semantic_order, unit


PINCH_THRESHOLD_M = 0.020
SMOOTH_BANDS_M = ((0.018, 0.022), (0.017, 0.023), (0.016, 0.024))


def canonical_xyzw(matrix):
    flat = np.asarray(matrix, dtype=float).reshape(-1, 3, 3)
    quat = Rotation.from_matrix(flat).as_quat(canonical=True)
    return quat.reshape(*np.asarray(matrix).shape[:-2], 4)


def packed_geometry(wrist, rotation, local, world):
    tips = world[:, :, -1]
    left_from_world = rotation[0].T
    return {
        "wrist_xyz": wrist,
        "wrist_xyzw": canonical_xyzw(rotation),
        "local_landmark_xyz": local,
        "world_tip_xyz": tips,
        "pinch_xyz": local[:, 1:, -1] - local[:, :1, -1],
        "cross_xyz": tips[1, None, :, :] - tips[0, :, None, :],
        "relative_wrist_xyz": left_from_world @ (wrist[1] - wrist[0]),
        "relative_wrist_xyzw": canonical_xyzw(left_from_world @ rotation[1]),
    }


def pinch_points(points, desired, weights):
    """P02's least squares endpoint displacement, applied only to finger shape."""
    tips = points[:, -1]
    residual = desired - (tips[1:] - tips[0])
    thumb = -(weights[:, None] * residual).sum(axis=0) / (1 + weights.sum())
    fingers = weights[:, None] * (residual + thumb)
    shifts = np.vstack((thumb, fingers))
    return points + shifts[:, None, :] * np.array([0.0, 1 / 3, 2 / 3, 1.0])[None, :, None]


def smooth_weights(distance, band):
    low, high = band
    u = np.clip((high - distance) / (high - low), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def order_for_sdk(topology):
    """Use the production RetargetNode's semantic sequence when IP is explicit."""
    root, _ = semantic_order(topology)
    result = [root]
    for chain in CHAINS:
        by_type = {node["joint_type"].upper(): node["original_order"]
                   for node in topology["joints"] if node["chain_type"] == chain}
        names = ("PIP", "IP", "DIP", "TIP") if chain != "Thumb" and "IP" in by_type else (
            "MCP", "PIP", "DIP", "TIP")
        result.extend(by_type[name] for name in names)
    return np.asarray(result, dtype=int)


class IntentMapper:
    def __init__(self, home_geometry, calibrated, candidates, *, smooth_band=None):
        if candidates["fit_scope"] != "Train" or candidates["configs"]["M1"]["scale"] != 1:
            raise ValueError("M1 was not fitted on the frozen Train-only, unit-scale cohort")
        if smooth_band is not None and tuple(smooth_band) not in SMOOTH_BANDS_M:
            raise ValueError("Requested smoothing band is outside the three authorized candidates")
        self.configs = candidates["configs"]
        self.mount_p = np.array(calibrated["wrist_to_palm_translation"])
        self.mount_r = np.array(calibrated["wrist_to_palm_rotation"])
        self.hand_scale = np.array(calibrated["hand_scale"])
        self.origins = home_geometry.local[:, :, 0]
        self.lengths = np.linalg.norm(np.diff(home_geometry.local, axis=2), axis=-1)
        self.basis = np.stack([hand_basis(knuckles[1:]) for knuckles in self.origins])
        self.smooth_band = smooth_band

    def place(self, name, raw_wrist, local):
        cfg = self.configs[name]
        world_r = np.array(cfg["rotation"])
        world_t = np.array(cfg["translation"])
        mount = np.array(cfg["mount"])
        pivot = np.array(cfg["pivot"])
        wrist = (pivot + (raw_wrist[:, :3] - pivot) * cfg["scale"]) @ world_r.T + world_t
        rotation = world_r @ Rotation.from_quat(raw_wrist[:, 3:]).as_matrix() @ mount
        palm = wrist + np.einsum("sij,sj->si", rotation, self.mount_p)
        world = np.einsum("sij,sfkj->sfki", rotation @ self.mount_r, local) + palm[:, None, None, :]
        return packed_geometry(wrist, rotation, local, world)

    def map(self, raw_wrist, palms, order):
        hard, before, selected = [], [], []
        activation, desired_pinch = [], []
        for side, ((root, indices), points) in enumerate(zip(order, palms, strict=True)):
            human = points[indices, :3] - points[root, :3]
            transform = self.basis[side] @ hand_basis(human[1:, 0]).T
            direction = unit(np.diff(human, axis=1)) @ transform.T
            origins = self.origins[side, :, None]
            shape = np.concatenate(
                (origins, origins + np.cumsum(direction * self.lengths[side, ..., None], axis=1)), axis=1
            )
            raw = human[1:, -1] - human[0, -1]
            distance = np.linalg.norm(raw, axis=-1)
            active = distance <= PINCH_THRESHOLD_M
            wanted = (raw @ transform.T) * self.hand_scale[side]
            hard_points = pinch_points(shape, wanted, active.astype(float))
            hard.append(hard_points)
            before.append(shape)
            selected.append(hard_points if self.smooth_band is None else
                            pinch_points(shape, wanted, smooth_weights(distance, self.smooth_band)))
            activation.append(active)
            desired_pinch.append(wanted)
        return {
            "mapped": self.place("M1", raw_wrist, np.stack(selected)),
            "hard_m1_audit": self.place("M1", raw_wrist, np.stack(hard)),
            "M0_audit": self.place("M0", raw_wrist, np.stack(hard)),
            "unadjusted": self.place("M1", raw_wrist, np.stack(before)),
            "desired_pinch": np.asarray(desired_pinch),
            "pinch_active": np.asarray(activation),
        }


def compiled_topologies(meta, topology_codes):
    topologies = {(record["stream"], record["version"]): record for record in meta["topologies"]}
    versions = meta["dictionary"]["topology_id"]
    selected = [topologies[(f"manus_{side}", versions[int(code)])]
                for side, code in zip(("left", "right"), topology_codes, strict=True)]
    return tuple(map(semantic_order, selected)), tuple(map(order_for_sdk, selected))


def sdk_keypoints(common, indices, side):
    xyzw = common[indices, :3]
    deg = 15.0 if side == 0 else -15.0
    turn = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(deg)])
    return np.ascontiguousarray(turn.apply(xyzw), dtype=np.float32)
