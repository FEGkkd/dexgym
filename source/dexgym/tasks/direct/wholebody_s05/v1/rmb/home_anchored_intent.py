"""Independent engineering targets for an offline, simulation-only START diagnosis."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ...data.synchronize import STREAMS
from ...p02.intent import hand_basis, semantic_order, unit
from .mapping import packed_geometry


VIRTUAL_WORLD_FROM_OPENVR = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
RAW_TRACKERS = tuple(STREAMS.index(side) for side in ("tracker_left_raw", "tracker_right_raw"))
ANATOMICAL_MOUNT_STATUS = "UNKNOWN"


def _proper(value, shape):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError("Expected finite named SE(3) transforms")
    r = matrix[..., :3, :3]
    if (not np.allclose(matrix[..., 3, :], [0., 0., 0., 1.], atol=1e-12, rtol=0)
            or not np.allclose(np.swapaxes(r, -2, -1) @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-8, rtol=0)):
        raise ValueError("Expected a right-handed unscaled transform")
    return matrix


def _rigid(position, rotation):
    shape = np.asarray(position).shape
    value = np.broadcast_to(np.eye(4), (*shape[:-1], 4, 4)).copy()
    value[..., :3, 3] = position
    value[..., :3, :3] = rotation
    return _proper(value, value.shape)


def _registered_pair(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if (poses.shape != (2, 7) or not np.isfinite(poses).all()
            or np.any(abs(np.linalg.norm(poses[:, 3:], axis=-1) - 1.) > 1e-5)):
        raise ValueError("Invalid signed left/right XYZW wrist poses")
    return _rigid(poses[:, :3], Rotation.from_quat(poses[:, 3:]).as_matrix())


def lock_registered_start(registered_start, robot_home_reference, *, align_yaw=False):
    """Use one simulation-world rigid map for each independently chosen START.

    Only the labeled left-to-right baseline determines optional yaw. Neither
    tracking, dynamic feedback nor policy output can select or update the map.
    """
    start = _proper(registered_start, (2, 4, 4))[:, :3, 3]
    home = _proper(robot_home_reference, (2, 4, 4))[:, :3, 3]
    rotation = VIRTUAL_WORLD_FROM_OPENVR
    if align_yaw:
        human_line = (rotation @ (start[1] - start[0]))[:2]
        home_line = (home[1] - home[0])[:2]
        if min(np.linalg.norm(human_line), np.linalg.norm(home_line)) < .001:
            raise ValueError("Yaw undefined for the labeled horizontal START baseline")
        cross = human_line[0] * home_line[1] - human_line[1] * home_line[0]
        yaw = np.arctan2(cross, human_line @ home_line)
        rotation = Rotation.from_rotvec([0., 0., yaw]).as_matrix() @ rotation
    return _rigid(home.mean(axis=0) - rotation @ start.mean(axis=0), rotation)


@dataclass(frozen=True)
class SignedHumanTick:
    raw_tracker: np.ndarray | None
    registered_wrist: np.ndarray | None
    manus_local: tuple


class SignedHumanPartition:
    """Extract original Tracker and MANUS layers from a signed HMB partition."""

    def __init__(self, data, metadata, sources):
        self.data = data
        self.sources = sources
        self.topology = {(item["stream"], item["version"]): semantic_order(item)
                         for item in metadata["topologies"]}

    def _source(self, tick, side):
        i = int(self.data["source_index"][tick, side, 0])
        if not 0 <= i < len(self.sources):
            raise ValueError("Valid HMB pose has no signed source")
        return self.sources[i]

    def _tracker(self, tick, sides):
        data = self.data
        if not (data["pose_valid"][tick, sides].all() and data["node_mask"][tick, sides, 0].all()):
            return None
        if any(self._source(tick, side)["recorded_frame"] != "openvr_world" for side in sides):
            return None
        return _registered_pair(data["pose"][tick, sides, 0])

    def __getitem__(self, tick):
        data = self.data
        hands = []
        for side in range(2):
            value = None
            if data["pose_valid"][tick, side]:
                source = self._source(tick, side)
                root, chain = self.topology[source["stream"], source["topology_id"]]
                mask = data["node_mask"][tick, side]
                if mask[root] and mask[chain].all():
                    pose = data["pose"][tick, side]
                    value = pose[chain, :3] - pose[root, :3]
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite named MANUS landmark")
            hands.append(value)
        return SignedHumanTick(self._tracker(tick, RAW_TRACKERS), self._tracker(tick, (2, 3)), tuple(hands))


class SimulationIntent:
    """Map registered wrists and retain a separate robot-morphology target.

    The optional engineering attachment maps the model palm coordinates into
    the registered wrist proxy. Without that scoped assumption, the world's
    fingertips and TCP target are unavailable. The raw geometry still exists.
    """

    def __init__(self, world_from_recording, robot_home_geometry, *, proxy_from_model_palm=None):
        self.world_from_recording = _proper(world_from_recording, (4, 4)).copy()
        g = robot_home_geometry
        home_model_palm = _rigid(g.palm, g.palm_rotation)
        home_scoring_tcp = _rigid(g.wrists, g.rotations)
        self.model_palm_from_scoring_tcp = np.linalg.inv(home_model_palm) @ home_scoring_tcp
        if proxy_from_model_palm is None:
            self.proxy_from_model_palm = None
        else:
            self.proxy_from_model_palm = _proper(proxy_from_model_palm, (2, 4, 4)).copy()
        local = np.asarray(g.local, dtype=np.float64)
        if local.shape != (2, 5, 4, 3) or not np.isfinite(local).all():
            raise ValueError("Expected named 2×5×4 model Home landmarks")
        self.origins = local[:, :, :1].copy()
        self.bone_lengths = np.linalg.norm(np.diff(local, axis=2), axis=-1)
        if np.any(self.bone_lengths < 1e-7):
            raise ValueError("Degenerate model Home bones")
        self.basis = np.stack([hand_basis(knuckles[1:, 0]) for knuckles in self.origins])

    def _local_shape(self, side, human):
        if human is None:
            return None
        human = np.asarray(human, dtype=np.float64)
        if human.shape != (5, 4, 3) or not np.isfinite(human).all():
            raise ValueError("Expected five finite named four-landmark MANUS chains")
        try:
            human_to_model = self.basis[side] @ hand_basis(human[1:, 0]).T
            directions = unit(np.diff(human, axis=1)) @ human_to_model.T
        except ValueError:
            return None
        origins = self.origins[side]
        return np.concatenate((origins, origins + np.cumsum(
            directions * self.bone_lengths[side, ..., None], axis=1)), axis=1)

    def map(self, row):
        registered = row.registered_wrist
        mapped = None if registered is None else self.world_from_recording @ _proper(registered, (2, 4, 4))
        local_shape = tuple(self._local_shape(side, hand) for side, hand in enumerate(row.manus_local))
        tcp = morphology = None
        if mapped is not None and self.proxy_from_model_palm is not None:
            palm = mapped @ self.proxy_from_model_palm
            tcp = palm @ self.model_palm_from_scoring_tcp
            if all(side is not None for side in local_shape):
                local = np.stack(local_shape)
                world = np.einsum("sij,sfkj->sfki", palm[:, :3, :3], local) + palm[:, None, None, :3, 3]
                morphology = packed_geometry(tcp[:, :3, 3], tcp[:, :3, :3], local, world)
        return {"raw_tracker": row.raw_tracker, "raw_manus_local": row.manus_local,
                "registered_wrist_proxy": registered, "sim_mapped_wrist_proxy": mapped,
                "simulation_scoring_tcp": tcp, "robot_morphology_local": local_shape,
                "robot_morphology_target": morphology, "anatomical_mount_status": ANATOMICAL_MOUNT_STATUS,
                "physical_calibration_qualified": False}
