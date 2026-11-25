"""Shared read-only task-space sequences and an explicitly separate RSI state."""

import json
from pathlib import Path

import numpy as np
import torch

from ..hand2_cube_supermimic.rotations import mat_to_quat, quat_to_mat
from ..wholebody_s0.bag import digest
from . import DT, SCHEMA
from .finger_reward import BILATERAL_NEAR_M, bilateral_tip_vectors
from .kinematics import MAPPED_HAND, MAPPED_FOUR_POINT, MIXED_HAND_GOALS, REFERENCE_HAND_FK, Pose, make_pose


def recorded_frame_steps(time_ns, source_time_ns):
    """Approximate the original 50 Hz clock using the already retargeted poses."""
    period = round(DT * 1_000_000_000)
    if (len(time_ns) < 2 or source_time_ns.shape != time_ns.shape
            or not np.all(np.diff(time_ns) == period)
            or not np.all(np.diff(source_time_ns) > 0)):
        raise ValueError("Invalid recording-to-reference timestamp map")
    stamp = source_time_ns - source_time_ns[0]
    frame = np.arange(len(stamp))
    # Repeatedly querying one period from a quantized sample biases the
    # playback rate. Snap to an absolute grid before advancing its clock.
    next_tick = (stamp + (period - 1) // 2) // period + 1
    target = next_tick * period
    after = np.searchsorted(stamp, target).clip(max=len(stamp) - 1)
    before = (after - 1).clip(min=0)
    nearest = np.where(np.abs(target - stamp[before]) <= np.abs(stamp[after] - target), before, after)
    next_frame = np.clip(nearest, np.minimum(frame + 1, len(stamp) - 1),
                         np.minimum(frame + 3, len(stamp) - 1))
    remaining = np.zeros(len(stamp), dtype=np.int32)
    for i in range(len(stamp) - 2, -1, -1):
        remaining[i] = 1 + remaining[next_frame[i]]
    return next_frame, remaining


class TaskBank:
    def __init__(self, manifest, split, device, *, goals_only=False,
                 mapped_hand_clip_fraction=0., hand_style_seed=0):
        if not 0. <= mapped_hand_clip_fraction <= 1. or (mapped_hand_clip_fraction and split != "train"):
            raise ValueError("Mapped hand mixtures require a training fraction in [0,1]")
        self.mapped_hand_clip_fraction = mapped_hand_clip_fraction
        self.hand_style_seed = hand_style_seed
        self.training_hand_mix = None
        self.mapped_hand_frames = None
        self.opposition_sampling = None
        manifest = Path(manifest)
        self.metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if self.metadata["schema"] != SCHEMA or self.metadata["rate_hz"] != round(1 / DT):
            raise ValueError("Unexpected task-space library schema or clock")
        info = self.metadata["splits"][split]
        path = manifest.parent / info["file"]
        if digest(path) != info["sha256"]:
            raise ValueError(f"Bank checksum differs: {path}")
        self.split = split
        self.hand_target_mode = MAPPED_HAND
        self.clips = info["clips"]
        starts = np.asarray([row["start"] for row in self.clips])
        stops = np.asarray([row["stop"] for row in self.clips])
        if starts[0] != 0 or stops[-1] != info["frames"] or np.any(starts[1:] != stops[:-1]):
            raise ValueError("The bank has discontinuous clip offsets")
        with np.load(path, allow_pickle=False) as data:
            self.pose = Pose(*(torch.from_numpy(data[k]).to(device) for k in Pose._fields))
            if not goals_only:
                self.rsi_q = torch.from_numpy(data["rsi_q"]).to(device)
                self.rsi_qd = torch.from_numpy(data["rsi_qd"]).to(device)
                self.reset_depth = torch.from_numpy(data["reset_depth_m"]).to(device)
                self.safe = torch.from_numpy(data["reset_safe"]).to(device)
            self.time_ns = data["time_ns"]
        lengths = stops - starts
        self.clip_starts = torch.as_tensor(starts, device=device)
        self.clip_lengths = torch.as_tensor(lengths, device=device)
        self.clip_id = torch.as_tensor(np.repeat(np.arange(len(lengths)), lengths), device=device)
        self.end = torch.from_numpy(np.repeat(stops - 1, lengths)).to(device)
        names = ["teleop", "egopro"]
        source_index = [names.index(row.get("source", "teleop")) for row in self.clips]
        self.egopro = torch.from_numpy(np.repeat(source_index, lengths) == 1).to(device)
        self.source_weights = self.metadata.get("train_source_weights") if split == "train" else None
        if not goals_only:
            frame = torch.arange(info["frames"], device=device)
            self.eligible = (self.safe & (self.end - frame >= 24)).nonzero().flatten()
            if self.source_weights is not None:
                if set(self.source_weights) != set(names) or not np.isclose(sum(self.source_weights.values()), 1.):
                    raise ValueError("Training source weights must cover the two source domains and sum to one")
                self.by_source = (self.eligible[~self.egopro[self.eligible]], self.eligible[self.egopro[self.eligible]])
                if any(len(values) == 0 for values in self.by_source):
                    raise ValueError("Expected safe RSI frames from both training sources")

    def prepare_recorded_clock(self):
        if self.split != "train":
            raise ValueError("Recording tempo augmentation requires the training split")
        size = len(self.time_ns)
        frame = np.arange(size)
        ending = self.end.cpu().numpy()
        steps = np.minimum(frame + 1, ending)
        remaining = (ending - frame).astype(np.int32)
        retimed = np.zeros(size, dtype=np.bool_)
        for row in self.clips:
            if row.get("timing") != "retimed":
                continue
            start, stop = row["start"], row["stop"]
            path = Path(row["reference_file"])
            if digest(path) != row["reference_sha256"]:
                raise ValueError(f"Recording tempo map checksum differs: {path}")
            with np.load(path, allow_pickle=False) as ref:
                if not np.array_equal(ref["time_ns"], self.time_ns[start:stop]):
                    raise ValueError(f"Recording tempo map time axis differs: {path}")
                local, steps_left = recorded_frame_steps(ref["time_ns"], ref["source_time_ns"])
            steps[start:stop] = start + local
            remaining[start:stop] = steps_left
            retimed[start:stop] = steps_left < np.arange(stop - start - 1, -1, -1)
        self.recorded_next = torch.as_tensor(steps, device=self.end.device)
        self.recorded_remaining = torch.as_tensor(remaining, device=self.end.device)
        self.recorded_eligible = torch.as_tensor(retimed, device=self.end.device) & (self.recorded_remaining >= 24)

    def advance_frame(self, frame, steps, recorded=None, *, max_steps=None):
        regular = torch.minimum(frame + steps, self.end[frame])
        if recorded is None:
            return regular
        result = frame
        limit = steps if isinstance(steps, int) else max_steps
        for step in range(limit):
            next_frame = self.recorded_next[result]
            result = next_frame if isinstance(steps, int) else torch.where(steps > step, next_frame, result)
        return torch.where(recorded, result, regular)

    @torch.no_grad()
    def set_reference_hand_targets(self, fk, reference_q=None):
        """Use all four FK landmarks of the archived RSI pose in the same wrist frame."""
        if fk.anchor_count != 3:
            raise ValueError("Reference hand FK requires proximal, middle and distal anchors")
        q = self.rsi_q if reference_q is None else reference_q
        hand = q.new_empty((len(q), 2, 5, 4, 3))
        for start in range(0, len(hand), 4096):
            end = min(start + 4096, len(hand))
            hand[start:end] = fk(q[start:end]).pose.hand
        self.hand_target_mode = REFERENCE_HAND_FK
        if self.mapped_hand_clip_fraction:
            self.mix_mapped_hand_clips(hand, fk)
            self.hand_target_mode = MIXED_HAND_GOALS
        self.pose = self.pose._replace(hand=hand)

    def mix_mapped_hand_clips(self, hand, fk):
        """Use paired online-style hand geometry on entire training clips."""
        input_hand = self.pose.hand
        if input_hand.shape[-2] != 3 or len(input_hand) != len(hand):
            raise ValueError("Mixed targets need mapped bone endpoints for every reference frame")
        rng = np.random.default_rng(self.hand_style_seed)
        mix = {}
        self.mapped_hand_frames = torch.zeros(len(hand), dtype=torch.bool, device=hand.device)
        home = hand.new_tensor(self.metadata["home"])[None]
        roots = fk(home).pose.hand[..., :1, :]
        for source in ("teleop", "egopro"):
            clips = [c for c in self.clips if c.get("source", "teleop") == source]
            if not clips:
                continue
            count = round(self.mapped_hand_clip_fraction * len(clips))
            ids = rng.choice(len(clips), size=count, replace=False)
            mapped_frames = 0
            for i in sorted(ids):
                row = clips[i]
                a, b = row["start"], row["stop"]
                hand[a:b, ..., 1:, :].copy_(input_hand[a:b])
                hand[a:b, ..., :1, :].copy_(roots)
                self.mapped_hand_frames[a:b] = True
                mapped_frames += b - a
            mix[source] = {"mapped_clips": count, "total_clips": len(clips),
                           "mapped_frames": mapped_frames,
                           "total_frames": sum(c["stop"] - c["start"] for c in clips)}
        self.training_hand_mix = mix

    @torch.no_grad()
    def set_mapped_hand_targets(self, fk):
        """Preserve mapped bone endpoints; append the same fixed roots as online."""
        if self.hand_target_mode == MAPPED_FOUR_POINT:
            return
        if self.hand_target_mode != MAPPED_HAND or self.pose.hand.shape[-2] != 3:
            raise ValueError("Mapped four-point targets require original three-point goals")
        home = self.pose.hand.new_tensor(self.metadata["home"])[None]
        roots = fk(home).pose.hand[..., :1, :].expand(len(self.pose.hand), -1, -1, -1, -1)
        self.pose = self.pose._replace(hand=torch.cat((roots, self.pose.hand), dim=-2))
        self.hand_target_mode = MAPPED_FOUR_POINT

    def at(self, index):
        return Pose(*(value[index] for value in self.pose))

    @torch.no_grad()
    def prepare_opposition_sampling(self, fraction):
        if self.split != "train" or self.hand_target_mode != REFERENCE_HAND_FK or not 0. < fraction <= 1.:
            raise ValueError("Opposition sampling requires the FK training bank and a fraction in (0,1]")
        candidates = []
        for frames in self.eligible.split(4096):
            distance = bilateral_tip_vectors(self.at(frames)).norm(dim=-1)
            nearby = distance.amin(-1) <= BILATERAL_NEAR_M
            candidates.append(frames[nearby])
        candidates = torch.cat(candidates)
        self.opposition_pools = (candidates[~self.egopro[candidates]], candidates[self.egopro[candidates]])
        self.opposition_sampling = {
            "fraction": fraction,
            "teleop_eligible_frames": len(self.opposition_pools[0]),
            "egopro_eligible_frames": len(self.opposition_pools[1]),
        }
        self.opposition_pool = candidates

    def sample_pool(self, pool, count, generator, opposition=None):
        selected = pool[torch.randint(len(pool), (count,), device=self.end.device, generator=generator)]
        if opposition is None or not len(opposition):
            return selected
        focused = opposition[torch.randint(len(opposition), (count,), device=self.end.device, generator=generator)]
        choose = torch.rand(count, device=self.end.device, generator=generator) < self.opposition_sampling["fraction"]
        return torch.where(choose, focused, selected)

    def sample(self, count, generator):
        if self.source_weights is not None:
            choose_ego = torch.rand(count, device=self.end.device, generator=generator) < self.source_weights["egopro"]
            pools = self.opposition_pools if self.opposition_sampling else (None, None)
            teleop, ego = (self.sample_pool(pool, count, generator, focused)
                           for pool, focused in zip(self.by_source, pools, strict=True))
            return torch.where(choose_ego, ego, teleop)
        focused = self.opposition_pool if self.opposition_sampling else None
        return self.sample_pool(self.eligible, count, generator, focused)


def bridge_pose(home, goal, fraction):
    """Interpolate only task-space coordinates from Home into a recorded clip."""
    w = fraction.square() * (3. - 2. * fraction)
    q0, q1 = mat_to_quat(home.rotation), mat_to_quat(goal.rotation)
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
    return make_pose(
        torch.lerp(home.wrist, goal.wrist, w[:, None, None]),
        quat_to_mat(torch.lerp(q0, q1, w[:, None, None])),
        torch.lerp(home.hand, goal.hand, w[:, None, None, None, None]),
    )
