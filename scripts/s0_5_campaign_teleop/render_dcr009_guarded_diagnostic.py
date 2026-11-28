#!/usr/bin/env python3
"""Create a three-panel MP4 from one guarded DCR-009 diagnostic trace and signed HMB."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from compare_start_handoff import ordered_q, valid_feedback
from dexgym.tasks.direct.wholebody_s05.p02.intent import semantic_order
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from hmb_cohort import IDENTITIES
from pair_nominal_hmb import signed_feedback_names
from replay_dcr009_guarded_diagnostic import ROOT_FIVE, align_pair_yaw
from summarize_nominal_replay import ACTIVE, FEEDBACK, HISTORICAL, stamp


RUN_ID = "HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
FRAME_SIZE = (1440, 640)
PREPOSITION = "preposition_start"
MAX_AGE_NS = 40_000_000
SIDES = ("#1775aa", "#a74c27")
INK, MUTED, GRID = "#20344c", "#54677c", "#dce5ef"
AXIS = ("#b73b4a", "#447957", "#5968a3")
VIRTUAL_R = np.asarray([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checked_inputs(attempt, plan_path, cohort_path, model_path):
    result = json.loads((attempt / "result.json").read_bytes())
    receipt = json.loads((attempt / "evidence" / "input.json").read_bytes())
    trace = attempt / "evidence" / "trace.jsonl"
    if not trace.is_file():
        trace = trace.with_suffix(".jsonl.gz")
    plan = json.loads(plan_path.read_bytes())
    if (plan["run_id"] != RUN_ID or plan["scope"] != "DIAGNOSTIC_ONLY" or plan["stage_credit"] is not False
            or receipt["scope"] != "DIAGNOSTIC_ONLY" or receipt["plan_sha256"] != digest(plan_path)
            or plan["cohort_sha256"] != digest(cohort_path) or receipt["cohort_receipt_sha256"] != plan["cohort_sha256"]):
        raise ValueError("Attempt, plan, cohort or diagnostic identity differs")
    expected_trace = result["trace_sha256"]
    if trace.suffix == ".gz":
        with gzip.open(trace, "rb") as stream:
            measured_sha = file_digest(stream, "sha256").hexdigest()
    else:
        measured_sha = digest(trace)
    if measured_sha != expected_trace:
        raise ValueError("Attempt result does not bind exact raw trace contents")
    root_id = result["root"]
    choices = [entry for entry in plan["selected_roots"] if entry["root"] == root_id and entry["split"] == "Train"]
    if len(choices) != 1 or receipt["root"]["root"] != root_id:
        raise ValueError("Attempt does not bind a single selected Train root")
    anchor_path = (plan_path.parent / choices[0]["anchor_path"]).resolve()
    if plan_path.parent.resolve() not in anchor_path.parents or digest(anchor_path) != choices[0]["anchor_sha256"]:
        raise ValueError("Signed diagnostic anchor identity differs")
    anchor = json.loads(anchor_path.read_bytes())
    if (anchor["root"] != root_id or anchor["anatomical_start_h"] is not None
            or anchor["scope"] != "DIAGNOSTIC_ONLY" or anchor["historical_window"] != receipt["historical_window"]
            or receipt["shared_frames_sha256"] != anchor["shared_frames_sha256"]
            or anchor["shared_frames_sha256"] != plan["shared_frames_sha256"]):
        raise ValueError("Signed START, window or frames disagree")
    return result, receipt, trace, anchor_path, anchor, digest(model_path)


def trace_data(path, model_sha, anchor, yaw_alignment=False):
    groups = defaultdict(dict)
    times, models, latched, outcomes, preposition_clocks = [], [], [], [], []
    stages, statuses, measurements, history_input = {}, [], [], []
    previous = -1
    phase_active = False
    measured_home = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete raw trace row")
            row = json.loads(line)
            stage, kind, mono = row["stage"], row["kind"], row["mono_ns"]
            if mono < previous:
                raise ValueError("Trace receipt time runs backwards")
            previous = mono
            if kind == "model":
                models.append(row["sha256"])
            elif kind == "timebase":
                times.append(row)
            elif kind == "start_anchor":
                latched.append(row)
            elif kind == "home_feedback":
                measured_home = row["anatomical_palm_world"]
            elif kind == "start_handoff":
                outcomes.append(row)
            elif kind == "start_measurement":
                measurements.append((mono, row.get("ready")))
            elif kind == "preposition_clock":
                preposition_clocks.append(mono)
            elif kind == "pilot_lifecycle":
                statuses.append((mono, row["status"]))
            elif kind == "input" and stage == HISTORICAL:
                history_input.append(row["original_schedule_ns"])
            if kind != "output":
                continue
            topic, value = row["topic"], row["value"]
            if topic in ACTIVE:
                key, expected = ACTIVE[topic]
                stages[topic] = value.get(key) == expected, value.get("session_id")
                session = latched[0].get("session") if latched else None
                phase_active = bool(session) and all(stages.get(name) == (True, session) for name in ACTIVE)
            if topic not in FEEDBACK or (stage == HISTORICAL and not phase_active):
                continue
            header = stamp(value.get("header", {}).get("stamp"))
            if header is None:
                continue
            group = groups[stage, header]
            if topic not in group:
                group[topic] = (mono, valid_feedback(topic, value), None)
            elif group[topic][2] is None:
                seen, q, _ = group[topic]
                group[topic] = (seen, q, mono)
    if (models != [model_sha] or len(latched) > 1 or len(times) > 1 or len(outcomes) > 1
            or len(preposition_clocks) > 1):
        raise ValueError("Trace model or singleton START/timebase proof differs")
    if history_input and (not times or not latched or not outcomes or outcomes[0]["status"] != "START_REACHED"
                          or min(history_input) < anchor["source_cutoff_ns"]
                          or times[0]["origin_schedule_ns"] != anchor["source_cutoff_ns"]):
        raise ValueError("Historical data is outside the accepted native START or signed timebase")
    locked = None
    if latched:
        observed = latched[0]
        locked = np.asarray(observed["shared_transform"], dtype=float)
        expected = np.asarray(anchor["registered_wrist_start_h"], dtype=float)
        rotation_matches = np.array_equal(locked[:3, :3], VIRTUAL_R)
        if yaw_alignment:
            if history_input or anchor["root"] != ROOT_FIVE or measured_home is None:
                raise ValueError("Yaw rendering requires the root-five START-only scope and measured Home")
            original = np.eye(4)[None]
            original[0, :3, :3] = VIRTUAL_R
            home = np.asarray(measured_home)
            original[0, :3, 3] = home[:, :3, 3].mean(axis=0) - VIRTUAL_R @ expected[:, :3, 3].mean(axis=0)
            aligned, _, yaw = align_pair_yaw(original, home, original[0] @ expected)
            if (not np.allclose(locked, aligned[0], atol=1e-10, rtol=0)
                    or abs(yaw["yaw_deg"] - observed["yaw_alignment"]["yaw_deg"]) > 1e-9):
                raise ValueError("Yaw overlay does not match the START target derived from the measured Home")
            rotation_matches = True
        if (observed["source_cutoff_ns"] != anchor["source_cutoff_ns"] or observed["inputs"] != anchor["inputs"]
                or locked.shape != (4, 4) or not np.isfinite(locked).all()
                or not rotation_matches
                or not np.array_equal(locked[3], [0, 0, 0, 1])
                or not np.allclose(locked @ expected, observed["start_world_registered_wrist"], atol=1e-5, rtol=0)):
            raise ValueError("Native locked diagnostic target differs from the sealed registered wrist")
    frames = defaultdict(list)
    for (stage, header), group in groups.items():
        if group.keys() != FEEDBACK.keys() or any(q is None for _, q, _ in group.values()):
            continue
        named = {name: angle for _, q, _ in group.values() for name, angle in q.items()}
        duplicate = min((other for _, _, other in group.values() if other is not None), default=None)
        frames[stage].append((max(at for at, _, _ in group.values()), header, named, duplicate))
    for current in frames.values():
        current.sort(key=lambda item: item[:2])
    return {"frames": frames, "locked": locked, "timebase": times[0] if times else None,
            "history_end": max(history_input, default=None), "measurements": measurements, "lifecycle": statuses,
            "outcome": outcomes[0] if outcomes else {}, "last_observer_ns": previous,
            "preposition_clock_ns": preposition_clocks[0] if preposition_clocks else None}


def feedback_at(frames, receipts, cutoff, ros_cutoff, names):
    stop = bisect_right(receipts, cutoff)
    start = bisect_right(receipts, cutoff - MAX_AGE_NS - 1, hi=stop)
    candidates = [row for row in frames[start:stop] if (row[3] is None or cutoff < row[3])
                  and (ros_cutoff is None or 0 <= ros_cutoff - row[1] <= MAX_AGE_NS)]
    if not candidates:
        return None
    arrival, header, q, _ = max(candidates, key=lambda row: (row[1], row[0]))
    return arrival, header, ordered_q(q, names)


def wrists_from_xyzw(value):
    if value.shape != (2, 7) or not np.isfinite(value).all() or np.any(abs(np.linalg.norm(value[:, 3:], axis=1)-1) > 1e-5):
        raise ValueError("Invalid original HMB registered pose")
    result = np.repeat(np.eye(4)[None], 2, axis=0)
    result[:, :3, 3] = value[:, :3]
    result[:, :3, :3] = Rotation.from_quat(value[:, 3:]).as_matrix()
    return result


def human_rows(cohort, docs, products, root, anchor, end_hmb):
    from dexgym.tasks.direct.wholebody_s05.hmb_reader import HMBReader, SIMULATED

    evidence = docs / "evidence"
    reader = HMBReader(evidence / IDENTITIES["hmb_manifest"][0], evidence / IDENTITIES["plan"][0],
                       evidence / IDENTITIES["config"][0], products or cohort["signed_hmb_ubuntu_products"],
                       "Train", SIMULATED)
    start_hmb = anchor["hmb_cutoff_ns"]
    samples = {}
    started = False
    for part in root["hmb_partitions"]:
        if not started:
            started = part["key"] == anchor["partition"]
        if not started:
            continue
        raw, meta, sources, _ = reader.read(part["key"])
        identity = meta["identity"]
        ticks = raw["tick_ns"]
        if (identity["root_episode_id"] != root["root"] or identity["source_manifest_sha256"] != root["manifest_sha256"]
                or identity["development_split"] != "Train" or len(ticks) != part["planned_ticks"]
                or not np.array_equal(ticks, raw["cutoff_ns"])):
            raise ValueError("HMB data do not bind the original signed timetable")
        if ticks[0] > end_hmb:
            break
        topologies = {(entry["stream"], entry["version"]): semantic_order(entry) for entry in meta["topologies"]}
        for i in np.flatnonzero((ticks >= start_hmb) & (ticks <= end_hmb)):
            time = int(ticks[i])
            wrists, fingers = None, [None, None]
            if raw["joint_valid"][i] and raw["safety_permission"][i] and raw["pose_valid"][i, :4].all():
                wrists = wrists_from_xyzw(raw["pose"][i, 2:4, 0])
                for side in range(2):
                    source = sources[raw["source_index"][i, side, 0]]
                    origin, chains = topologies[source["stream"], source["topology_id"]]
                    pose = raw["pose"][i, side]
                    if raw["node_mask"][i, side, origin] and raw["node_mask"][i, side, chains].all():
                        fingers[side] = pose[chains, :3] - pose[origin, :3]
            if time in samples:
                raise ValueError("Duplicate original HMB control cutoff")
            samples[time] = wrists, fingers, part["key"], int(i)
    original = samples.get(start_hmb)
    if (original is None or original[0] is None or original[2:] != (anchor["partition"], anchor["partition_tick"])
            or not np.allclose(original[0], anchor["registered_wrist_start_h"], atol=1e-5, rtol=0)):
        raise ValueError("Cannot find exact signed registered wrist anchor")
    return samples


def render_rows(trace, human, anchor, names, fps, max_seconds, preposition_from_seconds=None):
    historical = trace["history_end"] is not None and preposition_from_seconds is None
    stage = HISTORICAL if historical else PREPOSITION
    frames = trace["frames"][stage]
    if not historical and not frames:
        stage = "setup"
        frames = trace["frames"][stage]
    if not frames:
        raise ValueError("No same-stamp 54-axis feedback in the recorded display stage")
    arrivals = [row[0] for row in frames]
    step = round(1e9 / fps)
    hmb_start = anchor["hmb_cutoff_ns"]
    base = trace["timebase"]
    ticks = sorted(human)
    outcome = trace["outcome"]
    if not outcome and any(status == "FAULT_STOP" for _, status in trace["lifecycle"]):
        outcome = {"status": "PROTECTIVE_STOP_NO_HOME_TO_START_HANDOFF"}
    if preposition_from_seconds is not None:
        zero = trace["preposition_clock_ns"]
        if zero is None or stage != PREPOSITION:
            raise ValueError("Time-aligned preposition clip requires the common start clock and preposition feedback")
        origin = zero + round(preposition_from_seconds * 1e9)
        stop = min(zero + round((preposition_from_seconds + max_seconds) * 1e9), arrivals[-1])
    elif historical:
        zero = origin = base["origin_mono_ns"]
        last = min(trace["history_end"], anchor["historical_window"]["source_cutoff_stop_exclusive_ns"] - 1)
        stop = min(origin + round(max_seconds * 1e9), origin + last - base["origin_schedule_ns"])
    else:
        zero = arrivals[0]
        if stage == PREPOSITION and trace["measurements"]:
            zero = min(zero, trace["measurements"][0][0])
        stop = min(outcome.get("mono_ns", arrivals[-1]), arrivals[-1])
        origin = max(zero, stop - round(max_seconds * 1e9))
    if stop < origin:
        raise ValueError("Trace has no chronological clip interval")
    for output_ns in range(origin, stop + 1, step):
        mono = output_ns
        ros = None
        if historical:
            offset = mono - base["origin_mono_ns"]
            tick = ticks[bisect_right(ticks, hmb_start + offset) - 1]
            source = anchor["source_cutoff_ns"] + tick - hmb_start
            recorded_offset = source - base["origin_schedule_ns"]
            mono = base["origin_mono_ns"] + recorded_offset
            ros = base["origin_log_ns"] + base["epoch_delta_ns"] + recorded_offset
        else:
            tick = hmb_start
            source = anchor["source_cutoff_ns"]
        measured = feedback_at(frames, arrivals, mono, ros, names)
        wrists, fingers, partition, index = human[tick]
        locked = None if wrists is None or trace["locked"] is None else trace["locked"] @ wrists
        yield {"stage": stage, "elapsed_s": (output_ns - zero) / 1e9, "cutoff_mono_ns": mono,
               "source_cutoff_ns": source, "hmb_cutoff_ns": tick, "hmb_partition": partition, "hmb_tick": index,
               "source_is_frozen_anchor": not historical, "human_registered": wrists, "human_fingers": fingers,
               "world_target": locked, "measured": measured,
               "status": outcome.get("status", "NO_HANDOFF_RECEIPT"), "stage_start_mono_ns": zero}


def font(size, bold=False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (name, Path("/usr/share/fonts/truetype/dejavu") / name,
                  Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"))
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def projection(origin, centre, points, scale=365):
    yaw, elevation = np.deg2rad([36., 24.])
    x = [np.cos(yaw), np.sin(yaw), 0]
    y = [-np.sin(yaw) * np.sin(elevation), np.cos(yaw) * np.sin(elevation), np.cos(elevation)]
    coordinates = (np.asarray(points) - origin) @ np.asarray([x, y]).T
    return coordinates * [scale, -scale] + centre


def pixel(point):
    return tuple(float(value) for value in point)


def wire(draw, coordinates, color, width=2):
    if np.isfinite(coordinates).all():
        draw.line([pixel(p) for p in coordinates], fill=color, width=width, joint="curve")


def dot(draw, centre, radius, color, stroke=2):
    x, y = pixel(centre)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="white", outline=color, width=stroke)


class DrawVideo:
    def __init__(self, root, anchor, locked, center_offset_mm=None, start_only_profile=None):
        self.root = root
        self.center_offset_mm = center_offset_mm
        self.start_only_profile = start_only_profile
        h = np.asarray(anchor["registered_wrist_start_h"], dtype=float)
        self.source_center = VIRTUAL_R @ h[:, :3, 3].mean(axis=0)
        self.world_center = (locked @ h)[:, :3, 3].mean(axis=0) if locked is not None else None
        self.fonts = {"title": font(22, True), "small_title": font(17, True), "body": font(15),
                      "small": font(12), "footer": font(13)}

    def text(self, draw, xy, value, size="body", color=INK, **kwargs):
        draw.text(xy, value, font=self.fonts[size], fill=color, **kwargs)

    def panel(self, draw, x, title, caption):
        draw.rounded_rectangle((x, 89, x + 452, 581), radius=13, fill="white", outline=GRID, width=2)
        self.text(draw, (x + 17, 108), title, "small_title")
        self.text(draw, (x + 17, 131), caption, "small", MUTED)

    def wrist_panel(self, draw, x, data, origin, *, robot_fingers=None, arm=None):
        if data is None or origin is None:
            self.text(draw, (x + 226, 285), "No qualified observation", "body", MUTED, anchor="mm")
            return
        centre = np.asarray([x + 226, 324])
        wrists = data[:, :3, 3]
        project = lambda p: projection(origin, centre, p)
        if arm is not None:
            for side in range(2):
                wire(draw, project(arm[side]), SIDES[side], 3)
        if robot_fingers is not None:
            for side in range(2):
                for finger in robot_fingers[side]:
                    wire(draw, project(finger), SIDES[side], 2)
                    dot(draw, project(finger[-1]), 2, SIDES[side], 1)
        baseline = project(wrists)
        wire(draw, baseline, "#7f8fa1", 1)
        mp = project(wrists.mean(axis=0))
        dot(draw, mp, 4, "#596d84", 1)
        for side in range(2):
            p = wrists[side]
            for i, color in enumerate(AXIS):
                wire(draw, project(np.stack((p, p + .060 * data[side, :3, i]))), color)
            dot(draw, baseline[side], 5, SIDES[side])
            self.text(draw, (baseline[side, 0] + 5, baseline[side, 1] + 7), "L" if side == 0 else "R", "small", SIDES[side])
        midpoint = wrists.mean(axis=0)
        self.text(draw, (x + 17, 403), f"Wrist gap {np.linalg.norm(wrists[1] - wrists[0]) * 1000:.0f} mm", "body")
        coords = ", ".join(f"{point * 1000:+.0f}" for point in midpoint)
        self.text(draw, (x + 17, 425), f"Midpoint ({coords}) mm", "small", MUTED)
        wire(draw, np.asarray([[x + 363.5, 451], [x + 400., 451]]), "#7f8fa1")
        self.text(draw, (x + 363, 456), "100 mm", "small", MUTED)

    def local_hands(self, draw, x, hands):
        draw.line((x + 18, 474, x + 434, 474), fill=GRID, width=1)
        self.text(draw, (x + 18, 484), "MANUS hand shapes, each in its own local frame", "small", MUTED)
        for side, hand in enumerate(hands):
            label = "L" if side == 0 else "R"
            mid = (x + 130 + side * 190, 548)
            self.text(draw, (mid[0] - 64, 515), label, "small", SIDES[side])
            if hand is None:
                self.text(draw, (mid[0], mid[1]), "Unavailable", "small", MUTED, anchor="mm")
                continue
            for chain in hand:
                points = np.vstack((np.zeros(3), chain))
                points = projection(np.zeros(3), mid, points, scale=205)
                wire(draw, points, SIDES[side])
                dot(draw, points[-1], 2, SIDES[side], 1)

    def draw(self, frame, robot):
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", FRAME_SIZE, "#f3f7fb")
        draw = ImageDraw.Draw(canvas)
        root = self.root.split(":")[-1]
        self.text(draw, (18, 17), "DCR-009   Native MuJoCo guarded diagnostic", "title")
        offset = self.center_offset_mm
        if self.start_only_profile == "segmented":
            self.text(draw, (800, 23), "Segmented START / original center", "small_title", color=INK, anchor="ma")
        elif offset is not None:
            label = "Original center" if not offset else f"{'Left' if offset > 0 else 'Right'} {abs(offset)} mm"
            self.text(draw, (800, 23), label, "small_title", color=INK, anchor="ma")
        if frame["source_is_frozen_anchor"]:
            stage = "PREPOSITION  |  Frozen human anchor  |  NO HISTORICAL REPLAY"
            if frame["stage"] != PREPOSITION:
                stage = "SETUP ONLY  |  NO PREPOSITION OR HISTORICAL REPLAY"
        else:
            stage = "HISTORICAL  |  Original source cadence"
        self.text(draw, (18, 53), f"{stage}     {frame['elapsed_s']:06.3f} s", "body")
        self.text(draw, (1420, 21), f"Train root  {root}", "small_title", anchor="ra")
        self.text(draw, (1420, 55), f"Original HMB cutoff {frame['hmb_cutoff_ns'] / 1e9:.6f} s", "small", MUTED, anchor="ra")
        for x, title, caption in ((18, "Original human input", "Registered wrist markers; virtual viewing basis"),
                                  (494, "Locked registered-wrist target", "One common rigid map, applied to each marker"),
                                  (970, "MuJoCo measured", "Actual 54-axis feedback; kinematic model landmarks")):
            self.panel(draw, x, title, caption)
        raw = frame["human_registered"]
        if raw is not None:
            display = np.eye(4)
            display[:3, :3] = VIRTUAL_R
            raw = display @ raw
        self.wrist_panel(draw, 18, raw, self.source_center)
        self.local_hands(draw, 18, frame["human_fingers"])
        self.wrist_panel(draw, 494, frame["world_target"], self.world_center)
        self.text(draw, (511, 492), "Target wrist markers carry no calibrated palm frame.", "small", MUTED)
        measured = frame["measured"]
        if measured is not None:
            actual = robot.geometry(measured[2])
            poses = np.repeat(np.eye(4)[None], 2, axis=0)
            poses[:, :3, :3] = actual.rotations
            poses[:, :3, 3] = actual.wrists
            arms = np.asarray([
                np.vstack(([robot.data.joint(name).xanchor.copy() for name in robot.names[s*7:(s+1)*7]],
                           actual.wrists[s]))
                for s in range(2)
            ])
            show_origin = self.world_center
            if show_origin is None:
                show_origin = robot.geometry(robot.home).wrists.mean(axis=0)
            self.wrist_panel(draw, 970, poses, show_origin, robot_fingers=actual.landmarks, arm=arms)
            received_age = (frame["cutoff_mono_ns"] - measured[0]) / 1e6
            self.text(draw, (987, 492), f"All 54 axes share one header  |  received {received_age:.1f} ms before cutoff", "small", MUTED)
        else:
            self.wrist_panel(draw, 970, None, None)
        footer = "XYZ axes are marker/TCP axes. Human hand shapes use their signed local samples."
        self.text(draw, (20, 605), footer, "footer", MUTED)
        self.text(draw, (1420, 605), f"Native outcome: {frame['status']}  |  No formal stage credit", "footer", MUTED, anchor="ra")
        return canvas


def write_video(rows, output, painter, robot, fps, ffmpeg):
    moments = output / "render_frames.jsonl"
    video = output / "guarded_diagnostic.mp4"
    poster = output / "poster.png"
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "19", "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    observed = Counter()
    first = last = None
    poster_captured = False
    with moments.open("x", encoding="utf-8") as stream, subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for index, frame in enumerate(rows):
            image = painter.draw(frame, robot)
            measured = frame["measured"]
            if measured is not None and not poster_captured:
                image.save(poster)
                poster_captured = True
            encoder.stdin.write(image.tobytes())
            metadata = {key: frame[key] for key in ("stage", "elapsed_s", "cutoff_mono_ns", "source_cutoff_ns",
                                                    "hmb_cutoff_ns", "hmb_partition", "hmb_tick",
                                                    "source_is_frozen_anchor", "stage_start_mono_ns", "status")}
            metadata.update(index=index, original_registered_valid=frame["human_registered"] is not None,
                            locked_target_valid=frame["world_target"] is not None,
                            measured_complete_received_ns=measured[0] if measured is not None else None,
                            measured_header_ns=measured[1] if measured is not None else None)
            observed.update(output_frames=1, measured_frames=int(measured is not None),
                            source_frames=int(frame["human_registered"] is not None),
                            target_frames=int(frame["world_target"] is not None))
            stream.write(json.dumps(metadata, separators=(",", ":"), allow_nan=False) + "\n")
            first = metadata if first is None else first
            last = metadata
        encoder.stdin.close()
        if encoder.wait() or not observed["output_frames"] or not observed["measured_frames"]:
            raise RuntimeError("Encoding failed or the clip contains no complete actual simulation feedback")
    return {"video": video.name, "poster": poster.name, "frame_manifest": moments.name,
            "video_sha256": digest(video), "poster_sha256": digest(poster), "frame_manifest_sha256": digest(moments),
            "frame_counts": dict(observed), "first_frame": first, "last_frame": last,
            "encoded_duration_seconds": observed["output_frames"] / fps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("attempt", "plan", "cohort", "model", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[2] / "docs/s0-5")
    parser.add_argument("--hmb-products", type=Path)
    parser.add_argument("--max-seconds", type=float, default=8)
    parser.add_argument("--preposition-from-seconds", type=float)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 10 or not 1 <= args.fps <= 50:
        parser.error("Render no more than 10 seconds at 1–50 fps")
    if args.preposition_from_seconds is not None and not 0 <= args.preposition_from_seconds < 10:
        parser.error("Preposition start must be in the recorded 10-second bounded window")
    result, input_receipt, path, anchor_path, anchor, model_sha = checked_inputs(
        args.attempt, args.plan, args.cohort, args.model)
    cohort = json.loads(args.cohort.read_bytes())
    matches = [root for root in cohort["selected_roots"] if root["root"] == result["root"] and root["split"] == "Train"]
    if len(matches) != 1 or input_receipt["root"] != matches[0]:
        raise ValueError("Input receipt and original signed HMB root metadata differ")
    yaw_alignment = result.get("align_start_yaw_to_home") is True
    if yaw_alignment != (input_receipt.get("align_start_yaw_to_home") is True):
        raise ValueError("The result and input receipt disagree on the yaw alignment")
    trace = trace_data(path, model_sha, anchor, yaw_alignment)
    center = result.get("center_offset_mm")
    if center != input_receipt.get("center_offset_mm"):
        raise ValueError("Video center offset does not match the native input receipt")
    hmb_end = anchor["hmb_cutoff_ns"]
    historical = trace["history_end"] is not None and args.preposition_from_seconds is None
    if historical:
        hmb_end += round(args.max_seconds * 1e9)
    human = human_rows(cohort, args.docs, args.hmb_products, matches[0], anchor, hmb_end)
    robot = RobotModel(args.model)
    profile = result.get("start_only_profile")
    if profile != input_receipt.get("start_only_profile"):
        raise ValueError("Video START profile does not match the native input receipt")
    painter = DrawVideo(result["root"], anchor, trace["locked"], center, profile)
    rows = render_rows(trace, human, anchor, signed_feedback_names(robot), args.fps, args.max_seconds,
                       args.preposition_from_seconds)
    args.out.mkdir(exist_ok=False, parents=True)
    report = write_video(rows, args.out, painter, robot, args.fps, args.ffmpeg)
    if historical:
        input_mode = "SIGNED_HMB_REPLAY"
    elif args.preposition_from_seconds is not None:
        input_mode = "FROZEN_SIGNED_START_PREPOSITION"
    else:
        input_mode = "FROZEN_SIGNED_START_NO_HISTORICAL_REPLAY"
    report.update({"schema": "s05.teleop.dcr009_guarded_diagnostic_video.v1", "scope": "DIAGNOSTIC_ONLY",
                   "formal_coverage_credit": False, "root": result["root"], "split": "Train",
                   "native_result_sha256": digest(args.attempt / "result.json"), "source_trace_sha256": result["trace_sha256"],
                   "stored_render_trace_sha256": digest(path), "input_receipt_sha256": digest(args.attempt / "evidence" / "input.json"),
                   "plan_sha256": digest(args.plan), "cohort_sha256": digest(args.cohort),
                   "anchor_sha256": digest(anchor_path), "actual_model_sha256": model_sha,
                   "anatomical_mount_status": "UNKNOWN", "locked_map_from_actual_native_start": trace["locked"] is not None,
                   "original_input_mode": input_mode,
                   "center_offset_mm": center, "preposition_from_seconds": args.preposition_from_seconds,
                   "start_only_profile": profile,
                   "align_start_yaw_to_home": yaw_alignment,
                   "feedback": "Same-header four native MuJoCo groups; FK only; monotonic receipt cutoff and 40 ms freshness",
                   "timing": "Historical samples chosen at preceding signed HMB cutoff with recorded ROS epoch. Preposition uses actual observer monotonic receipt time.",
                   "fps": args.fps, "frame_size": FRAME_SIZE})
    with (args.out / "render_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(args.out / report["video"])


if __name__ == "__main__":
    main()
