"""Train a new S04 policy and a new world model in the hardware-aligned simulator."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import mujoco
import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn.functional as F

from .aligned_physics import AlignedPhysics
from .aligned_task import ServoEnv
from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.tasks.direct.wholebody_s0.bag import digest
from dexgym.tasks.direct.wholebody_s0_1.bank import TaskBank
from dexgym.tasks.direct.wholebody_s0_1.supermimic import Learner, joint_names, optimizer_steps
from dexgym.tasks.direct.wholebody_s0_2.interface import world
from dexgym.tasks.direct.wholebody_s04 import CONTRACT
from dexgym.tasks.direct.wholebody_s04.task import Env, interhand_clearance_cost, world_errors, wrist_pose_cost
from dexgym.tasks.direct.wholebody_s0.train_common import atomic_checkpoint
from .sweep import ScanBank, sweep, windows


def write(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


class Progress:
    def __init__(self, folder, objective):
        self.path = folder/"state.json"
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {
            "target_iteration": objective, "reviewed_iteration": 0, "trained_iteration": 0,
            "reviews": [], "setup": "全新策略与动力学，正常重力，SDK 手型固定，每手 20 点",
            "status": "初始化", "started_at": now(), "recommended_iteration": None,
        }

    def set(self, **items):
        self.data.update(items, updated_at=now())
        write(self.path, self.data)


class WristSamplingBank(TaskBank):
    def __init__(self, path, fraction, pool_path):
        super().__init__(path,"train","cuda:0")
        self.focus_fraction = fraction
        with np.load(pool_path) as data:
            self.wrist_pools = {name:torch.from_numpy(data[name]).to(self.end.device) for name in data.files}
        all_hard = torch.cat([self.wrist_pools[k] for k in ("teleop","egopro")])
        if not len(all_hard) or not (self.safe[all_hard] & (self.end[all_hard]-all_hard>=24)).all():
            raise ValueError("Wrist sampling must use screened train starts")
        for name in ("teleop","egopro"):
            if not len(self.wrist_pools[name]):
                self.wrist_pools[name] = all_hard
            key = "severe_"+name
            if not len(self.wrist_pools[key]):
                self.wrist_pools[key] = self.wrist_pools[name]

    def sample(self, count, generator):
        regular = super().sample(count,generator)
        device = regular.device
        ego = self.egopro[regular]
        draw = torch.rand((2,count),device=device,generator=generator)
        teleop, egopro, severe_t, severe_e = [
            self.sample_pool(self.wrist_pools[name],count,generator)
            for name in ("teleop","egopro","severe_teleop","severe_egopro")]
        focused = torch.where(ego,egopro,teleop)
        severe = torch.where(ego,severe_e,severe_t)
        focused = torch.where(draw[1]<.35,severe,focused)
        return torch.where(draw[0]<self.focus_fraction,focused,regular)


def sweep_stats(path, split):
    with np.load(path/f"sweep_{split}.npz") as z:
        status = z["status"]
        error = z["error_arm_shape_depth_mm"][:, :2].astype(np.float64).max(-1)
        scored = status == 1
        depths = z["max_contact_mm"]
        arm = z["wrist_and_hand"]
    good = error[scored]
    assert len(good) and np.isfinite(good).all(), split
    return dict(
        frames=int(len(status)), scored=int(scored.sum()), hard_stops=int((status==2).sum()),
        invalid=int((~scored).sum()), p95_mm=float(np.quantile(good,.95)),
        p50_mm=float(np.median(good)), all_ten_hit_15=float((scored&(error<=15)).mean()),
        contact_frames=int((depths>0).sum()), wrist_p95_deg=float(np.quantile(arm[scored,0],.95)),
        wrist_position_p95_mm=float(np.quantile(arm[scored,1],.95)),
    )


@torch.inference_mode()
def initial_reset(phy, starts):
    ids = torch.arange(len(starts), device=phy.device)
    reset_ids = torch.arange(phy.num_envs, device=phy.device)
    padded = list(starts) + [starts[0]] * (phy.num_envs-len(starts))
    phy.bank.same_frame = True
    phy.reset(reset_ids, frames=torch.as_tensor(padded, device=phy.device), exact=True)
    phy.bank.same_frame = False
    return ids


@torch.inference_mode()
def suites(bank, backend, model, weights, entries):
    env = Env(backend, bank, model, 49, evaluation=True)
    phy = env.direct
    phy.segment_steps = 500
    actor = env.task.make_actor().to(backend.device).eval()
    actor.load_state_dict(weights)
    frames = [10148, 38378, *[e["frame"] for e in entries]]
    ids = initial_reset(phy, frames)
    phy.stationary_steps = torch.full((backend.num_envs,), 125, device=backend.device, dtype=torch.long)
    errors, speed, contact = [], [], []
    live = torch.ones(len(ids), device=backend.device, dtype=torch.bool)
    for _ in range(100):
        phy.step(actor(phy.observation()))
        tr = phy.transition
        row = tr["rollout_stats"]
        live &= ~tr["drop"][ids]
        errors.append(torch.where(live, row["tip_world_worst_mm"][ids], 150.))
        speed.append(row["tip_speed_mm_s"][ids])
        contact.append(tr["depth"][ids] > 0)
    e = torch.stack(errors).cpu().numpy()[-25:]
    v = torch.stack(speed).cpu().numpy()[-25:]
    c = torch.stack(contact).cpu().numpy()
    valid = live.cpu().numpy()
    hard = np.r_[[True, True], [entry.get("hard_tail", False) for entry in entries]]
    hold = dict(cases=len(frames), completed=int(valid.sum()), all_p95_mm=float(np.quantile(e,.95)),
                hard_p95_mm=float(np.quantile(e[:,hard],.95)), hit_15=float((e<=15).mean()),
                tip_speed_rms_mm_s=float(np.sqrt(np.mean(v[:,valid]**2))) if valid.any() else None,
                contact_frames=int(c.sum()), case_10148_mm=float(np.median(e[:,0])),
                case_38378_mm=float(np.median(e[:,1])))
    options = windows(bank.clips)
    rows = [10148, 38378, 27552, *[entry["frame"] for entry in entries if entry.get("hard_tail")]]
    clips = [next(w for w in options if w[1] <= frame < w[2]) for frame in rows]
    starts = [max(w[0], frame-48) for frame,w in zip(rows,clips,strict=True)]
    durations = [min(w[2], frame+14) - start for frame,w,start in zip(rows,clips,starts,strict=True)]
    ids = initial_reset(phy, starts)
    phy.stationary_steps.zero_()
    trajectories, errors, wrists, rate, contacts = [], [], [], [], []
    live = torch.ones_like(ids, dtype=torch.bool)
    first_failure = torch.full_like(ids, max(durations))
    for index in range(max(durations)):
        phy.step(actor(phy.observation()))
        tr = phy.transition
        drop = tr["drop"][ids]
        first_failure = torch.where(live & drop, index, first_failure)
        live &= ~drop
        r = tr["rollout_stats"]
        trajectories.append(world(phy.actual)[ids]*1000)
        errors.append(torch.where(live, r["tip_world_worst_mm"][ids], 150.))
        wrists.append(r["wrist_rotation_worst_deg"][ids])
        rate.append(r["arm_residual_rate_deg_s"][ids])
        contacts.append(tr["depth"][ids]>0)
    tapes = torch.stack(trajectories).cpu().numpy()
    e, w, r, c = [torch.stack(tape).cpu().numpy() for tape in (errors, wrists, rate, contacts)]
    stop = first_failure.cpu().numpy()
    pooled, details = [], []
    for j, (frame,start,planned) in enumerate(zip(rows,starts,durations,strict=True)):
        visible = min(planned, stop[j])
        jitter = None
        if visible >= 11:
            p = tapes[:visible,j]
            hp = np.diff(p - savgol_filter(p,11,2,axis=0), axis=0)[2:-2] / .02
            jitter = float(np.sqrt(np.mean(np.sum(hp**2,axis=-1))))
        focus = e[max(0, frame-start-8):min(planned,frame-start+9),j]
        pooled.extend(focus)
        details.append(dict(frame=frame,start=start,planned=planned,valid=int(visible),
                            p95_mm=float(np.quantile(e[:planned,j],.95)), at_frame_mm=float(e[frame-start,j]),
                            within15=int((e[:planned,j]<=15).sum()), highpass_tip_rms_mm_s=jitter,
                            residual_rate_deg_s=float(np.sqrt(np.mean(r[1:visible,j]**2))) if visible>1 else None,
                            wrist_p95_deg=float(np.quantile(w[:visible,j],.95)) if visible else None,
                            contact_frames=int(c[:min(visible+1,planned),j].sum())))
    jitters = [d["highpass_tip_rms_mm_s"] for d in details if d["highpass_tip_rms_mm_s"] is not None]
    hard = dict(windows=len(rows), p95_mm=float(np.quantile(pooled,.95)),
                highpass_tip_rms_mm_s=float(np.median(jitters)) if jitters else None,
                contact_frames=sum(d["contact_frames"] for d in details),
                stopped=sum(d["valid"]<d["planned"] for d in details), details=details)
    return hold, hard


class ReviewTrainer(Learner):
    def __init__(self, task, cfg, folder, progress, profile, model, evaluation_envs, bank_path, hold_cases):
        super().__init__(task, cfg, folder)
        self.progress = progress
        self.profile = profile
        self.model = model
        self.evaluation_envs = evaluation_envs
        self.bank_path = bank_path
        self.hold_cases = hold_cases
        self.bank_sha256 = digest(bank_path)
        self.last_progress = 0.
        self.previous_review = None
        self.hold_teacher = None
        self.hold_geometry = None

    def policy_update(self):
        sample = super().policy_update()
        if time.monotonic() - self.last_progress > 9:
            self.last_progress = time.monotonic()
            self.progress.set(status="训练", trained_iteration=self.iteration+1,
                              transitions=self.sim_steps)
        return sample

    def save_actor(self, path, next_iteration=None):
        atomic_checkpoint(dict(**CONTRACT, iteration=next_iteration, features=self.task.policy_dim,
                               bank_sha256=self.bank_sha256, actor=self.actor.state_dict()), path)
        if next_iteration and path.name != "model_final_actor.pt":
            self.assess(next_iteration, path)

    def assess(self, iteration, actor_path):
        folder = self.log_dir/f"eval_{iteration:05d}"
        folder.mkdir(exist_ok=True)
        self.progress.set(status="验证与独立测试", trained_iteration=iteration,
                          transitions=self.sim_steps, evaluating=iteration)
        weights = {k:v.detach().cpu() for k,v in self.actor.state_dict().items()}
        current = {}
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.cuda.empty_cache()
            sim = AlignedPhysics(self.model,self.task.bank.metadata["joint_names"],self.evaluation_envs,self.profile)
            for split in ("validation","test"):
                self.progress.set(status=f"回放{('验证集' if split=='validation' else '独立测试集')}")
                bank = ScanBank(self.bank_path,split,sim.device)
                sweep(bank,self.model,sim,weights,folder,split,collect_anchor=False)
                current[split] = sweep_stats(folder,split)
                if split == "validation":
                    current["hold"], current["hard"] = suites(bank,sim,self.model,weights,self.hold_cases)
                del bank
            del sim
            torch.cuda.empty_cache()
        current.update(iteration=iteration, actor_file=actor_path.name, actual_command_transition=self.sim_steps)
        current["adjustment"] = self.adjust(current)
        write(self.log_dir/f"review_{iteration:05d}.json",current)
        rows = self.progress.data["reviews"]
        fields = ("iteration","validation","test","hold","hard","adjustment","actor_file")
        brief = {f:current[f] for f in fields}
        rows = [r for r in rows if r["iteration"]!=iteration] + [brief]
        def score(row):
            v,h,s = (row[k] for k in ("validation","hard","hold"))
            return (v["p95_mm"]+.24*h["p95_mm"] + 20*(1-v["all_ten_hit_15"])
                    + .09*(h["highpass_tip_rms_mm_s"] or 0)
                    + .45*(s["tip_speed_rms_mm_s"] or 0) + .04*v["hard_stops"])
        best = min(rows,key=score)
        self.progress.set(reviews=rows, reviewed_iteration=iteration, evaluating=None,
                          recommended_iteration=best["iteration"], status="训练")
        self.previous_review = current
        print("MILESTONE",iteration,json.dumps(
            {key:value for key,value in current.items() if key in ("validation","test","adjustment")},
            ensure_ascii=False), flush=True)

    def adjust(self, current):
        env = self.task.env.direct
        v,h,s = (current[k] for k in ("validation","hard","hold"))
        last = self.previous_review
        previous = dict(smoothness=env.smoothness_weight, wrist=env.wrist_pose_weight,
                        tip_tail=env.tip_tail_weight, contact=env.interhand_clearance_weight,
                        actor_lr=self.cfg.actor_lr)
        next_setting = previous.copy()
        notes = []
        if last is None:
            notes.append("建立新动力学下第一份闭环基线")
        else:
            va,ha,sa = (last[k] for k in ("validation","hard","hold"))
            losing_precision = v["p95_mm"] > va["p95_mm"] + .55 or v["all_ten_hit_15"] < va["all_ten_hit_15"] - .009
            jitter, old_jitter = h["highpass_tip_rms_mm_s"],ha["highpass_tip_rms_mm_s"]
            quiet, old_quiet = s["tip_speed_rms_mm_s"],sa["tip_speed_rms_mm_s"]
            if losing_precision:
                next_setting["actor_lr"] = max(1.2e-5, previous["actor_lr"]*.86)
                notes.append("验证精度回退，收小策略更新步长")
            elif ((jitter is not None and old_jitter is not None and jitter>1.12*old_jitter+.3)
                  or (quiet is not None and old_quiet is not None and quiet>1.18*old_quiet+.1)):
                next_setting["smoothness"] = min(.095, previous["smoothness"]+.007)
                notes.append("抖动回升，在验证精度未回退时小幅增加平滑约束")
            if h["p95_mm"]>15 and h["p95_mm"]>ha["p95_mm"]-.6:
                next_setting["wrist"] = min(.07, previous["wrist"]+.004)
                next_setting["tip_tail"] = min(.075, previous["tip_tail"]+.005)
                notes.append("难段尾误差改善不足，增加通用腕姿和指尖尾误差权重")
            if v["contact_frames"] > va["contact_frames"]*1.1+4:
                next_setting["contact"] = min(.078, previous["contact"]+.006)
                notes.append("验证接触增多，提高双手间距惩罚")
        env.smoothness_weight = next_setting["smoothness"]
        env.wrist_pose_weight = next_setting["wrist"]
        env.tip_tail_weight = next_setting["tip_tail"]
        env.interhand_clearance_weight = next_setting["contact"]
        self.cfg.actor_lr = next_setting["actor_lr"]
        if self.hold_teacher is not None:
            if last and losing_precision and s["hard_p95_mm"] >= last["hold"]["hard_p95_mm"]-.4:
                self.hold_teacher.weight = max(.035,self.hold_teacher.weight*.8)
                notes.append("整体精度回退且难姿态无收益，收小定点示教权重")
            notes.append(f"当前定点示教权重 {self.hold_teacher.weight:.3f}")
            next_setting["hold_teacher_weight"] = self.hold_teacher.weight
        if self.hold_geometry is not None:
            if last and losing_precision:
                self.hold_geometry.weight = max(.012,self.hold_geometry.weight*.85)
            next_setting["hold_geometry_weight"] = self.hold_geometry.weight
            notes.append(f"训练定点的几何约束权重 {self.hold_geometry.weight:.3f}")
        if isinstance(self.task.bank,WristSamplingBank):
            current_fraction = self.task.bank.focus_fraction
            if last and losing_precision:
                current_fraction = max(.09,current_fraction-.025)
                notes.append("整体精度回退，小幅减小难轨迹采样比例")
            elif last and h["p95_mm"]>15 and h["p95_mm"]>=last["hard"]["p95_mm"]-.5:
                current_fraction = min(.27,current_fraction+.025)
                notes.append("增加通用难轨迹的覆盖比例")
            self.task.bank.focus_fraction = current_fraction
            next_setting["wrist_sampling_fraction"] = current_fraction
        return dict(before=previous, next=next_setting, reasons=notes or ["指标稳定，保持本轮任务权重"])


class HoldDemonstrations:
    """Use only train poses verified to settle within 15 mm without contacts."""

    def __init__(self, path, device, weight):
        with np.load(path) as tape:
            self.obs = torch.from_numpy(tape["observations"]).to(device)
            self.actions = torch.from_numpy(tape["actions"]).to(device)
            self.prob = torch.from_numpy(tape["weight"]).float().to(device)
        self.generator = torch.Generator(device=device).manual_seed(1107)
        self.weight = weight
        self.metrics = {}

    def __call__(self, actor):
        index = torch.multinomial(self.prob,160,replacement=True,generator=self.generator)
        action = actor(self.obs[index].float())
        reference = self.actions[index]
        loss = F.smooth_l1_loss(action,reference,beta=.18)
        self.metrics["hold_teacher_loss"] = loss.detach()
        return self.weight*loss


class HoldGeometryLoss:
    """Use analytic endpoint error on on-policy train holds in the calibrated system."""

    def __init__(self, task, weight):
        self.task = task
        self.weight = weight
        self.metrics = {}

    def __call__(self, actor, obs, state, memory):
        physical = self.task.env.direct
        idx = (memory["stationary_steps"]>4).nonzero().flatten()
        self.metrics["hold_geometry_rows"] = float(len(idx))
        if not len(idx):
            self.metrics["hold_geometry_loss"] = 0.
            return obs.new_zeros(())
        frame = memory["frame"][idx]
        ref = self.task.bank.rsi_q[frame]
        goal = physical._goal(frame,memory["bridge"][idx])
        with torch.no_grad():
            initial_pose = physical.task.observe.fk(ref).pose
            initial_error = (world_errors(initial_pose,goal).amax(-1)+1e-10).sqrt()
            weights = (initial_error/.012).clamp(.35,3.)
        command = physical.decoder(actor(obs[idx]),ref)
        pose = physical.task.observe.fk(command).pose
        relative_error = (world_errors(pose,goal)+1e-10).sqrt()/.015
        displacement = F.smooth_l1_loss(relative_error,torch.zeros_like(relative_error),
                                         reduction="none",beta=.75)
        loss = (.72*displacement.amax(-1)+.28*displacement.mean(-1)
                +.09*wrist_pose_cost(pose,goal)+.24*interhand_clearance_cost(pose,goal))
        value = (weights*loss).mean()
        self.metrics["hold_geometry_loss"] = value.detach()
        return self.weight*value


class EasyStateRetention:
    """Preserve successful tracking with an earlier actor from this same run."""

    def __init__(self, task, actor_path):
        self.task = task
        self.reference = task.make_actor().to(task.device).eval().requires_grad_(False)
        self.reference.load_state_dict(torch.load(actor_path,map_location="cpu",weights_only=True)["actor"])
        self.metrics = {}

    def __call__(self, actor, obs, state, memory):
        with torch.no_grad():
            goal, _ = self.task.goals(memory)
            pose = self.task.direct.observe.fk(state[:, :54]).pose
            error = (world_errors(pose,goal).amax(-1)+1e-10).sqrt()
            confidence = ((.013-error)/.008).clamp(0.,1.)
            target = self.reference(obs)
        delta = (actor(obs)-target).square().mean(-1)
        self.metrics["easy_retention_fraction"] = (confidence>0).float().mean()
        self.metrics["easy_retention_action_rms_deg"] = 8*(delta.detach()*confidence).mean().sqrt()
        return .8*(delta*confidence).mean()


def verify_transition(env, backend):
    x, memory, aux = env.task.read()
    encoded = env.task.encode_memory(memory)
    older = memory["servo"].clone()
    torch.testing.assert_close(env.task.decode_memory(encoded)["servo"],older,atol=0,rtol=0)
    torch.testing.assert_close(env.direct.observation(current_only=True),env.task.policy_observation(x,memory,aux))
    reference = env.task.bank.rsi_q[memory["frame"]]
    pulse = torch.full((env.num_envs,14),.12,device=env.device)
    command,wanted = env.task.control_target(pulse,memory)
    _, _, _, timeout, _ = env.step(pulse)
    torch.testing.assert_close(env.transition["c"],command,atol=0,rtol=0)
    torch.testing.assert_close(env.transition["u"],wanted,atol=0,rtol=0)
    hand = env.direct.decoder.hand
    assert torch.equal(command[:,hand],reference[:,hand]) and torch.equal(wanted[:,hand],reference[:,hand])
    keep = ~env.transition["drop"] & ~timeout
    if keep.any():
        expected = torch.lerp(command[:,backend.arm],older,env.task.decay)
        torch.testing.assert_close(backend.filtered_target[keep],expected[keep],atol=3e-6,rtol=0)
    assert (command[:,backend.arm] - reference[:,backend.arm]).abs().max() <= math.radians(8)+1e-6
    return dict(filter_memory_verified=True, exact_sdk_command=True, action_dim=14, policy_input=env.task.policy_dim)


def main():
    parser = argparse.ArgumentParser(description="Train the 20-point S0-4 policy in the aligned MuJoCo backend")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--hold-cases", type=Path, required=True)
    parser.add_argument("--iterations",type=int,default=20000)
    parser.add_argument("--envs",type=int,default=64)
    parser.add_argument("--evaluation-envs",type=int,default=112)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--hold-teacher",action="store_true")
    parser.add_argument("--hold-geometry",action="store_true")
    parser.add_argument("--mixed-wrist",action="store_true")
    args = parser.parse_args()
    if args.mixed_wrist and (args.hold_geometry or args.hold_teacher):
        parser.error("Wrist sampling uses its own regular-motion coverage objective")
    smoke = "smoke_wrist" if args.mixed_wrist else "smoke_geometry" if args.hold_geometry else "smoke"
    folder = args.output/(smoke if args.smoke else "train")
    folder.mkdir(parents=True, exist_ok=True)
    progress = Progress(folder,args.iterations)
    try:
        if CONTRACT["goal_points_per_hand"] != 20:
            raise ValueError("Set S04_GOAL_POINTS=20 for aligned training")
        if not args.resume and (folder/"provenance.json").is_file():
            raise RuntimeError("Run already initialized")
        seed = 49
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("high")
        bank = (WristSamplingBank(args.bank,.18,args.output/"wrist_sample_pool.npz") if args.mixed_wrist
                else TaskBank(args.bank,"train","cuda:0"))
        path = args.model
        model = mujoco.MjModel.from_xml_path(str(path))
        assert joint_names(model)==bank.metadata["joint_names"] and digest(path)==bank.metadata["teleop_model_sha256"]
        profile = json.loads(args.profile.read_text())
        backend = AlignedPhysics(model,bank.metadata["joint_names"],args.envs,profile)
        env = ServoEnv(backend,bank,model,profile,seed,stationary_reset_fraction=.12)
        env.direct.decoder.check_bank(bank.rsi_q)
        env.direct.smoothness_weight = .043
        env.direct.quiet_smoothness_scale = 1.65
        env.direct.wrist_pose_weight = .026
        env.direct.tip_tail_weight = .032
        env.direct.interhand_clearance_weight = .042
        env.task.imagined_actor_grad_cap = 1.15
        cfg = SuperMimicConfig(
            seed=seed, max_iterations=args.iterations, buffer_steps=1800, warmup_steps=66,
            utd=.5, wm_only_iters=4 if args.smoke else 350, wm_window=8, wm_batches=3, wm_batch=768,
            wm_hidden=(1280,1280,1280), pi_batch=384, horizon=8, isolate_wm_init=True,
            control_error_features=True, contact_partial_intervals=True, contact_weight=.2,
            contact_depth_weight=.03, failure_warmup_iters=4 if args.smoke else 260,
            failure_clip=.25, lr=1.1e-4, actor_lr=4.0e-5, actor_bootstrap_weight=.65, max_step_reward=1.,
            action_noise=0, critic_epochs=3, optimizer="fused-adam", amp=False, compile=False,
            compile_policy=False, profile_training=True,
            save_interval=args.iterations+1 if args.smoke else 1000, log_interval=1 if args.smoke else 10,
        )
        engine = ReviewTrainer(env.task,cfg,folder,progress,profile,model,args.evaluation_envs,
                               args.bank,json.loads(args.hold_cases.read_text()))
        proof = verify_transition(env,backend)
        engine.obs = env.direct.observation()
        if args.resume:
            previous = json.loads((folder/"provenance.json").read_text())
            assert previous["servo_profile_sha256"]==digest(args.profile) and previous["bank_sha256"]==digest(args.bank)
            engine.load(args.resume)
            row = progress.data["reviews"]
            if row:
                new = row[-1]["adjustment"]["next"]
                engine.previous_review = row[-1]
                env.direct.smoothness_weight,env.direct.wrist_pose_weight = new["smoothness"],new["wrist"]
                env.direct.tip_tail_weight,env.direct.interhand_clearance_weight = new["tip_tail"],new["contact"]
                cfg.actor_lr = new["actor_lr"]
                if args.mixed_wrist:
                    bank.focus_fraction = new.get("wrist_sampling_fraction", bank.focus_fraction)
        if args.hold_teacher:
            weight = .15
            if engine.previous_review:
                weight = engine.previous_review["adjustment"]["next"].get("hold_teacher_weight",weight)
            teacher = HoldDemonstrations(args.output/"hold_teacher.npz",env.device,weight)
            assert teacher.obs.shape[-1]==env.task.policy_dim and teacher.actions.shape[-1]==14
            engine.hold_teacher = teacher
            env.task.actor_reference_anchor = teacher
        if args.hold_geometry:
            weight = .025
            if engine.previous_review:
                weight = engine.previous_review["adjustment"]["next"].get("hold_geometry_weight", weight)
            engine.hold_geometry = HoldGeometryLoss(env.task,weight)
            env.task.actor_online_anchor = engine.hold_geometry
        if args.mixed_wrist:
            env.task.actor_online_anchor = EasyStateRetention(env.task,args.output/"train/model_3000_actor.pt")
        if not args.resume:
            assert engine.iteration==0 and engine.sim_steps==0 and not any(optimizer_steps(engine).values())
            write(folder/"provenance.json",dict(**CONTRACT, initialization="random_policy_random_world_model",
                  parent_checkpoint=None, model_sha256=digest(path), servo_profile=profile,
                  servo_profile_sha256=digest(args.profile), bank=str(args.bank), bank_sha256=digest(args.bank),
                  config=cfg.to_dict(), backend="MuJoCo Warp aligned to native 2ms", envs=args.envs,
                  proof=proof, seed=seed, created_at=now()))
        progress.set(status="采集校准仿真 rollout", initialized=True, error=None,
                      hold_geometry=args.hold_geometry, uses_hold_demonstrations=args.hold_teacher,
                      uses_wrist_sampling=args.mixed_wrist,
                      trained_iteration=engine.iteration, backend="MuJoCo Warp / 2 ms", target_iteration=args.iterations)
        start = optimizer_steps(engine)
        engine.learn()
        updates = {k:v-start[k] for k,v in optimizer_steps(engine).items()}
        assert updates["wm"]>0 and updates["actor"]>0, updates
        write(folder/"training_proof.json",dict(iterations=args.iterations,optimizer_steps=updates,
                  transitions=engine.sim_steps,finished_at=now(),contract=proof))
        progress.set(status="训练与验证完成", trained_iteration=args.iterations, transitions=engine.sim_steps)
    except BaseException as exc:
        progress.set(status="运行中断", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__=="__main__":
    main()
