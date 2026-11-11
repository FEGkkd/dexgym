"""Collect -> WM fit -> differentiable actor -> lambda-return critic.

Optimization follows CumulusAlpha/Dex-Supermimic 2becb738 (fixed variant).
Task/physics semantics are injected through an adapter; no simulator imports here.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .buffer import ReplayBuffer
from .common import RunningNorm
from .networks import Actor, Critic, FailureHead, WorldModel


class SuperMimicEngine:
    def __init__(self, task, cfg, log_dir):
        self.task = task
        self.env = task.env
        self.cfg = cfg
        self.device = task.device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.obs = self.env.reset()[0]["policy"]
        self.env.max_common_steps = cfg.warmup_steps + int(cfg.max_iterations / cfg.utd)
        actor_factory = getattr(task, "make_actor", None)
        self.actor = (actor_factory() if actor_factory is not None else Actor(task.policy_dim, task.action_dim))
        self.actor.to(self.device)
        # Keep the initialization of the other networks comparable across WM widths.
        with torch.random.fork_rng(devices=[], enabled=cfg.isolate_wm_init):
            if cfg.isolate_wm_init:
                torch.random.default_generator.manual_seed(cfg.seed + 1)
            self.wm = WorldModel(
                task.dynamics_dim, task.contact_dim, task.codec, task.action_dim,
                task.error_floor, cfg.wm_hidden, cfg.amp,
            ).to(self.device)
        self.fail_head = FailureHead(task.context_dim, task.state_dim).to(self.device)
        self.critic = Critic(task.context_dim, cfg.amp).to(self.device)
        self.torque_norm = RunningNorm(task.action_dim, eps=1e-3).to(self.device)
        self.optimizers = {k: torch.optim.Adam(v.parameters(), lr=cfg.lr) for k, v in self.modules().items()}
        self.optimizers["actor"] = torch.optim.AdamW(
            self.actor.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.buffer = ReplayBuffer(cfg.buffer_steps, self.env.num_envs, self.device, cfg.holdout_fraction)
        self.step_wm = torch.compile(self.wm) if cfg.compile else self.wm
        self.step_actor = torch.compile(self.actor) if cfg.compile else self.actor
        self.reward_step = torch.compile(task.reward) if cfg.compile_policy else task.reward
        self.task_features = torch.compile(task.context) if cfg.compile_policy else task.context
        self.wm_features = torch.compile(task.dynamics) if cfg.compile_policy else task.dynamics
        self.iteration = 0
        self.sim_steps = 0
        self.started_at = time.perf_counter()
        self.wall_offset = 0.0
        self.last_gradients = {}
        (self.log_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    def modules(self):
        return dict(actor=self.actor, wm=self.wm, fail_head=self.fail_head, critic=self.critic)

    @torch.no_grad()
    def collect(self, steps):
        stats = []
        replay_obs = getattr(self.task, "replay_observation", None)
        for _ in range(steps):
            x, m, aux = self.task.read()
            sim = self.task.simulator_context
            context = self.task.context(x, m, aux, sim)
            training_obs = self.obs[: self.buffer.train_end]
            self.actor.norm.update(training_obs[torch.isfinite(training_obs).all(-1)])
            action = (
                self.actor(self.obs)
                + self.cfg.action_noise * torch.randn(self.env.num_envs, self.task.action_dim, device=self.device)
            ).clamp(-1, 1)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            cap = self.env.transition
            c = cap["c"]
            u = cap["u"]
            next_x = self.task.reader.state()
            valid = ~cap["nan"]
            for value in (
                x,
                cap["x1"],
                cap["contact"],
                cap["torque"],
                u,
                c,
                action,
                self.task.encode_memory(m),
                next_x,
                self.obs,
                context,
            ):
                valid = valid & torch.isfinite(value).all(-1)
            train_valid = valid.clone()
            train_valid[self.buffer.train_end :] = False
            self.torque_norm.update(cap["torque"][train_valid])
            dyn = self.task.dynamics(x[train_valid], sim[train_valid], aux[train_valid])
            self.wm.norm.update(self.wm.inputs(dyn, u[train_valid]))
            self.wm.delta_norm.update(self.task.codec.difference(x[train_valid], cap["x1"][train_valid]))
            selected = {k: v[train_valid] for k, v in m.items()}
            _, cage, _ = self.task.reward(
                cap["x1"][train_valid],
                selected,
                action[train_valid],
                c[train_valid],
                cap["contact"][train_valid],
                cap["torque"][train_valid],
            )
            next_m = self.task.advance_memory(
                cap["x1"][train_valid],
                selected,
                action[train_valid],
                c[train_valid],
                cage,
                self.task.current_goal(selected),
                soft=False,
            )
            next_context = self.task.context(cap["x1"][train_valid], next_m, aux[train_valid], sim[train_valid])
            self.fail_head.norm.update(self.fail_head.features(cap["x1"][train_valid], next_context))
            transition = dict(
                x=x,
                x1=cap["x1"],
                push=next_x - cap["x1"],
                u=u,
                c=c,
                action=action,
                m=self.task.encode_memory(m),
                obs=self.obs if replay_obs is None else replay_obs(self.obs),
                aux=aux,
                contact=cap["contact"],
                torque=cap["torque"],
                reward=reward,
                done=terminated | truncated,
                terminated=terminated,
                valid=valid,
            )
            eligible_start = getattr(self.task, "imagination_start_mask", None)
            if eligible_start is not None:
                transition["imagination_valid"] = valid & eligible_start(m, self.cfg.horizon)
            self.buffer.add(transition)
            denom = valid.float().sum().clamp_min(1)
            stats.append(
                torch.stack(
                    (
                        (reward * valid).sum() / denom,
                        cap["drop"].float().mean(),
                        torch.where(valid, self.task.rotation_error(cap["x1"], m), 0.0).sum() / denom,
                    )
                )
            )
            self.obs = obs["policy"]
            ids = (terminated | truncated).nonzero().flatten()
            if len(ids):
                post, _ = self.env.reset(env_ids=ids)
                self.obs = torch.where((terminated | truncated)[:, None], post["policy"], self.obs)
            self.sim_steps += self.env.num_envs
        values = torch.stack(stats).mean(0)
        return dict(zip(("reward", "drop_rate", "rot_error"), values))

    def fit_world_model(self, holdout=False):
        c = self.cfg
        t = self.task
        w = self.buffer.sample(
            c.wm_batch,
            c.wm_window,
            holdout,
            failure_frac=0.0 if holdout else c.wm_failure_frac,
            failure_horizon=c.wm_failure_horizon,
        )
        memory = t.decode_memory(w["m"][:, 0])
        aux = w["aux"][:, 0]
        sim = t.simulator_context[w["env"]]
        x = w["x"][:, 0]
        losses = []
        state_losses = []
        contact_losses = []
        features = []
        contact_loss = getattr(t, "world_model_contact_loss", None)
        std = torch.maximum(self.wm.delta_norm.std(), self.wm.error_floor)
        for j in range(c.wm_window):
            x1, contact = self.step_wm(x, w["u"][:, j], self.wm_features(x, sim, aux))
            sl = (t.codec.difference(x1, w["x1"][:, j]) / std).square().mean(-1)
            per_channel = F.binary_cross_entropy_with_logits(contact, w["contact"][:, j], reduction="none")
            cl = per_channel.sum(-1) if contact_loss is None else contact_loss(per_channel, w["importance"][:, j])
            losses.append(c.final_state_weight * sl + c.contact_weight * cl)
            state_losses.append(sl)
            contact_losses.append(cl)
            torque = t.torque(x, w["u"][:, j], sim)
            _, cage, _ = self.reward_step(
                x1, memory, w["action"][:, j], w["c"][:, j], contact.sigmoid(), torque, soft=True
            )
            memory = t.advance_memory(x1, memory, w["action"][:, j], w["c"][:, j], cage, t.current_goal(memory))
            features.append(self.fail_head.features(x1.detach(), t.context(x1.detach(), memory, aux, sim)))
            x, aux = t.replay_push(x1, w["push"][:, j], aux)
        mask = w["mask"]
        denom = mask.sum().clamp_min(1)
        wm_loss = (torch.stack(losses, 1) * mask).sum() / mask.shape[0]
        if not holdout:
            self.optimize("wm", wm_loss)
        true = self.replay_failure_features(w, sim)
        predicted = torch.stack(features, 1).detach()
        label = w["terminated"].float()
        failure_loss = (
            (
                F.binary_cross_entropy_with_logits(self.fail_head(true), label, reduction="none")
                + F.binary_cross_entropy_with_logits(self.fail_head(predicted), label, reduction="none")
            )
            * mask
            * w["importance"]
        ).sum() / (2 * denom)
        if not holdout:
            self.optimize("fail_head", failure_loss)
        return dict(
            wm_loss=wm_loss.detach(),
            failure_loss=failure_loss.detach(),
            state_loss=(torch.stack(state_losses, 1) * mask).sum().detach() / denom,
            contact_digit_loss=(torch.stack(contact_losses, 1) * mask).sum().detach() / denom,
        )

    @torch.no_grad()
    def replay_failure_features(self, w, sim):
        batch, window = w["x1"].shape[:2]
        x = w["x1"].flatten(0, 1)
        memory = self.task.decode_memory(w["m"].flatten(0, 1))
        next_memory = getattr(self.task, "replay_failure_memory", None)
        if next_memory is not None:
            memory = next_memory(memory, w["action"].flatten(0, 1))
        context = self.task.context(
            x,
            memory,
            w["aux"].flatten(0, 1),
            sim[:, None].expand(-1, window, -1).reshape(batch * window, -1),
        )
        return self.fail_head.features(x, context).reshape(batch, window, -1)

    def optimize(self, name, loss):
        opt = self.optimizers[name]
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.modules()[name].parameters(), 1.0)
        if name == "actor":
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
        self.last_gradients[name] = float(norm)
        if not torch.isfinite(norm):
            if name == "actor":
                return
            raise FloatingPointError(f"Non-finite {name} gradient")
        opt.step()

    def lr_at(self, it):
        c = self.cfg
        progress = min(it / c.lr_decay_iters, 1.0)
        return c.lr * (c.lr_final_ratio + (1 - c.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    def sample_starts(self):
        b = self.buffer
        rows = (b.ptr - b.size + torch.arange(b.size, device=self.device)) % b.capacity
        mask = b.data.get("imagination_valid", b.data["valid"])
        eligible = mask[rows, : b.train_end].nonzero()
        if not len(eligible):
            raise RuntimeError("Replay contains no finite imagination starts")
        chosen = eligible[torch.randint(len(eligible), (self.cfg.pi_batch,), device=self.device)]
        row, env = rows[chosen[:, 0]], chosen[:, 1]
        out = {k: v[row, env] for k, v in b.data.items()}
        out.update(env=env, sim_context=self.task.simulator_context[env])
        return out

    def policy_update(self):
        c = self.cfg
        t = self.task
        w = self.sample_starts()
        x = w["x"]
        m = t.decode_memory(w["m"])
        history = t.history_from_obs(w["obs"])
        observation_from_history = getattr(t, "observation_from_history", None)
        contact_failure = getattr(t, "imagined_contact_failure", None)
        aux = w["aux"]
        sim = w["sim_context"]
        bias = t.encoder_bias[w["env"]]
        frozen = (self.wm, self.fail_head, self.critic)
        for module in frozen:
            module.requires_grad_(False)
        inputs = []
        rewards = []
        continues = []
        survival = x.new_ones(x.shape[0])
        objective = x.new_zeros(x.shape[0])
        try:
            for step in range(c.horizon):
                context = self.task_features(x, m, aux, sim)
                inputs.append(t.critic(context))
                observation = (
                    t.flatten_history(history) if observation_from_history is None
                    else observation_from_history(history, m)
                )
                action = self.step_actor(observation)
                target, u = t.control_target(action, m, bias)
                x1, contact = self.step_wm(x, u, self.wm_features(x, sim, aux))
                torque = t.torque(x, u, sim)
                contact_probability = contact.sigmoid()
                reward, cage, _ = self.reward_step(x1, m, action, target, contact_probability, torque, soft=True)
                done = t.termination(x1, m)
                next_m = t.advance_memory(x1, m, action, target, cage, t.sample_goal(x.shape[0]))
                failure = contact_failure(contact_probability) if contact_failure is not None else None
                if c.failure_head and self.iteration >= c.wm_only_iters + c.failure_warmup_iters:
                    next_context = self.task_features(x1, next_m, aux, sim)
                    fail = self.fail_head(self.fail_head.features(x1, next_context)).sigmoid().clamp(max=c.failure_clip)
                    failure = fail if failure is None else torch.maximum(failure, fail)
                if failure is not None:
                    done = 1 - (1 - done) * (1 - failure)
                objective = objective + c.gamma**step * survival * reward
                survival = survival * (1 - done)
                rewards.append(reward.detach())
                continues.append((1 - done).detach())
                x, aux = t.imagine_push(x1, next_m, aux)
                m = next_m
                obs = t.policy_observation(x, m, aux)
                history = torch.cat((history[:, 1:], obs[:, None]), 1)
            final = t.critic(self.task_features(x, m, aux, sim))
            inputs.append(final)
            objective = objective + c.gamma**c.horizon * survival * self.critic(final)
            actor_loss = -objective.mean()
            self.optimize("actor", actor_loss)
        finally:
            for module in frozen:
                module.requires_grad_(True)
        inputs = torch.stack(inputs).detach()
        rs = torch.stack(rewards)
        cs = torch.stack(continues)
        with torch.no_grad():
            self.critic.norm.update(inputs)
            values = self.critic(inputs)
            ret = values[-1]
            targets = []
            for step in reversed(range(c.horizon)):
                ret = rs[step] + c.gamma * cs[step] * ((1 - c.lam) * values[step + 1] + c.lam * ret)
                targets.append(ret)
            targets = torch.stack(targets[::-1]).flatten()
        features = inputs[:-1].flatten(0, 1)
        count = len(features)
        batch = count // c.critic_minibatches
        for _ in range(c.critic_epochs):
            permutation = torch.randperm(count, device=self.device)
            for i in range(c.critic_minibatches):
                ids = permutation[i * batch : (i + 1) * batch]
                critic_loss = (self.critic(features[ids]) - targets[ids]).square().mean()
                self.optimize("critic", critic_loss)
        return dict(actor_loss=actor_loss.detach(), critic_loss=critic_loss.detach(), survival=survival.detach().mean())

    def save(self, path, next_iteration=None):
        e = self.env
        payload = dict(
            format_version=1,
            task_id=self.task.task_id,
            config=self.cfg.to_dict(),
            modules={k: v.state_dict() for k, v in self.modules().items()},
            optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
            torque_norm=self.torque_norm.state_dict(),
            iteration=self.iteration if next_iteration is None else next_iteration,
            sim_steps=self.sim_steps,
            common_step=e.common_step_counter,
            curriculum={"adaptive_scale": e.disturbance.adaptive_scale, "success_scale": e.disturbance.success_scale},
            wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        )
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def load(self, path):
        p = torch.load(path, map_location=self.device, weights_only=False)
        if p["format_version"] != 1 or p["task_id"] != self.task.task_id:
            raise ValueError("Checkpoint task/schema mismatch")
        for k, v in self.modules().items():
            v.load_state_dict(p["modules"][k])
        for k, v in self.optimizers.items():
            v.load_state_dict(p["optimizers"][k])
        self.torque_norm.load_state_dict(p["torque_norm"])
        self.iteration = p["iteration"]
        self.sim_steps = p["sim_steps"]
        self.env.common_step_counter = p["common_step"]
        self.env.disturbance.adaptive_scale = p["curriculum"]["adaptive_scale"]
        self.env.disturbance.success_scale = p["curriculum"]["success_scale"]
        self.wall_offset = p["wall_seconds"]
        self.started_at = time.perf_counter()
        torch.set_rng_state(p["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in p["cuda_rng"]])
        np.random.set_state(p["numpy_rng"])
        random.setstate(p["python_rng"])
        # As upstream, replay is refilled in the current randomized environments.

    def learn(self):
        self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window))
        acc = 0.0
        with (self.log_dir / "metrics.jsonl").open("a", buffering=1) as log:
            for it in range(self.iteration, self.cfg.max_iterations):
                self.iteration = it
                lr = self.lr_at(it)
                for opt in self.optimizers.values():
                    for group in opt.param_groups:
                        group["lr"] = lr
                acc += 1 / self.cfg.utd
                steps = int(acc)
                acc -= steps
                stats = self.collect(steps)
                for _ in range(self.cfg.wm_batches):
                    stats.update(self.fit_world_model())
                if it >= self.cfg.wm_only_iters:
                    stats.update(self.policy_update())
                if it % self.cfg.log_interval == 0:
                    if self.buffer.train_end < self.env.num_envs:
                        with torch.no_grad():
                            stats.update({"holdout/" + k: v for k, v in self.fit_world_model(True).items()})
                    stats = {k: float(v) for k, v in stats.items()}
                    stats.update(
                        iteration=it,
                        sim_steps=self.sim_steps,
                        lr=lr,
                        wall_seconds=self.wall_offset + time.perf_counter() - self.started_at,
                    )
                    stats.update({"grad/" + k: v for k, v in self.last_gradients.items()})
                    line = json.dumps(stats)
                    log.write(line + "\n")
                    print(line, flush=True)
                if (it + 1) % self.cfg.save_interval == 0:
                    self.save(self.log_dir / f"model_{it + 1}.pt", it + 1)
        self.save(self.log_dir / "model_final.pt", self.cfg.max_iterations)
