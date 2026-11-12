"""Return-weighted resets within a concatenated motion library."""

import torch


class ReturnResetSampler:
    def __init__(self, library, num_envs, gamma, reward_max, *, clip_groups=None, bin_steps=50, window=20):
        self.gamma = gamma
        self.reward_max = reward_max
        self.window = window
        self.num_envs = num_envs
        self.end = library.end
        device = library.start.device
        start = library.start
        limit = library.reset_limit[start]
        num_bins = (limit - start) // bin_steps + 1
        first_bin = num_bins.cumsum(0) - num_bins
        clip = torch.arange(len(start), device=device).repeat_interleave(num_bins)
        index = torch.arange(len(clip), device=device)
        self.bin_start = start[clip] + (index - first_bin[clip]) * bin_steps
        self.bin_length = (limit[clip] - self.bin_start + 1).clamp(max=bin_steps)
        frames = torch.arange(library.frames, device=device)
        frame_clip = torch.searchsorted(start, frames, right=True) - 1
        offset = (frames - start[frame_clip]) // bin_steps
        self.frame_bin = first_bin[frame_clip] + torch.minimum(offset, num_bins[frame_clip] - 1)
        self.value = torch.ones(len(clip), device=device)
        self.count = torch.zeros_like(self.value)
        self.effort = torch.zeros_like(self.value)
        self.draws = torch.zeros_like(self.value)
        self.weights = torch.full_like(self.value, 1 / len(clip))
        self.clip = clip
        self.num_clips = len(start)
        groups = torch.arange(len(start), device=device) if clip_groups is None else torch.as_tensor(
            clip_groups, device=device
        )
        ids, groups = torch.unique(groups, return_inverse=True)
        self.group = groups[clip]
        self.num_groups = len(ids)

        shape = (num_envs, window)
        self.phase = torch.empty(shape, dtype=torch.long, device=device)
        self.reward = torch.empty(shape, device=device)
        self.done = torch.empty(shape, dtype=torch.bool, device=device)
        self.valid = torch.empty_like(self.done)
        self.mass = reward_max * (1 - gamma ** torch.arange(window + 1, device=device)) / (1 - gamma)
        self.cursor = 0
        self.updates = 0

    @torch.no_grad()
    def sample(self, count):
        if count == 0:
            return self.bin_start[:0]
        selected = torch.multinomial(self.weights, count, replacement=True)
        offset = (torch.rand(count, device=self.weights.device) * self.bin_length[selected]).long()
        self.draws.scatter_add_(0, selected, torch.ones_like(offset, dtype=self.draws.dtype))
        return self.bin_start[selected] + offset

    @torch.no_grad()
    def record(self, frame, reward, done, complete, invalid):
        col, n = self.cursor, self.num_envs
        self.phase[:, col] = frame[:n]
        failure = done[:n] & ~complete[:n]
        self.reward[:, col] = reward[:n] - self.reward_max * failure
        self.done[:, col] = done[:n]
        self.valid[:, col] = ~invalid[:n]
        self.cursor += 1
        if self.cursor == self.window:
            self._update()
            self.cursor = 0

    def _update(self):
        scores = torch.empty_like(self.reward)
        usable = torch.empty_like(self.valid)
        future = torch.zeros_like(self.reward[:, 0])
        valid = torch.ones_like(self.valid[:, 0])
        for j in range(self.window - 1, -1, -1):
            ended = self.done[:, j]
            valid = self.valid[:, j] & (ended | valid)
            future = self.reward[:, j] + self.gamma * torch.where(ended, 0., future)
            phase = self.phase[:, j]
            steps = (self.end[phase] - phase).clamp(1, self.window - j)
            scores[:, j] = (future / self.mass[steps]).clamp(-1., 1.)
            usable[:, j] = valid
        bins = self.frame_bin[self.phase].flatten()
        count = torch.zeros_like(self.count).scatter_add_(0, bins, usable.float().flatten())
        total = torch.zeros_like(self.value).scatter_add_(0, bins, scores.masked_fill(~usable, 0.).flatten())
        combined = self.count + count
        self.value.add_((total - count * self.value) / combined.clamp_min(1))
        self.count.copy_(combined.clamp(max=1024))
        focus = torch.softmax(-25.0 * self.value, dim=0)
        self.effort.add_(focus.square())
        self._distribution(focus)
        self.updates += 1

    def _distribution(self, focus):
        total = self.effort.sum()
        keep = torch.where(total > 0, self.effort / total.clamp_min(torch.finfo(self.effort.dtype).tiny), focus)
        adaptive = 0.4 * focus + 0.1 * keep
        mass = adaptive.new_zeros(self.num_groups).scatter_add_(0, self.group, adaptive)
        cap = max(0.02, 0.5 / self.num_groups)
        scale = (cap / mass.clamp_min(torch.finfo(mass.dtype).tiny)).clamp(max=1.)
        adaptive = adaptive * scale[self.group]
        self.weights.copy_(adaptive + (1. - adaptive.sum()) / len(focus))

    def state_dict(self):
        return dict(value=self.value, count=self.count, effort=self.effort, draws=self.draws, updates=self.updates)

    def load_state_dict(self, state):
        for name in ("value", "count", "effort", "draws"):
            getattr(self, name).copy_(state[name])
        self.updates = state["updates"]
        self.cursor = 0
        self._distribution(torch.softmax(-25.0 * self.value, dim=0))

    def stats(self):
        clip_draws = self.draws.new_zeros(self.num_clips).scatter_add_(0, self.clip, self.draws)
        group_mass = self.weights.new_zeros(self.num_groups).scatter_add_(0, self.group, self.weights)
        group_draws = self.draws.new_zeros(self.num_groups).scatter_add_(0, self.group, self.draws)
        return dict(
            updates=self.updates, bins=len(self.weights), sampled_bins=int((self.draws > 0).sum()),
            sampled_clips=int((clip_draws > 0).sum()), draws=int(self.draws.sum()),
            sampled_source_episodes=int((group_draws > 0).sum()), max_source_probability=float(group_mass.max()),
            min_probability=float(self.weights.min()), max_probability=float(self.weights.max()),
        )
