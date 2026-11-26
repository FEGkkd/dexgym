"""One-frame Replay with explicit causal and source boundaries for S0-5."""
import torch

IDENTITIES = ("source_id", "partition_id", "episode_id", "generation_id")
REQUIRED = frozenset((*IDENTITIES, "obs", "done", "valid", "source_train"))


class CausalReplayBuffer:
    """Store one frame per environment tick and reconstruct normalized input order on demand.

    The task provides raw observations with its exact all-zero padding convention. Source
    identities stay outside Actor observations. Train and holdout lanes must be backed by
    prevalidated disjoint source sets even when the same role is replayed across many lanes.
    """

    def __init__(self, capacity, num_envs, device, history, train_end, *,
                 train_root_ids, holdout_root_ids, storage_device=None):
        if (capacity < history or history < 1 or not 0 < train_end <= num_envs):
            raise ValueError("invalid replay shape or lane assignment")
        if not train_root_ids or set(train_root_ids) & set(holdout_root_ids):
            raise ValueError("seeded train/holdout source sets must be disjoint")
        if (train_end == num_envs) != (not holdout_root_ids):
            raise ValueError("holdout lanes and dataset roles differ")
        self.capacity, self.num_envs, self.history, self.train_end = capacity, num_envs, history, train_end
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device or device)
        self.train_sources = torch.tensor(sorted(train_root_ids), dtype=torch.int64, device=self.storage_device)
        self.holdout_sources = torch.tensor(sorted(holdout_root_ids), dtype=torch.int64, device=self.storage_device)
        self.ptr = self.size = 0
        self.wrapped = False
        self.data = {}
        self.pools = {}

    def add(self, transition):
        if not REQUIRED <= transition.keys():
            raise ValueError(f"missing causal replay fields: {sorted(REQUIRED - transition.keys())}")
        if transition["obs"].ndim != 2:
            raise ValueError("store one raw frame per lane")
        for name in IDENTITIES:
            if transition[name].dtype != torch.int64 or transition[name].shape != (self.num_envs,):
                raise ValueError(f"{name} requires one int64 per lane")
        for name in ("done", "valid", "source_train"):
            if transition[name].dtype != torch.bool or transition[name].shape != (self.num_envs,):
                raise ValueError(f"{name} requires one bool per lane")
        source = transition["source_id"].to(self.storage_device)
        role = transition["source_train"].to(self.storage_device)
        if (not bool(role[: self.train_end].all()) or bool(role[self.train_end :].any()) or
                not bool(torch.isin(source[: self.train_end], self.train_sources).all()) or
                not bool(torch.isin(source[self.train_end :], self.holdout_sources).all())):
            raise ValueError("replay source crosses the frozen split")
        if not self.data:
            if any(v.shape[0] != self.num_envs for v in transition.values()):
                raise ValueError("all replay fields must have the environment leading axis")
            self.data = {
                key: torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=self.storage_device)
                for key, value in transition.items()
            }
        elif set(transition) != set(self.data) or any(
            value.shape != self.data[key].shape[1:] or value.dtype != self.data[key].dtype
            for key, value in transition.items()
        ):
            raise ValueError("replay schema changed")
        self.pools.clear()
        self.wrapped |= self.size == self.capacity
        for key, value in transition.items():
            safe = value.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) if value.is_floating_point() else value
            self.data[key][self.ptr].copy_(safe)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.data.values())

    def chronological(self):
        return (self.ptr - self.size + torch.arange(self.size, device=self.storage_device)) % self.capacity

    def _starts(self, window, holdout, imagination=False):
        key = (window, holdout, imagination)
        if key in self.pools:
            return self.pools[key]
        if not 1 <= window <= self.size:
            raise ValueError("window exceeds available replay")
        low, high = (self.train_end, self.num_envs) if holdout else (0, self.train_end)
        if low == high:
            raise ValueError("no lanes matching the requested source role")
        rows = self.chronological()
        view = slice(low, high)
        eligible = self.data["valid"][rows, view]
        if "full_step" in self.data:
            # Failure transitions stay readable and can terminate a window even if the
            # step was cut short. They cannot be used as later prediction inputs.
            complete = self.data["full_step"][rows, view]
        else:
            complete = eligible
        size = self.size - window + 1
        valid = eligible[:size].clone()
        if imagination:
            valid &= self.data.get("imagination_valid", self.data["valid"])[rows[:size], view]
            valid &= ~self.data["done"][rows[:size], view]
        if self.wrapped:
            # Earlier frames were evicted, so a zero prefix would differ from
            # the online builder for a still-active segment.
            valid[: self.history - 1] = False
        for lag in range(1, window):
            later = rows[lag : lag + size]
            earlier = rows[lag - 1 : lag - 1 + size]
            valid &= eligible[lag : lag + size] & complete[lag - 1 : lag - 1 + size]
            valid &= ~self.data["done"][earlier, view]
            for field in IDENTITIES:
                valid &= self.data[field][later, view] == self.data[field][earlier, view]
        starts = valid.nonzero()
        if not len(starts):
            raise ValueError("no finite same-generation causal windows")
        starts[:, 1] += low
        self.pools[key] = starts
        return starts

    def _history(self, times, lanes):
        if self.wrapped and bool((times < self.history - 1).any()):
            raise ValueError("retained replay lacks real causal predecessors")
        data = self.data
        rows = self.chronological()
        current = rows[times]
        # Start with the current frame and retain each predecessor only while
        # the whole observed prefix through it is finite and in this segment.
        playout = data["obs"][current, lanes]
        frames = [playout]
        alive = data["valid"][current, lanes]
        successor = current
        for lag in range(1, self.history):
            index = times - lag
            previous = rows[index.clamp_min(0)]
            alive = alive & (index >= 0) & data["valid"][previous, lanes] & ~data["done"][previous, lanes]
            for field in IDENTITIES:
                alive = alive & (data[field][previous, lanes] == data[field][successor, lanes])
            value = data["obs"][previous, lanes]
            frames.append(torch.where(alive[..., None], value, 0))
            successor = previous
        return torch.cat(frames[::-1], dim=-1)

    def _sample_indices(self, pool, batch, generator):
        if batch < 1:
            raise ValueError("positive batch size required")
        # Device-specific tensors use the caller's generator on the compute device.
        return torch.randint(len(pool), (batch,), device=self.device, generator=generator).to(self.storage_device)

    def sample(self, batch, window=1, holdout=False, generator=None, failure_frac=0.0, failure_horizon=16):
        if not 0 <= failure_frac < 1 or failure_horizon < 1:
            raise ValueError("invalid failure oversampling settings")
        pool = self._starts(window, holdout)
        selected = self._sample_indices(pool, batch, generator)
        weight = torch.ones(batch, device=self.storage_device)
        if failure_frac:
            failure = self._failure_membership(pool, failure_horizon)
            positions = failure.nonzero(as_tuple=True)[0]
            draws = int(batch * failure_frac) if len(positions) else 0
            if draws:
                selected[:draws] = positions[self._sample_indices(positions, draws, generator)]
                selected = selected[torch.randperm(batch, device=self.device, generator=generator).to(self.storage_device)]
                probability = (1 - draws / batch) / len(pool) + (draws / batch) * failure[selected] / len(positions)
                weight = 1 / (len(pool) * probability)
        time, lane = pool[selected].unbind(-1)
        slots = time[:, None] + torch.arange(window, device=self.storage_device)
        lanes = lane[:, None].expand_as(slots)
        rows = self.chronological()[slots]
        result = {key: value[rows, lanes].to(self.device) for key, value in self.data.items()
                  if key != "obs" and key not in IDENTITIES and key != "source_train"}
        result["obs"] = self._history(slots, lanes).to(self.device)
        result["mask"] = torch.ones((batch, window), device=self.device)
        result["importance"] = weight[:, None].expand(-1, window).to(self.device)
        result["env"] = lane.to(self.device)
        return result

    def _failure_membership(self, starts, horizon):
        rows = self.chronological()
        times, lanes = starts.unbind(-1)
        active = torch.ones(len(starts), dtype=torch.bool, device=self.storage_device)
        member = torch.zeros_like(active)
        failed = self.data.get("safety_failure", self.data.get("terminated", self.data["done"]))
        initial_row = rows[times]
        for offset in range(min(horizon, self.size)):
            index = times + offset
            row = rows[index.clamp_max(self.size - 1)]
            active &= index < self.size
            active &= self.data["valid"][row, lanes]
            for field in IDENTITIES:
                active &= self.data[field][row, lanes] == self.data[field][initial_row, lanes]
            member |= active & failed[row, lanes]
            active &= ~self.data["done"][row, lanes]
        return member

    def sample_starts(self, batch, generator=None):
        pool = self._starts(1, False, imagination=True)
        time, lane = pool[self._sample_indices(pool, batch, generator)].unbind(-1)
        rows = self.chronological()[time]
        result = {key: value[rows, lane].to(self.device) for key, value in self.data.items()
                  if key != "obs" and key not in IDENTITIES and key != "source_train"}
        result["obs"] = self._history(time, lane).to(self.device)
        result["env"] = lane.to(self.device)
        return result
