"""GPU-batched MuJoCo Warp physics and per-substep contact events."""

from contextlib import contextmanager

import mujoco_warp as mjw
import torch
import warp as wp

from .physics import OVERFLOW, SUBSTEPS, Physics


@wp.kernel
def contact_depth(count: wp.array(dtype=wp.int32), world: wp.array(dtype=wp.int32),
                  dist: wp.array(dtype=wp.float32), result: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if i < count[0]:
        wp.atomic_max(result, world[i], wp.max(0.0, -dist[i]))


class WarpPhysics(Physics):
    def __init__(self, model, names, num_envs, device="cuda:0", *, graph=True):
        super().__init__(model, names, num_envs, device)
        wp.init()
        self.stream = wp.Stream(device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        with self.simulation_stream():
            self.sim = mjw.put_model(model)
            self.data = mjw.make_data(model, nworld=num_envs, nconmax=96, njmax=512)
            self.depth = wp.zeros(num_envs, dtype=wp.float32)
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
            self.force = wp.to_torch(self.data.qfrc_actuator)
            self.overflow = wp.to_torch(self.data.overflow)
            self.depth_torch = wp.to_torch(self.depth)
            self.graph = None
            if graph:
                # Compile kernels before entering CUDA graph capture.
                self.substep()
                with wp.ScopedCapture(stream=self.stream) as capture:
                    self.substep()
                self.graph = capture.graph

    @contextmanager
    def simulation_stream(self):
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream), torch.cuda.stream(self.torch_stream):
            yield caller
        caller.wait_stream(self.torch_stream)

    def substep(self):
        m, d = self.sim, self.data
        mjw.step(m, d)
        # The integration has changed qpos. Measure contact at that state.
        mjw.kinematics(m, d)
        mjw.collision(m, d)
        self.depth.zero_()
        wp.launch(contact_depth, dim=d.contact.dist.shape[0],
                  inputs=[d.nacon, d.contact.worldid, d.contact.dist, self.depth])

    def state(self):
        return torch.cat((self.qpos, self.qvel), -1)

    def reset(self, ids, q, qd):
        with self.simulation_stream():
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[ids] = True
            mjw.reset_data(self.sim, self.data, reset=wp.from_torch(mask, dtype=wp.bool))
            self.qpos[ids], self.qvel[ids] = q, qd
            self.ctrl[ids[:, None], self.actuator_ids] = q

    def interval(self, target, task):
        with self.simulation_stream() as caller:
            self.ctrl[:, self.actuator_ids] = target
            state = self.state()
            torque = torch.zeros_like(self.force)
            peak = target.new_zeros(self.num_envs)
            first = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            flags = torch.zeros_like(first)
            for j in range(1, SUBSTEPS + 1):
                if self.graph is None:
                    self.substep()
                else:
                    wp.capture_launch(self.graph)
                measured = self.state()
                depth = self.depth_torch
                event = self.flags(task, measured, depth) | (OVERFLOW * (self.overflow != 0).int())
                live = first == 0
                hit = live & (event != 0)
                first = torch.where(hit, j, first)
                flags = torch.where(hit, event, flags)
                state = torch.where(live[:, None], measured, state)
                torque = torch.where(live[:, None], self.force, torque)
                peak = torch.where(live, torch.maximum(peak, depth), peak)
            for value in (state, torque, peak, first, flags):
                value.record_stream(caller)
        return state, torque, peak, first, flags
