import torch
from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.algorithms.supermimic.imagination import critic_td_lambda_loss, imagined_rollout, lambda_returns
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.motion.motion_manager import MotionManager

MJCF = "/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"
NPZ = "/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz"


def test_toy_bptt_improves_known_controllable_system():
    gain = torch.nn.Parameter(torch.tensor(-0.3))
    optimizer = torch.optim.Adam([gain], lr=0.05)

    def loss():
        state = torch.zeros(4)
        for _ in range(32):
            action = gain.tanh()
            state = state + 0.1 * action
        return (state - 1).square().mean()

    before = float(loss().detach())
    for _ in range(20):
        optimizer.zero_grad()
        result = loss()
        result.backward()
        optimizer.step()
    assert float(loss().detach()) < before * 0.2


def test_32_step_world_model_bptt_freezes_model_and_updates_actor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion = MotionManager(NPZ, 2, device)
    fk = DifferentiableMarvinFK(MJCF, device)
    actor = GoalActor().to(device)
    world = DeltaWorldModel(hidden=64).to(device)
    critic = PrivilegedCritic().to(device)
    with torch.no_grad():
        world.network[-1].weight.normal_(0, 0.01)
    for module in (world, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    origin = motion.data["q_arm"][0]
    scale = (motion.data["q_arm"] - origin).abs().amax(0) + 0.15
    mapping = ArmActionMap(origin, origin - 3, origin + 3, scale)
    state = torch.cat((origin, torch.zeros_like(origin), origin))[None].repeat(2, 1)
    start = {
        "state": state,
        "history": torch.zeros(2, 10, 56, device=device),
        "frame": torch.tensor([10, 700], device=device),
    }
    output = imagined_rollout(actor, world, critic, fk, mapping, motion, start, horizon=32)
    assert output["imagined_rewards"].shape == (32, 2)
    assert torch.isfinite(output["imagined_rewards"]).all()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    before = actor.decoder[-1].weight.detach().clone()
    optimizer.zero_grad()
    output["actor_loss"].backward()
    assert actor.decoder[-1].weight.grad is not None
    assert torch.isfinite(actor.decoder[-1].weight.grad).all()
    assert actor.decoder[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in world.parameters())
    assert all(parameter.grad is None for parameter in critic.parameters())
    optimizer.step()
    assert not torch.equal(before, actor.decoder[-1].weight)
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    critic_loss = critic_td_lambda_loss(critic, motion, output)
    assert torch.isfinite(critic_loss)
    critic_loss.backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())
    toy = lambda_returns(torch.ones(3, 2, device=device), torch.zeros(4, 2, device=device))
    assert toy.shape == (3, 2)
