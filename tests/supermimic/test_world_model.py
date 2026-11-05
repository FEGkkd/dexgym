import torch
from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.algorithms.supermimic.networks import DeltaWorldModel


def test_eight_step_autoregressive_model_learns_action_conditioned_dynamics():
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = ArmActionMap(
        torch.zeros(14, device=device),
        -torch.ones(14, device=device),
        torch.ones(14, device=device),
        torch.ones(14, device=device),
        0.08,
    )
    initial_q = torch.randn(48, 14, device=device) * 0.05
    initial_v = torch.randn_like(initial_q) * 0.1
    state = torch.cat((initial_q, initial_v, initial_q), dim=-1)
    states, actions, next_states = [], [], []
    with torch.no_grad():
        for _ in range(8):
            action = torch.randn(48, 14, device=device) * 0.3
            command = mapping(action, state[:, 28:42])
            velocity = 0.9 * state[:, 14:28] + 0.1 * (command - state[:, :14])
            next_state = torch.cat((state[:, :14] + 0.02 * velocity, velocity, command), dim=-1)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
    windows = {
        "state": torch.stack(states, dim=1),
        "action": torch.stack(actions, dim=1),
        "next_state": torch.stack(next_states, dim=1),
    }
    model = DeltaWorldModel(hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    initial_loss = float(model.autoregressive_loss(windows, mapping)[0].detach())
    for _ in range(80):
        optimizer.zero_grad()
        loss, errors = model.autoregressive_loss(windows, mapping)
        assert errors.shape == (48, 8)
        loss.backward()
        optimizer.step()
    final_loss = float(model.autoregressive_loss(windows, mapping)[0].detach())
    assert final_loss < initial_loss * 0.5


def test_eight_step_loss_trains_termination_head_on_real_hard_labels():
    model = DeltaWorldModel(hidden=32)
    mapping = ArmActionMap(torch.zeros(14), -torch.ones(14), torch.ones(14), torch.ones(14))
    windows = {
        "state": torch.zeros(8, 8, 42),
        "action": torch.zeros(8, 8, 14),
        "next_state": torch.zeros(8, 8, 42),
        "termination_goal": torch.zeros(8, 8, 90),
        "terminated": torch.zeros(8, 8, dtype=torch.bool),
    }
    windows["terminated"][:4, -1] = True
    loss, _ = model.autoregressive_loss(windows, mapping)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.termination_head.parameters()
    )
