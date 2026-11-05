import mujoco
import numpy as np
import torch
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK

MJCF = "/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"


def test_torch_fk_matches_mujoco_on_random_legal_arms_and_has_gradients():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    fk = DifferentiableMarvinFK(MJCF, dtype=torch.float64)
    names = fk.names
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    addresses = [model.jnt_qposadr[j] for j in ids]
    limits = model.jnt_range[ids]
    rng = np.random.default_rng(140116)
    q = rng.uniform(limits[:, 0] + 0.1, limits[:, 1] - 0.1, size=(8, 14))
    qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    position, rotation = fk(qt)
    for i in range(len(q)):
        data.qpos[:] = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
        data.qpos[addresses] = q[i]
        mujoco.mj_forward(model, data)
        for side, suffix in enumerate(("L", "R")):
            for link, name in enumerate((f"Link3_{suffix}", f"Link4_{suffix}", f"tcp_{suffix}")):
                obj = mujoco.mjtObj.mjOBJ_SITE if link == 2 else mujoco.mjtObj.mjOBJ_BODY
                ident = mujoco.mj_name2id(model, obj, name)
                p = data.site_xpos[ident] if link == 2 else data.xpos[ident]
                r = data.site_xmat[ident].reshape(3, 3) if link == 2 else data.xmat[ident].reshape(3, 3)
                np.testing.assert_allclose(position[i, side, link].detach().numpy(), p, atol=1e-7)
                np.testing.assert_allclose(rotation[i, side, link].detach().numpy(), r, atol=1e-7)
    loss = position[:, :, 2].square().sum() + rotation[:, :, 2].square().sum()
    loss.backward()
    assert torch.isfinite(qt.grad).all()
    assert torch.all(qt.grad.abs().sum(dim=0) > 1e-7)
    eps = 1e-5
    plus = qt.detach().clone()
    minus = qt.detach().clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    numeric = (fk(plus)[0][0, 0, 2, 0] - fk(minus)[0][0, 0, 2, 0]) / (2 * eps)
    analytical = torch.autograd.grad(fk(qt)[0][0, 0, 2, 0], qt)[0][0, 0]
    torch.testing.assert_close(numeric, analytical, atol=1e-5, rtol=1e-3)
