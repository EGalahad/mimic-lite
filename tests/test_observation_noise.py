from types import SimpleNamespace
from pathlib import Path
from hydra import compose, initialize_config_dir

import torch
import active_adaptation as aa

try:
    aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")

from active_adaptation.utils.math import matrix_from_quat
from mimic_lite.tasks.observations.common import (
    random_noise, add_spherical_noise, perturb_quaternion, projected_gravity_history,
)
from mimic_lite.tasks.observations.track import ref_root_ori_future_b


def test_noise_bounds_rotation_and_episode_bias():
    cfg_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        for stage in ("train", "adapt", "finetune"):
            cfg = compose(config_name="train", overrides=[f"+exp=ppo_roa/{stage}"])
            assert cfg.task.observation.policy.projected_gravity_history.bias_noise_std == 0.03
            assert cfg.task.observation.policy.joint_pos_history.noise_std == 0.01
            for name in ("command", "command_long"):
                command = cfg.task.observation[name]
                assert command.ref_joint_pos_future.noise_std == 0.01
                assert command.ref_root_ori_future_b.noise_std == 0.02
    torch.manual_seed(7)
    x = torch.zeros(20000, 3)
    uniform = random_noise(x, 0.01)
    assert uniform.abs().max() <= 0.01
    assert abs(uniform.std().item() - 0.01 / 3**0.5) < 0.0001
    assert add_spherical_noise(x, 0.005).norm(dim=-1).max() <= 0.005001
    q = torch.zeros(20000, 4)
    q[:, 0] = 1
    noisy = perturb_quaternion(q, 0.02)
    torch.testing.assert_close(noisy.norm(dim=-1), torch.ones(20000))
    assert (2 * torch.atan2(noisy[:, 1:].norm(dim=-1), noisy[:, 0])).max() <= 0.020001

    env = SimpleNamespace(num_envs=4, device=torch.device("cpu"), command_manager=None,
        scene=SimpleNamespace(articulations={"robot": SimpleNamespace(
            data=SimpleNamespace(root_link_quat_w=q[:4].clone()))}))
    obs = projected_gravity_history(noise_std=0.0, bias_noise_std=0.03, history_steps=[0, 1])
    obs._initialize(env)
    bias = obs.bias_quat.clone()
    before = obs.compute().clone()
    obs.update()
    torch.testing.assert_close(obs.compute(), before)
    torch.testing.assert_close(obs.bias_quat, bias)
    obs.reset(torch.tensor([1]))
    torch.testing.assert_close(obs.bias_quat[[0, 2, 3]], bias[[0, 2, 3]])
    assert not torch.equal(obs.bias_quat[1], bias[1])
    torch.testing.assert_close(obs.buffer.norm(dim=-1), torch.ones(4, 2))

    matrices = matrix_from_quat(q[:4]).unsqueeze(1)
    env.command_manager = SimpleNamespace(future_steps=torch.tensor([0]), ref_root_ori_future_b_matrix=matrices)
    ori = ref_root_ori_future_b(noise_std=0.02, future_steps=[0])
    ori._initialize(env)
    rows = ori.compute().reshape(4, 2, 3)
    torch.testing.assert_close(rows @ rows.transpose(-1, -2), torch.eye(2).expand(4, 2, 2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(matrices, matrix_from_quat(q[:4]).unsqueeze(1))


if __name__ == "__main__":
    test_noise_bounds_rotation_and_episode_bias()
    print("observation noise checks passed")
