from types import SimpleNamespace
import torch
from active_adaptation.utils.rollout_compile import compile_rollout_function
from mimic_lite.tasks.observations.common import joint_vel_history


def test_history_compiled_read_wraparound_and_partial_reset():
    term = object.__new__(joint_vel_history)
    term._initialized = True
    term.env = SimpleNamespace(num_envs=3, device=torch.device('cpu'))
    term.buffer_size = 5
    term.head = 0
    term.history_offsets = torch.tensor([0, 1, 4])
    term.history_indices = term.history_offsets.clone()
    term.buffer = torch.zeros(3, 5, 2)
    term.joint_ids = torch.tensor([0, 1])
    term.noise_std = 0.0
    term.asset = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(3, 2)))
    compiled = compile_rollout_function(term.compute)
    history = torch.zeros(3, 5, 2)
    for step in range(13):
        term.asset.data.joint_vel.fill_(step + 1)
        term.update()
        history = torch.roll(history, 1, dims=1)
        history[:, 0] = step + 1
        if step == 7:
            term.reset(torch.tensor([1]))
            history[1].fill_(step + 1)
        torch.testing.assert_close(compiled(), history[:, [0, 1, 4]].reshape(3, -1))
