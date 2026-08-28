from types import SimpleNamespace

import torch

from mimic_lite.tasks.rewards.feet import feet_air_time


def test_feet_air_time_reward_math_is_independent_of_debug_visualization():
    reward = feet_air_time.__new__(feet_air_time)
    reward.reward_time = torch.tensor([[0.05, 0.20]])
    reward.is_first_contact = torch.tensor([[True, True]])
    reward.thres = 0.10
    reward.command_manager = SimpleNamespace(
        is_standing_env=torch.tensor([[False]])
    )

    torch.testing.assert_close(reward._compute(), torch.tensor([[-0.05]]))
