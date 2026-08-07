from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import active_adaptation as aa
import torch

try:
    aa.get_backend()
except RuntimeError:
    aa.set_backend("mjlab")

from active_adaptation.envs.mdp.randomizations.common import random_joint_offset
from active_adaptation.envs.mdp.randomizations.base import Randomization


def test_random_joint_offset_registry_uses_core_implementation() -> None:
    assert Randomization.registry["random_joint_offset"] is random_joint_offset


def test_random_joint_offset_samples_each_environment_independently() -> None:
    randomization = object.__new__(random_joint_offset)
    randomization.offset_range = torch.tensor(
        [[-0.1, 0.1], [-0.2, 0.2]],
    )
    randomization.joint_ids = torch.tensor([1, 3])
    randomization.action_manager = SimpleNamespace(offset=torch.zeros(4, 5))
    env_ids = torch.tensor([0, 2])
    sampled_offsets = torch.tensor([[0.01, 0.02], [0.03, 0.04]])

    with patch(
        "active_adaptation.envs.mdp.randomizations.common.uniform",
        return_value=sampled_offsets,
    ) as sample:
        randomization.reset(env_ids)

    low, high = sample.call_args.args
    assert low.shape == (2, 2)
    assert high.shape == (2, 2)
    torch.testing.assert_close(
        randomization.action_manager.offset[env_ids.unsqueeze(1), randomization.joint_ids],
        sampled_offsets,
    )
    torch.testing.assert_close(
        randomization.action_manager.offset[torch.tensor([1, 3])],
        torch.zeros(2, 5),
    )
