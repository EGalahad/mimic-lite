from active_adaptation.envs.mdp.randomizations.base import Randomization
from typing import TYPE_CHECKING, Tuple
import torch

try:
    import isaaclab.utils.string as string_utils
except ModuleNotFoundError:
    from mjlab.utils.lab_api import string as string_utils

if TYPE_CHECKING:
    from mimic_lite.tasks.actions import JointPosition

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset = self.env.scene.articulations["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager: "JointPosition" = self.env.input_managers["action"]

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[:, 0], self.offset_range[:, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset
