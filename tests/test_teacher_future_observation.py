from types import SimpleNamespace
import unittest

import torch
import active_adaptation as aa

aa.set_backend("mjlab")

from mimic_lite.tasks.observations.track import ref_root_pos_future_b


class TeacherFutureObservationTest(unittest.TestCase):
    def test_root_position_selects_requested_teacher_steps(self) -> None:
        observation = object.__new__(ref_root_pos_future_b)
        observation.command_manager = SimpleNamespace(
            ref_root_pos_future_b=torch.arange(18).reshape(1, 3, 6)
        )
        observation.env = SimpleNamespace(num_envs=1)
        observation.future_step_indices = torch.tensor([0, 2])

        torch.testing.assert_close(
            observation.compute(),
            torch.cat(
                (
                    observation.command_manager.ref_root_pos_future_b[:, 0],
                    observation.command_manager.ref_root_pos_future_b[:, 2],
                ),
                dim=1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
