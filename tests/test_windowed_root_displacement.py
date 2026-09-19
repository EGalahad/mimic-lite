import math
from types import SimpleNamespace

import torch

from mimic_lite.tasks.rewards.track import (
    WindowedRootDisplacementBuffer,
    body_pos_exp,
    windowed_root_displacement_exp,
)


def _run(
    robot: torch.Tensor,
    reference: torch.Tensor,
    history_steps: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    buffer = WindowedRootDisplacementBuffer(robot.shape[1], history_steps, "cpu")
    errors = []
    residuals = []
    for robot_step, reference_step in zip(robot, reference):
        error, residual = buffer.update(robot_step, reference_step)
        errors.append(error)
        residuals.append(residual)
    return torch.stack(errors), torch.stack(residuals)


def test_exact_tracking_has_zero_windowed_root_error() -> None:
    time = torch.arange(12, dtype=torch.float32)
    reference = torch.stack((0.2 * time, -0.1 * time), dim=-1)[:, None]
    errors, _ = _run(reference.clone(), reference, (5,))
    torch.testing.assert_close(errors, torch.zeros_like(errors))


def test_windowed_root_error_uses_displacement_residual() -> None:
    time = torch.arange(12, dtype=torch.float32)
    reference = torch.stack((0.05 * time.square(), torch.zeros_like(time)), dim=-1)[
        :, None
    ]
    robot = reference * torch.tensor([0.8, 1.0])
    errors, residuals = _run(robot, reference, (5,))
    expected = (robot[10] - robot[5]) - (reference[10] - reference[5])
    torch.testing.assert_close(residuals[10], expected)
    torch.testing.assert_close(errors[10], expected.norm(dim=-1))


def test_windowed_root_reset_drops_selected_history() -> None:
    buffer = WindowedRootDisplacementBuffer(1, (3,), "cpu")
    for step in range(4):
        buffer.update(
            torch.tensor([[10.0 + step, 0.0]]),
            torch.tensor([[float(step), 0.0]]),
        )
    buffer.reset(torch.tensor([0]))
    error, residual = buffer.update(
        torch.tensor([[1.2, -0.1]]),
        torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(residual, torch.tensor([[0.2, -0.1]]))
    torch.testing.assert_close(error, torch.tensor([math.sqrt(0.05)]))


def test_windowed_xyz_keeps_absolute_height_and_resets_selected_envs() -> None:
    # XYZ-axis cases plus a combined residual; select torso rather than body 0.
    offsets = torch.tensor([[0.15, 0, 0], [0, 0.2, 0], [0, 0, 0.3], [0.3, 0.4, 1.2]])
    reference = torch.zeros(4, 2, 3)
    reference[:, 1, 2] = 0.8
    robot = reference.clone()
    robot[:, 1] += offsets
    reward = SimpleNamespace(
        body_indices_tracking=[1],
        command_manager=SimpleNamespace(
            robot_body_link_pos_w=robot,
            ref_body_pos_w=reference,
        ),
        history=WindowedRootDisplacementBuffer(4, (200,), "cpu"),
        error=torch.zeros(4),
        sigma=0.3,
    )
    for step in range(201):
        robot[:, 1, 0] += 0.02
        reference[:, 1, 0] += 0.02
        windowed_root_displacement_exp.update(reward)
        expected = offsets.norm(dim=-1) if step < 200 else offsets[:, 2].abs()
        torch.testing.assert_close(reward.error, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        windowed_root_displacement_exp._compute(reward),
        torch.exp(-offsets[:, 2].abs() / reward.sigma).unsqueeze(1),
        atol=1e-5, rtol=1e-5,
    )

    windowed_root_displacement_exp.reset(reward, torch.tensor([0, 3]))
    windowed_root_displacement_exp.update(reward)
    torch.testing.assert_close(reward.error, torch.tensor([0.15, 0, 0.3, 1.3]), atol=1e-5, rtol=1e-5)
    assert reward.history.robot_history.shape == (4, 201, 2)


def test_global_root_reward_keeps_xyz_offset() -> None:
    offsets = torch.tensor([[0.15, 0, 0], [0, 0.2, 0], [0, 0, 0.3], [0.3, 0.4, 1.2]])
    reward = SimpleNamespace(
        body_indices_tracking=[1],
        command_manager=SimpleNamespace(
            body_pos_error=torch.stack((torch.zeros(4), offsets.norm(dim=-1)), dim=1),
        ),
        sigma=0.3,
    )
    torch.testing.assert_close(
        body_pos_exp._compute(reward),
        torch.exp(-offsets.norm(dim=-1) / reward.sigma).unsqueeze(1),
    )
