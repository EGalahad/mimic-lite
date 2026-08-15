import math

import torch

from mimic_lite.tasks.rewards.track import WindowedRootDisplacementBuffer


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
