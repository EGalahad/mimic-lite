import torch

from mimic_lite.tasks.transforms import _spatial_motion_from_local_poses


def _compose_body_frames(
    position: torch.Tensor,
    rotation: torch.Tensor,
    offset_position: torch.Tensor,
    offset_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        position
        + torch.einsum("nhbij,bj->nhbi", rotation, offset_position),
        rotation @ offset_rotation[None, None],
    )


def test_body_spatial_motion_is_independently_frame_invariant():
    dtype = torch.float64
    angles = torch.tensor(
        [[[0.0, 0.2], [0.3, -0.1], [-0.4, 0.5], [0.8, -0.6]]],
        dtype=dtype,
    )
    cos, sin = angles.cos(), angles.sin()
    rotation = torch.zeros(1, 4, 2, 3, 3, dtype=dtype)
    rotation[..., 0, 0] = cos
    rotation[..., 0, 1] = -sin
    rotation[..., 1, 0] = sin
    rotation[..., 1, 1] = cos
    rotation[..., 2, 2] = 1.0
    position = torch.arange(24, dtype=dtype).reshape(1, 4, 2, 3) / 10.0
    current_index = 1

    offset_angles = torch.tensor([0.6, -0.35], dtype=dtype)
    offset_cos, offset_sin = offset_angles.cos(), offset_angles.sin()
    offset_rotation = torch.zeros(2, 3, 3, dtype=dtype)
    offset_rotation[:, 0, 0] = offset_cos
    offset_rotation[:, 0, 2] = offset_sin
    offset_rotation[:, 1, 1] = 1.0
    offset_rotation[:, 2, 0] = -offset_sin
    offset_rotation[:, 2, 2] = offset_cos
    offset_position = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.4, 0.25, 0.15]], dtype=dtype
    )

    expected = _spatial_motion_from_local_poses(
        position,
        rotation,
        position[:, current_index],
        rotation[:, current_index],
    )
    reframed_position, reframed_rotation = _compose_body_frames(
        position, rotation, offset_position, offset_rotation
    )
    actual = _spatial_motion_from_local_poses(
        reframed_position,
        reframed_rotation,
        reframed_position[:, current_index],
        reframed_rotation[:, current_index],
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(
        actual[0][:, current_index], torch.zeros(1, 2, 3, dtype=dtype)
    )
    torch.testing.assert_close(
        actual[1][:, current_index],
        torch.eye(3, dtype=dtype)[None, None].expand(1, 2, 3, 3),
    )
