"""Convert native-timeline Bumi qpos into the 50 Hz MimicLite NPZ contract."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


BUMI_POLICY_JOINT_NAMES = (
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
)

BUMI_MOTION_JOINT_NAMES = (
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)

BUMI_BODY_NAMES = (
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
)

BUMI_MOTION_BODY_NAMES = ("motion_root", *BUMI_BODY_NAMES)
TARGET_TRACKER_FPS = 50.0
BUMI_ROOT_LINEAR_VELOCITY_LIMIT = 3.0
BUMI_ROOT_ANGULAR_VELOCITY_LIMIT = 6.0
BUMI_FOOT_CONTACT_GEOM_NAMES = (
    (
        "l_foot_contact_0_geom",
        "l_foot_contact_1_geom",
        "l_foot_contact_2_geom",
        "l_foot_contact_3_geom",
        "l_foot1_collision",
        "l_foot2_collision",
        "l_foot3_collision",
        "l_foot4_collision",
        "l_foot5_collision",
        "l_foot6_collision",
        "l_foot7_collision",
    ),
    (
        "r_foot_contact_0_geom",
        "r_foot_contact_1_geom",
        "r_foot_contact_2_geom",
        "r_foot_contact_3_geom",
        "r_foot1_collision",
        "r_foot2_collision",
        "r_foot3_collision",
        "r_foot4_collision",
        "r_foot5_collision",
        "r_foot6_collision",
        "r_foot7_collision",
    ),
)

# Pipeline-v3 compatibility constants for ``align_bumi_qpos_to_ground``.
# Pipeline v4 no longer uses that post-GMR translation in batch conversion;
# the physical collision sole remains the source of read-only ground audits.
BUMI_GROUND_REFERENCE_PERCENTILE = 0.1
BUMI_GROUND_TARGET_FOOT_CONTACT_HEIGHT_M = 0.002
BUMI_GROUND_MAX_ABS_OFFSET_M = 0.05
BUMI_GROUND_MAX_PENETRATION_M = 0.005
BUMI_GROUND_MAX_PENETRATING_FRAME_FRACTION = 0.002
BUMI_CONTACT_FOOT_HEIGHT_P05_MIN_M = -0.015
BUMI_CONTACT_FOOT_HEIGHT_MEDIAN_MAX_M = 0.015
BUMI_DOUBLE_SUPPORT_LOW_FOOT_MEDIAN_MIN_M = -0.005
BUMI_DOUBLE_SUPPORT_HIGH_FOOT_MEDIAN_MAX_M = 0.020

# Conservative v1 dataset gates.  These are deliberately stricter than some
# actuator limits: a clip may be physically rate-limited yet still be too
# aggressive to enter tracker training without replay/review.
BUMI_V1_TRAINING_MAX_JOINT_VELOCITY_ABS = 20.0
BUMI_V1_TRAINING_MAX_BODY_LINEAR_VELOCITY_NORM = 10.0
BUMI_V1_TRAINING_MAX_BODY_ANGULAR_VELOCITY_NORM = 30.0


def _joint_velocity_limit(joint_name: str) -> float:
    if "arm" in joint_name or "elbow" in joint_name:
        return 50.0
    if joint_name == "waist_yaw_joint" or "leg_yaw" in joint_name:
        return 9.0
    return 12.0


BUMI_JOINT_VELOCITY_LIMITS = {
    name: _joint_velocity_limit(name) for name in BUMI_POLICY_JOINT_NAMES
}

BUMI_JOINT_POSITION_LIMITS = {
    "waist_yaw_joint": (-1.57, 1.57),
    "l_arm_pitch_joint": (-3.14, 1.57),
    "l_arm_roll_joint": (-0.14, 1.94),
    "l_arm_yaw_joint": (-1.57, 1.57),
    "l_elbow_pitch_joint": (-2.26, 0.0),
    "r_arm_pitch_joint": (-3.14, 1.57),
    "r_arm_roll_joint": (-1.94, 0.14),
    "r_arm_yaw_joint": (-1.57, 1.57),
    "r_elbow_pitch_joint": (-2.26, 0.0),
    "l_leg_pitch_joint": (-2.09, 2.09),
    "l_leg_roll_joint": (-0.66, 1.57),
    "l_leg_yaw_joint": (-2.53, 2.53),
    "l_knee_pitch_joint": (0.0, 2.24),
    "l_ankle_pitch_joint": (-0.96, 0.44),
    "l_ankle_roll_joint": (-0.17, 0.17),
    "r_leg_pitch_joint": (-2.09, 2.09),
    "r_leg_roll_joint": (-1.57, 0.66),
    "r_leg_yaw_joint": (-2.53, 2.53),
    "r_knee_pitch_joint": (0.0, 2.24),
    "r_ankle_pitch_joint": (-0.96, 0.44),
    "r_ankle_roll_joint": (-0.17, 0.17),
}

BUMI_NOMINAL_JOINT_POS = {
    "l_arm_roll_joint": 0.3,
    "r_arm_roll_joint": -0.3,
    "l_leg_pitch_joint": -0.1495,
    "r_leg_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "r_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.172,
    "r_ankle_pitch_joint": -0.172,
}


def _joint_addresses(
    model: mujoco.MjModel,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Bumi model is missing joint {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Bumi joint {name!r} is not a hinge")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def validate_bumi_model(model: mujoco.MjModel) -> None:
    if model.nq != 28 or model.nv != 27:
        raise ValueError(f"Expected Bumi nq=28/nv=27, got nq={model.nq}/nv={model.nv}")
    if model.nbody - 1 != 22:
        raise ValueError(f"Expected 22 Bumi bodies, got {model.nbody - 1}")
    model_joint_names = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        model_joint_names.append(str(name))
    if tuple(model_joint_names) != BUMI_POLICY_JOINT_NAMES:
        raise ValueError(
            "Bumi model joint order mismatch: "
            f"expected {BUMI_POLICY_JOINT_NAMES}, got {tuple(model_joint_names)}"
        )
    for joint_name in BUMI_POLICY_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if not np.allclose(
            model.jnt_range[joint_id],
            BUMI_JOINT_POSITION_LIMITS[joint_name],
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Bumi position limit mismatch for {joint_name}: "
                f"{model.jnt_range[joint_id]} / "
                f"{BUMI_JOINT_POSITION_LIMITS[joint_name]}"
            )
    model_body_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        for body_id in range(1, model.nbody)
    )
    if model_body_names != BUMI_BODY_NAMES:
        raise ValueError(
            f"Bumi model body order mismatch: expected {BUMI_BODY_NAMES}, got {model_body_names}"
        )


def nominal_bumi_qpos(model: mujoco.MjModel) -> np.ndarray:
    validate_bumi_model(model)
    qpos = model.qpos0.copy()
    qpos[:3] = np.asarray([0.0, 0.0, 0.472])
    qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    qpos_addresses, _ = _joint_addresses(model, BUMI_POLICY_JOINT_NAMES)
    qpos[qpos_addresses] = np.asarray(
        [BUMI_NOMINAL_JOINT_POS.get(name, 0.0) for name in BUMI_POLICY_JOINT_NAMES]
    )
    return qpos


def bumi_foot_contact_heights(
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> np.ndarray:
    """Return the lowest collision-sole world Z for each foot and frame."""

    validate_bumi_model(model)
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos shape (frames, {model.nq}), got {qpos.shape}")
    if qpos.shape[0] == 0:
        raise ValueError("qpos must contain at least one frame")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos contains non-finite values")

    foot_geoms: list[list[tuple[int, int, float, float]]] = []
    for names in BUMI_FOOT_CONTACT_GEOM_NAMES:
        geoms = []
        for name in names:
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            if geom_id < 0:
                raise ValueError(f"Bumi model is missing foot contact geom {name!r}")
            geom_type = int(model.geom_type[geom_id])
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                half_length = 0.0
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half_length = float(model.geom_size[geom_id, 1])
            else:
                raise ValueError(
                    f"Unsupported foot contact geom type for {name!r}: {geom_type}"
                )
            geoms.append(
                (
                    geom_id,
                    geom_type,
                    float(model.geom_size[geom_id, 0]),
                    half_length,
                )
            )
        foot_geoms.append(geoms)

    heights = np.empty((qpos.shape[0], 2), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame_idx, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for foot_idx, geoms in enumerate(foot_geoms):
            lowest = np.inf
            for geom_id, geom_type, radius, half_length in geoms:
                height = float(data.geom_xpos[geom_id, 2]) - radius
                if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    rotation = data.geom_xmat[geom_id].reshape(3, 3)
                    height -= abs(float(rotation[2, 2])) * half_length
                lowest = min(lowest, height)
            heights[frame_idx, foot_idx] = lowest
    return heights


def audit_bumi_ground_contact(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    source_foot_contact_mask: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Measure physical Bumi sole clearance without changing the motion.

    All-foot extrema remain visible for debugging, but the training gate uses
    source-contact feet.  A pitched swing foot must not force the entire robot
    upward and make the stance foot float.
    """

    foot_contact_heights = bumi_foot_contact_heights(model, qpos)
    support_foot_height = foot_contact_heights.min(axis=1)
    below_ground_fraction = float(np.mean(support_foot_height < 0.0))
    below_penetration_limit_fraction = float(
        np.mean(
            support_foot_height
            < -BUMI_GROUND_MAX_PENETRATION_M
        )
    )
    minimum_height = float(support_foot_height.min())
    result: dict[str, float | int | bool] = {
        "native_min_foot_contact_height_m": minimum_height,
        "native_support_foot_height_p05_m": float(
            np.percentile(support_foot_height, 5.0)
        ),
        "native_support_foot_height_median_m": float(
            np.median(support_foot_height)
        ),
        "native_foot_contact_below_ground_fraction": (
            below_ground_fraction
        ),
        "native_foot_contact_below_minus_5mm_fraction": (
            below_penetration_limit_fraction
        ),
    }
    if source_foot_contact_mask is None:
        result["gate_native_ground_contact_pass"] = bool(
            minimum_height >= -BUMI_GROUND_MAX_PENETRATION_M
            and below_ground_fraction
            <= BUMI_GROUND_MAX_PENETRATING_FRAME_FRACTION
        )
        return result

    contact_mask = np.asarray(source_foot_contact_mask, dtype=bool)
    if contact_mask.shape != foot_contact_heights.shape:
        raise ValueError(
            "source_foot_contact_mask must have shape "
            f"{foot_contact_heights.shape}, got {contact_mask.shape}"
        )
    if not np.any(contact_mask):
        raise ValueError("source_foot_contact_mask contains no contact samples")

    contact_heights = foot_contact_heights[contact_mask]
    double_support_mask = np.all(contact_mask, axis=1)
    contact_p05 = float(np.percentile(contact_heights, 5.0))
    contact_median = float(np.median(contact_heights))
    result.update(
        {
            "source_contact_sample_count": int(contact_heights.size),
            "source_double_support_frame_count": int(
                np.count_nonzero(double_support_mask)
            ),
            "native_contact_foot_height_p05_m": contact_p05,
            "native_contact_foot_height_median_m": contact_median,
            "native_contact_foot_below_minus_5mm_fraction": float(
                np.mean(contact_heights < -BUMI_GROUND_MAX_PENETRATION_M)
            ),
        }
    )
    if np.any(double_support_mask):
        double_support_heights = foot_contact_heights[double_support_mask]
        double_low_median = float(
            np.median(double_support_heights.min(axis=1))
        )
        double_high_median = float(
            np.median(double_support_heights.max(axis=1))
        )
        result.update(
            {
                "native_double_support_low_foot_height_median_m": (
                    double_low_median
                ),
                "native_double_support_high_foot_height_median_m": (
                    double_high_median
                ),
            }
        )
        double_support_pass = bool(
            double_low_median
            >= BUMI_DOUBLE_SUPPORT_LOW_FOOT_MEDIAN_MIN_M
            and double_high_median
            <= BUMI_DOUBLE_SUPPORT_HIGH_FOOT_MEDIAN_MAX_M
        )
    else:
        double_support_pass = True

    result["gate_native_ground_contact_pass"] = bool(
        contact_p05 >= BUMI_CONTACT_FOOT_HEIGHT_P05_MIN_M
        and contact_median <= BUMI_CONTACT_FOOT_HEIGHT_MEDIAN_MAX_M
        and double_support_pass
    )
    return result


def align_bumi_qpos_to_ground(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    reference_percentile: float = BUMI_GROUND_REFERENCE_PERCENTILE,
    target_foot_contact_height_m: float = (
        BUMI_GROUND_TARGET_FOOT_CONTACT_HEIGHT_M
    ),
    max_abs_offset_m: float = BUMI_GROUND_MAX_ABS_OFFSET_M,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Apply one bounded root-Z translation to align a clip with world Z=0.

    The reference is computed from the lower physical collision sole in each
    frame, including toe/heel effects from foot pitch and roll.  No per-frame
    foot locking is performed: the returned motion differs from the input by
    one constant vertical translation only.
    """

    if (
        not np.isfinite(reference_percentile)
        or reference_percentile < 0.0
        or reference_percentile > 100.0
    ):
        raise ValueError(
            "reference_percentile must be finite and in [0, 100], got "
            f"{reference_percentile}"
        )
    if not np.isfinite(target_foot_contact_height_m):
        raise ValueError(
            "target_foot_contact_height_m must be finite, got "
            f"{target_foot_contact_height_m}"
        )
    if not np.isfinite(max_abs_offset_m) or max_abs_offset_m <= 0.0:
        raise ValueError(
            f"max_abs_offset_m must be finite and positive, got {max_abs_offset_m}"
        )

    source = np.asarray(qpos, dtype=np.float64)
    foot_contact_heights = bumi_foot_contact_heights(model, source)
    minimum_foot_height = foot_contact_heights.min(axis=1)
    reference_before = float(
        np.percentile(minimum_foot_height, reference_percentile)
    )
    requested_offset = float(target_foot_contact_height_m - reference_before)
    applied_offset = float(
        np.clip(requested_offset, -max_abs_offset_m, max_abs_offset_m)
    )
    alignment_clipped = not np.isclose(
        requested_offset,
        applied_offset,
        atol=1.0e-12,
        rtol=0.0,
    )

    aligned = source.copy()
    aligned[:, 2] += applied_offset
    minimum_foot_height_after = minimum_foot_height + applied_offset
    below_ground_before = float(np.mean(minimum_foot_height < 0.0))
    below_ground_after = float(np.mean(minimum_foot_height_after < 0.0))
    below_penetration_limit_after = float(
        np.mean(
            minimum_foot_height_after
            < -BUMI_GROUND_MAX_PENETRATION_M
        )
    )
    minimum_after = float(minimum_foot_height_after.min())
    gate_pass = bool(
        not alignment_clipped
        and minimum_after >= -BUMI_GROUND_MAX_PENETRATION_M
        and below_ground_after
        <= BUMI_GROUND_MAX_PENETRATING_FRAME_FRACTION
    )
    return aligned, {
        "ground_alignment_reference_percentile": float(reference_percentile),
        "ground_target_foot_contact_height_m": float(
            target_foot_contact_height_m
        ),
        "ground_reference_foot_contact_height_before_m": reference_before,
        "ground_reference_foot_contact_height_after_m": float(
            reference_before + applied_offset
        ),
        "ground_requested_root_z_offset_m": requested_offset,
        "ground_applied_root_z_offset_m": applied_offset,
        "ground_alignment_clipped": bool(alignment_clipped),
        "min_foot_contact_height_before_m": float(minimum_foot_height.min()),
        "min_foot_contact_height_after_m": minimum_after,
        "foot_contact_below_ground_fraction_before": below_ground_before,
        "foot_contact_below_ground_fraction_after": below_ground_after,
        "foot_contact_below_minus_5mm_fraction_after": (
            below_penetration_limit_after
        ),
        "gate_ground_contact_pass": gate_pass,
    }


def _normalize_continuous_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (frames, 4), got {result.shape}")
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("qpos contains a zero-length root quaternion")
    result /= norms
    for frame_idx in range(1, result.shape[0]):
        if np.dot(result[frame_idx - 1], result[frame_idx]) < 0.0:
            result[frame_idx] *= -1.0
    return result


def target_timestamps(
    source_timestamps_s: np.ndarray,
    *,
    target_fps: float,
) -> np.ndarray:
    timestamps = np.asarray(source_timestamps_s, dtype=np.float64).reshape(-1)
    if timestamps.shape[0] == 0:
        raise ValueError("source_timestamps_s must not be empty")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("source_timestamps_s contains non-finite values")
    if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("source_timestamps_s must be strictly increasing")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")
    if timestamps.shape[0] == 1:
        return timestamps.copy()
    duration = float(timestamps[-1] - timestamps[0])
    target_length = int(np.floor(duration * target_fps + 1.0e-9)) + 1
    return timestamps[0] + np.arange(target_length, dtype=np.float64) / target_fps


def _linear_interpolate(
    source_timestamps_s: np.ndarray,
    target_timestamps_s: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    output = np.stack(
        [
            np.interp(target_timestamps_s, source_timestamps_s, flat[:, column])
            for column in range(flat.shape[1])
        ],
        axis=-1,
    )
    return output.reshape(target_timestamps_s.shape[0], *values.shape[1:])


def resample_bumi_qpos(
    model: mujoco.MjModel,
    source_timestamps_s: np.ndarray,
    source_qpos: np.ndarray,
    *,
    target_fps: float = TARGET_TRACKER_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample root/joints once; derived FK and velocities are not inputs."""

    validate_bumi_model(model)
    source_timestamps_s = np.asarray(source_timestamps_s, dtype=np.float64).reshape(-1)
    source_qpos = np.asarray(source_qpos, dtype=np.float64)
    if source_qpos.shape != (source_timestamps_s.shape[0], model.nq):
        raise ValueError(
            f"source_qpos must have shape ({source_timestamps_s.shape[0]}, {model.nq}), "
            f"got {source_qpos.shape}"
        )
    if not np.all(np.isfinite(source_qpos)):
        raise ValueError("source_qpos contains non-finite values")
    target_times = target_timestamps(source_timestamps_s, target_fps=target_fps)
    if source_timestamps_s.shape[0] == 1:
        return target_times, source_qpos.copy()

    output = np.repeat(model.qpos0[None, :], target_times.shape[0], axis=0)
    output[:, :3] = _linear_interpolate(
        source_timestamps_s,
        target_times,
        source_qpos[:, :3],
    )
    source_quaternions = _normalize_continuous_quaternions_wxyz(source_qpos[:, 3:7])
    rotations = R.from_quat(source_quaternions[:, [1, 2, 3, 0]])
    interpolated_xyzw = Slerp(source_timestamps_s, rotations)(target_times).as_quat()
    output[:, 3:7] = interpolated_xyzw[:, [3, 0, 1, 2]]

    joint_qpos_addresses, _ = _joint_addresses(model, BUMI_POLICY_JOINT_NAMES)
    output[:, joint_qpos_addresses] = _linear_interpolate(
        source_timestamps_s,
        target_times,
        source_qpos[:, joint_qpos_addresses],
    )

    for joint_name, qpos_address in zip(
        BUMI_POLICY_JOINT_NAMES,
        joint_qpos_addresses,
        strict=True,
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lower, upper = model.jnt_range[joint_id]
        minimum = float(output[:, qpos_address].min())
        maximum = float(output[:, qpos_address].max())
        if minimum < lower - 1.0e-6 or maximum > upper + 1.0e-6:
            raise ValueError(
                f"Resampled {joint_name} violates [{lower}, {upper}]: [{minimum}, {maximum}]"
            )
        output[:, qpos_address] = np.clip(output[:, qpos_address], lower, upper)
    return target_times, output


def differentiate_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    timestamps_s: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if qpos.shape != (timestamps_s.shape[0], model.nq):
        raise ValueError(f"qpos/timestamps shape mismatch: {qpos.shape}/{timestamps_s.shape}")
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    if qpos.shape[0] == 1:
        return qvel
    for frame_idx in range(qpos.shape[0]):
        if frame_idx == 0:
            left, right = 0, 1
        elif frame_idx == qpos.shape[0] - 1:
            left, right = qpos.shape[0] - 2, qpos.shape[0] - 1
        else:
            left, right = frame_idx - 1, frame_idx + 1
        dt = float(timestamps_s[right] - timestamps_s[left])
        if dt <= 0.0:
            raise ValueError(f"Non-positive differentiation dt at frame {frame_idx}: {dt}")
        mujoco.mj_differentiatePos(
            model,
            qvel[frame_idx],
            dt,
            qpos[left],
            qpos[right],
        )
    return qvel


def materialize_bumi_tracking_motion(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    timestamps_s: np.ndarray,
) -> dict[str, np.ndarray]:
    validate_bumi_model(model)
    qpos = np.asarray(qpos, dtype=np.float64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    qvel = differentiate_qpos(model, qpos, timestamps_s)
    motion_qpos_addresses, motion_dof_addresses = _joint_addresses(
        model,
        BUMI_MOTION_JOINT_NAMES,
    )
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in BUMI_BODY_NAMES
        ],
        dtype=np.int64,
    )
    if np.any(body_ids < 0):
        raise ValueError("Bumi model is missing a required body")

    frames = qpos.shape[0]
    body_pos_w = np.zeros((frames, 23, 3), dtype=np.float64)
    body_quat_w = np.zeros((frames, 23, 4), dtype=np.float64)
    body_lin_vel_w = np.zeros((frames, 23, 3), dtype=np.float64)
    body_ang_vel_w = np.zeros((frames, 23, 3), dtype=np.float64)
    body_quat_w[:, 0, 0] = 1.0

    data = mujoco.MjData(model)
    object_velocity = np.empty(6, dtype=np.float64)
    for frame_idx in range(frames):
        data.qpos[:] = qpos[frame_idx]
        data.qvel[:] = qvel[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos_w[frame_idx, 1:] = data.xpos[body_ids]
        body_quat_w[frame_idx, 1:] = data.xquat[body_ids]
        for output_body_idx, body_id in enumerate(body_ids, start=1):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                object_velocity,
                0,
            )
            body_ang_vel_w[frame_idx, output_body_idx] = object_velocity[:3]
            body_lin_vel_w[frame_idx, output_body_idx] = object_velocity[3:]

    arrays = {
        "joint_pos": qpos[:, motion_qpos_addresses].astype(np.float32),
        "joint_vel": qvel[:, motion_dof_addresses].astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Materialized field {name} contains non-finite values")
    return arrays


def export_bumi_tracking_npz(
    output_path: str | Path,
    *,
    model_path: str | Path,
    source_timestamps_s: np.ndarray,
    source_qpos: np.ndarray,
    target_fps: float = TARGET_TRACKER_FPS,
) -> Mapping[str, float | int | str | bool]:
    model_path = Path(model_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    target_times, target_qpos = resample_bumi_qpos(
        model,
        source_timestamps_s,
        source_qpos,
        target_fps=target_fps,
    )
    arrays = materialize_bumi_tracking_motion(model, target_qpos, target_times)
    tracker_foot_contact_height = bumi_foot_contact_heights(
        model,
        target_qpos,
    ).min(axis=1)
    tracker_below_ground_fraction = float(
        np.mean(tracker_foot_contact_height < 0.0)
    )
    tracker_below_penetration_limit_fraction = float(
        np.mean(
            tracker_foot_contact_height
            < -BUMI_GROUND_MAX_PENETRATION_M
        )
    )
    tracker_minimum_foot_contact_height = float(
        tracker_foot_contact_height.min()
    )
    tracker_ground_contact_pass = bool(
        tracker_minimum_foot_contact_height
        >= -BUMI_GROUND_MAX_PENETRATION_M
        and tracker_below_ground_fraction
        <= BUMI_GROUND_MAX_PENETRATING_FRAME_FRACTION
    )
    rounded_fps = int(round(target_fps))
    if not np.isclose(target_fps, rounded_fps):
        raise ValueError(f"Tracking NPZ requires integer FPS, got {target_fps}")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".npz",
            dir=output.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        np.savez_compressed(
            temp_path,
            fps=np.asarray([rounded_fps], dtype=np.int32),
            **arrays,
        )
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    source_timestamps_s = np.asarray(source_timestamps_s, dtype=np.float64)
    source_duration = float(source_timestamps_s[-1] - source_timestamps_s[0])
    target_duration = float(target_times[-1] - target_times[0])
    return {
        "output": str(output),
        "frames": int(target_times.shape[0]),
        "fps": rounded_fps,
        "source_duration_s": source_duration,
        "target_duration_s": target_duration,
        "duration_drift_s": target_duration - source_duration,
        "tracker_min_foot_contact_height_m": (
            tracker_minimum_foot_contact_height
        ),
        "tracker_foot_contact_below_ground_fraction": (
            tracker_below_ground_fraction
        ),
        "tracker_foot_contact_below_minus_5mm_fraction": (
            tracker_below_penetration_limit_fraction
        ),
        "tracker_ground_contact_pass": tracker_ground_contact_pass,
    }
