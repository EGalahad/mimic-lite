import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets.asset_cfg import (
    ActuatorCfg,
    AssetCfg,
    ContactSensorCfg,
    InitialStateCfg,
    MjlabCollisionCfg,
)
from active_adaptation.registry import Registry
from mjhub import resolve_asset_reference

registry = Registry.instance()

ATOM01_REPO_ID = "elijahgalahad/atom01"
ATOM01_REVISION = "main"
ATOM01_CAPSULE_MJCF_REF = f"hf://{ATOM01_REPO_ID}@{ATOM01_REVISION}/mjcf/atom01-capsule.xml"
ATOM01_MESH_MJCF_REF = f"hf://{ATOM01_REPO_ID}@{ATOM01_REVISION}/mjcf/atom01-mesh.xml"
ATOM01_URDF_REF = f"hf://{ATOM01_REPO_ID}@{ATOM01_REVISION}/urdf/atom01.urdf"

ATOM01_JOINT_NAMES = [
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
]

ATOM01_BODY_NAMES = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]


def _motor_actuator(
    joint_names_expr: str,
    *,
    effort: float,
    velocity: float,
    stiffness: float,
    damping: float,
    friction: float,
    armature: float,
) -> ActuatorCfg:
    return ActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=effort,
        velocity_limit=velocity,
        stiffness=stiffness,
        damping=damping,
        friction=friction,
        armature=armature,
    )


ATOM01_INIT_STATE = InitialStateCfg(
    pos=(0.0, 0.0, 0.75),
    joint_pos={
        ".*_thigh_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        "left_arm_pitch_joint": 0.18,
        "left_arm_roll_joint": 0.06,
        "left_elbow_pitch_joint": 0.78,
        "right_arm_pitch_joint": 0.18,
        "right_arm_roll_joint": -0.06,
        "right_elbow_pitch_joint": 0.78,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)


def _build_atom01_cfg(mjcf_ref: str) -> AssetCfg:
    return AssetCfg(
        mjcf_path=resolve_asset_reference(mjcf_ref),
        usd_path=resolve_asset_reference(ATOM01_URDF_REF),
        init_state=ATOM01_INIT_STATE,
        self_collisions=True,
        actuators={
            "thigh": _motor_actuator(
                ".*_thigh_yaw_joint|.*_thigh_roll_joint|.*_thigh_pitch_joint",
                effort=120.0,
                velocity=20.0,
                stiffness=100.0,
                damping=3.3,
                friction=0.01,
                armature=0.01,
            ),
            "knee": _motor_actuator(
                ".*_knee_joint",
                effort=120.0,
                velocity=20.0,
                stiffness=150.0,
                damping=5.0,
                friction=0.01,
                armature=0.01,
            ),
            "ankle": _motor_actuator(
                ".*_ankle_pitch_joint|.*_ankle_roll_joint",
                effort=27.0,
                velocity=25.0,
                stiffness=40.0,
                damping=2.0,
                friction=0.05,
                armature=0.01,
            ),
            "torso": _motor_actuator(
                "torso_joint",
                effort=120.0,
                velocity=20.0,
                stiffness=150.0,
                damping=5.0,
                friction=0.01,
                armature=0.01,
            ),
            "shoulder": _motor_actuator(
                ".*_arm_pitch_joint|.*_arm_roll_joint|.*_arm_yaw_joint",
                effort=27.0,
                velocity=25.0,
                stiffness=40.0,
                damping=2.0,
                friction=0.01,
                armature=0.01,
            ),
            "elbow_pitch": _motor_actuator(
                ".*_elbow_pitch_joint",
                effort=27.0,
                velocity=25.0,
                stiffness=30.0,
                damping=1.5,
                friction=0.01,
                armature=0.01,
            ),
            "elbow_yaw": _motor_actuator(
                ".*_elbow_yaw_joint",
                effort=27.0,
                velocity=25.0,
                stiffness=20.0,
                damping=1.0,
                friction=0.01,
                armature=0.01,
            ),
        },
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                primary_contact_match_mode="subtree",
                primary_contact_match_pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                primary_contact_match_entity="robot",
                secondary_contact_match_mode="body",
                secondary_contact_match_pattern="terrain",
                track_air_time=True,
                history_length=4,
                reduce="netforce",
            ),
            ContactSensorCfg(
                name="self_collision",
                primary_contact_match_mode="subtree",
                primary_contact_match_pattern="base_link",
                primary_contact_match_entity="robot",
                secondary_contact_match_mode="subtree",
                secondary_contact_match_pattern="base_link",
                secondary_contact_match_entity="robot",
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        ],
        mjlab_collisions=[
            MjlabCollisionCfg(
                geom_names_expr=(".*_collision",),
                disable_other_geoms=False,
            ),
        ],
        joint_names_simulation=ATOM01_JOINT_NAMES,
        body_names_simulation=ATOM01_BODY_NAMES,
        joint_symmetry_mapping=symmetry_utils.mirrored(
            {
                "left_thigh_yaw_joint": (-1, "right_thigh_yaw_joint"),
                "left_thigh_roll_joint": (-1, "right_thigh_roll_joint"),
                "left_thigh_pitch_joint": (1, "right_thigh_pitch_joint"),
                "left_knee_joint": (1, "right_knee_joint"),
                "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
                "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
                "torso_joint": (-1, "torso_joint"),
                "left_arm_pitch_joint": (1, "right_arm_pitch_joint"),
                "left_arm_roll_joint": (-1, "right_arm_roll_joint"),
                "left_arm_yaw_joint": (-1, "right_arm_yaw_joint"),
                "left_elbow_pitch_joint": (1, "right_elbow_pitch_joint"),
                "left_elbow_yaw_joint": (-1, "right_elbow_yaw_joint"),
            }
        ),
        spatial_symmetry_mapping=symmetry_utils.mirrored(
            {
                "base_link": "base_link",
                "left_thigh_yaw_link": "right_thigh_yaw_link",
                "left_thigh_roll_link": "right_thigh_roll_link",
                "left_thigh_pitch_link": "right_thigh_pitch_link",
                "left_knee_link": "right_knee_link",
                "left_ankle_pitch_link": "right_ankle_pitch_link",
                "left_ankle_roll_link": "right_ankle_roll_link",
                "left_toe_link": "right_toe_link",
                "torso_link": "torso_link",
                "left_arm_pitch_link": "right_arm_pitch_link",
                "left_arm_roll_link": "right_arm_roll_link",
                "left_arm_yaw_link": "right_arm_yaw_link",
                "left_elbow_pitch_link": "right_elbow_pitch_link",
                "left_elbow_yaw_link": "right_elbow_yaw_link",
            }
        ),
    )


ATOM01_CAPSULE_CFG = _build_atom01_cfg(ATOM01_CAPSULE_MJCF_REF)
ATOM01_MESH_CFG = _build_atom01_cfg(ATOM01_MESH_MJCF_REF)

registry.register("asset", "atom01-capsule", ATOM01_CAPSULE_CFG)
registry.register("asset", "atom01-mesh", ATOM01_MESH_CFG)
