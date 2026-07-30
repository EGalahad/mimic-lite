"""MJLab-only Bumi robot configuration for MimicLite tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.registry import Registry
from active_adaptation.utils.symmetry import mirrored


registry = Registry.instance()

BUMI_XML = ROBOT_MODEL_DIR / "bumi" / "bumi.xml"
BUMI_DELAY_MIN_LAG = 0
BUMI_DELAY_MAX_LAG = 4
BUMI_FOOT_COLLISION_PATTERN = r"^(l|r)_foot[1-7]_collision$"

BUMI_POLICY_JOINT_NAMES = [
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
]

BUMI_MOTION_JOINT_NAMES = [
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
]

BUMI_BODY_NAMES = [
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
]
BUMI_MOTION_BODY_NAMES = ["motion_root", *BUMI_BODY_NAMES]


@dataclass(frozen=True)
class BumiActuatorSpec:
    name: str
    target_names_expr: tuple[str, ...]
    stiffness: float
    damping: float
    effort_limit: float
    velocity_limit: float
    armature: float
    frictionloss: float = 0.2


BUMI_ACTUATOR_SPECS = (
    BumiActuatorSpec(
        "leg_pitch", (".*_leg_pitch_joint",), 60.0, 3.0, 60.0, 12.0, 0.007642512
    ),
    BumiActuatorSpec("waist", ("waist_yaw_joint",), 53.0, 3.4, 27.0, 9.0, 0.0045024),
    BumiActuatorSpec(
        "leg_roll", (".*_leg_roll_joint",), 60.0, 3.0, 27.0, 12.0, 0.007642512
    ),
    BumiActuatorSpec("arm_pitch", (".*_arm_pitch_joint",), 12.0, 0.4, 5.5, 50.0, 0.001),
    BumiActuatorSpec(
        "leg_yaw", (".*_leg_yaw_joint",), 60.0, 2.5, 27.0, 9.0, 0.001860625
    ),
    BumiActuatorSpec("arm_roll", (".*_arm_roll_joint",), 12.0, 0.4, 5.5, 50.0, 0.001),
    BumiActuatorSpec(
        "knee_pitch", (".*_knee_pitch_joint",), 45.0, 2.0, 60.0, 12.0, 0.007642512
    ),
    BumiActuatorSpec("arm_yaw", (".*_arm_yaw_joint",), 12.0, 0.4, 5.5, 50.0, 0.001),
    BumiActuatorSpec(
        "ankle_pitch", (".*_ankle_pitch_joint",), 10.0, 0.5, 13.0, 12.0, 0.01
    ),
    BumiActuatorSpec(
        "elbow_pitch", (".*_elbow_pitch_joint",), 12.0, 0.4, 5.5, 50.0, 0.001
    ),
    BumiActuatorSpec(
        "ankle_roll", (".*_ankle_roll_joint",), 10.0, 0.5, 13.0, 12.0, 0.01
    ),
)

BUMI_ACTION_SCALE = {
    expr: 0.25 * spec.effort_limit / spec.stiffness
    for spec in BUMI_ACTUATOR_SPECS
    for expr in spec.target_names_expr
}

BUMI_INIT_JOINT_POS = {
    ".*_leg_pitch_joint": -0.1495,
    ".*_knee_pitch_joint": 0.3215,
    ".*_ankle_pitch_joint": -0.172,
    "l_arm_roll_joint": 0.3,
    "r_arm_roll_joint": -0.3,
}

BUMI_JOINT_SYMMETRY = mirrored(
    {
        "waist_yaw_joint": (-1, "waist_yaw_joint"),
        "l_arm_pitch_joint": (1, "r_arm_pitch_joint"),
        "l_arm_roll_joint": (-1, "r_arm_roll_joint"),
        "l_arm_yaw_joint": (-1, "r_arm_yaw_joint"),
        "l_elbow_pitch_joint": (1, "r_elbow_pitch_joint"),
        "l_leg_pitch_joint": (1, "r_leg_pitch_joint"),
        "l_leg_roll_joint": (-1, "r_leg_roll_joint"),
        "l_leg_yaw_joint": (-1, "r_leg_yaw_joint"),
        "l_knee_pitch_joint": (1, "r_knee_pitch_joint"),
        "l_ankle_pitch_joint": (1, "r_ankle_pitch_joint"),
        "l_ankle_roll_joint": (-1, "r_ankle_roll_joint"),
    }
)

BUMI_SPATIAL_SYMMETRY = mirrored(
    {
        "base_link": "base_link",
        "waist_yaw_link": "waist_yaw_link",
        "l_arm_pitch_link": "r_arm_pitch_link",
        "l_arm_roll_link": "r_arm_roll_link",
        "l_arm_yaw_link": "r_arm_yaw_link",
        "l_elbow_pitch_link": "r_elbow_pitch_link",
        "l_leg_pitch_link": "r_leg_pitch_link",
        "l_leg_roll_link": "r_leg_roll_link",
        "l_leg_yaw_link": "r_leg_yaw_link",
        "l_knee_pitch_link": "r_knee_pitch_link",
        "l_ankle_pitch_link": "r_ankle_pitch_link",
        "l_ankle_roll_link": "r_ankle_roll_link",
    }
)


def get_bumi_spec():
    """Load a fresh Bumi spec while preserving its model-file asset directory."""
    import mujoco

    if not BUMI_XML.is_file():
        raise FileNotFoundError(
            f"Bumi MJCF not found at {BUMI_XML}. Prepare it with "
            "`python projects/mimic-lite/scripts/prepare_bumi_assets.py "
            "--source /path/to/bumi/xmls`."
        )
    assets_dir = BUMI_XML.parent / "assets"
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Bumi mesh directory not found: {assets_dir}")

    return getattr(mujoco, "MjSpec").from_file(str(BUMI_XML))


def make_mjlab_cfg():
    """Build a fresh MJLab entity and contact sensors for Bumi."""
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=spec.target_names_expr,
            stiffness=spec.stiffness,
            damping=spec.damping,
            effort_limit=spec.effort_limit,
            armature=spec.armature,
            frictionloss=spec.frictionloss,
            delay_min_lag=BUMI_DELAY_MIN_LAG,
            delay_max_lag=BUMI_DELAY_MAX_LAG,
        )
        for spec in BUMI_ACTUATOR_SPECS
    )

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.472),
            joint_pos=BUMI_INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_bumi_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                condim={BUMI_FOOT_COLLISION_PATTERN: 3, ".*_collision": 1},
                priority={BUMI_FOOT_COLLISION_PATTERN: 1},
                friction={BUMI_FOOT_COLLISION_PATTERN: (0.6,)},
                disable_other_geoms=False,
            ),
        ),
        joint_names_simulation=BUMI_POLICY_JOINT_NAMES,
        body_names_simulation=BUMI_BODY_NAMES,
        joint_symmetry_mapping=BUMI_JOINT_SYMMETRY,
        spatial_symmetry_mapping=BUMI_SPATIAL_SYMMETRY,
    )
    sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=4,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    return cfg, sensors


def make_cfg(backend: Literal["isaaclab", "mjlab"]):
    if backend == "mjlab":
        return make_mjlab_cfg()
    if backend == "isaaclab":
        raise NotImplementedError(
            "Bumi tracking is MJLab-only because no validated USD/URDF asset is available."
        )
    raise ValueError(f"Unsupported Bumi backend: {backend}")


registry.register("asset", "bumi", make_cfg)


__all__ = [
    "BUMI_ACTION_SCALE",
    "BUMI_ACTUATOR_SPECS",
    "BUMI_BODY_NAMES",
    "BUMI_DELAY_MAX_LAG",
    "BUMI_DELAY_MIN_LAG",
    "BUMI_MOTION_BODY_NAMES",
    "BUMI_MOTION_JOINT_NAMES",
    "BUMI_POLICY_JOINT_NAMES",
    "BUMI_XML",
    "get_bumi_spec",
    "make_cfg",
    "make_mjlab_cfg",
]
