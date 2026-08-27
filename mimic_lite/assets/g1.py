from dataclasses import dataclass

from active_adaptation.assets.humanoids.g1 import (
    BODY_NAMES_SIMULATION,
    JOINT_NAMES_SIMULATION,
    JOINT_SYMMETRY_MAPPING,
    SPATIAL_SYMMETRY_MAPPING,
)
from active_adaptation.registry import Registry

from mjhub import resolve_asset_reference

registry = Registry.instance()


G1_XMLS_REPO_ID = "elijahgalahad/g1_xmls"
G1_XMLS_REVISION = "main"

G1_MJCF_REF_BY_MODE = {
    5: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_5_11.xml",
    11: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_5_11.xml",
    13: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_13_15.xml",
    15: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_13_15.xml",
}

G1_URDF_REF_BY_MODE = {
    5: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_5_11.urdf",
    11: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_5_11.urdf",
    13: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_13_15.urdf",
    15: f"hf://{G1_XMLS_REPO_ID}@{G1_XMLS_REVISION}/g1-mode_13_15.urdf",
}

TOE_BODY_BY_ANKLE_BODY = {
    "left_ankle_roll_link": "left_toe_link",
    "right_ankle_roll_link": "right_toe_link",
}


def _with_toe_body_names(body_names):
    result = []
    for body_name in body_names:
        result.append(body_name)
        toe_body_name = TOE_BODY_BY_ANKLE_BODY.get(body_name)
        if toe_body_name is not None:
            result.append(toe_body_name)
    return result


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected inertia of a two-stage planetary gearbox.

    Formula matches mjlab.utils.actuator.reflected_inertia_from_two_stage_planetary.
    """
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


# Motor specs (from Unitree/mjlab constants + 5010 spec from mode table).
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)

ROTOR_INERTIAS_5010 = (0.084e-4, 0.015e-4, 0.068e-4)
GEARS_5010 = (1, 4, 4)

ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_14, GEARS_7520_14)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_22, GEARS_7520_22)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)
ARMATURE_5010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5010, GEARS_5010)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


@dataclass(frozen=True)
class _MotorSpec:
    joint_names_expr: str
    effort_limit: float
    velocity_limit: float
    stiffness: float
    damping: float
    friction: float
    armature: float


def _motor_actuator(joint_names_expr: str, effort: float, velocity: float, armature: float) -> _MotorSpec:
    return _MotorSpec(
        joint_names_expr=joint_names_expr,
        effort_limit=float(effort),
        velocity_limit=float(velocity),
        stiffness=_stiffness(armature),
        damping=_damping(armature),
        friction=0.01,
        armature=armature,
    )


# Shared actuator groups.
G1_ACTUATOR_5020_UPPER = _motor_actuator(
    joint_names_expr=(
        ".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|"
        ".*_shoulder_yaw_joint|.*_wrist_roll_joint"
    ),
    effort=25.0,
    velocity=37.0,
    armature=ARMATURE_5020,
)

G1_ACTUATOR_HIP_PITCH_7520_14 = _motor_actuator(
    joint_names_expr=".*_hip_pitch_joint",
    effort=88.0,
    velocity=32.0,
    armature=ARMATURE_7520_14,
)

G1_ACTUATOR_HIP_PITCH_7520_22 = _motor_actuator(
    joint_names_expr=".*_hip_pitch_joint",
    effort=139.0,
    velocity=20.0,
    armature=ARMATURE_7520_22,
)

G1_ACTUATOR_HIP_YAW_7520_14 = _motor_actuator(
    joint_names_expr=".*_hip_yaw_joint|waist_yaw_joint",
    effort=88.0,
    velocity=32.0,
    armature=ARMATURE_7520_14,
)

G1_ACTUATOR_HIP_ROLL_KNEE_7520_22 = _motor_actuator(
    joint_names_expr=".*_hip_roll_joint|.*_knee_joint",
    effort=139.0,
    velocity=20.0,
    armature=ARMATURE_7520_22,
)

G1_ACTUATOR_WRIST_4010 = _motor_actuator(
    joint_names_expr=".*_wrist_pitch_joint|.*_wrist_yaw_joint",
    effort=5.0,
    velocity=22.0,
    armature=ARMATURE_4010,
)

G1_ACTUATOR_WRIST_5010 = _motor_actuator(
    joint_names_expr=".*_wrist_pitch_joint|.*_wrist_yaw_joint",
    effort=13.4,
    velocity=27.0,
    armature=ARMATURE_5010,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2x5020 equivalent at joint level.
G1_ACTUATOR_WAIST = _motor_actuator(
    joint_names_expr="waist_pitch_joint|waist_roll_joint",
    effort=50.0,
    velocity=37.0,
    armature=2.0 * ARMATURE_5020,
)

G1_ACTUATOR_ANKLE = _motor_actuator(
    joint_names_expr=".*_ankle_pitch_joint|.*_ankle_roll_joint",
    effort=50.0,
    velocity=37.0,
    armature=2.0 * ARMATURE_5020,
)


INIT_POS = (0.0, 0.0, 0.76)
INIT_JOINT_POS = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
}


def _build_g1_cfg(mode: int, backend: str):
    if mode not in (5, 11, 13, 15):
        raise ValueError(f"Unsupported mode: {mode}")

    hip_pitch_act = G1_ACTUATOR_HIP_PITCH_7520_14 if mode in (5, 13) else G1_ACTUATOR_HIP_PITCH_7520_22
    wrist_act = G1_ACTUATOR_WRIST_4010 if mode in (5, 11) else G1_ACTUATOR_WRIST_5010

    actuators = {
            "g1_5020_upper": G1_ACTUATOR_5020_UPPER,
            "g1_hip_pitch": hip_pitch_act,
            "g1_hip_yaw_waist_yaw": G1_ACTUATOR_HIP_YAW_7520_14,
            "g1_hip_roll_knee": G1_ACTUATOR_HIP_ROLL_KNEE_7520_22,
            "g1_wrist_pitch_yaw": wrist_act,
            "g1_waist": G1_ACTUATOR_WAIST,
            "g1_ankle": G1_ACTUATOR_ANKLE,
    }
    body_names = _with_toe_body_names(list(BODY_NAMES_SIMULATION))

    if backend == "mjlab":
        import mujoco
        from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.utils.spec_config import CollisionCfg

        mjcf_path = resolve_asset_reference(G1_MJCF_REF_BY_MODE[mode])
        cfg = EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=INIT_POS, joint_pos=INIT_JOINT_POS, joint_vel={".*": 0.0}
            ),
            spec_fn=lambda: mujoco.MjSpec.from_file(str(mjcf_path)),
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(
                    BuiltinPositionActuatorCfg(
                        target_names_expr=(spec.joint_names_expr,),
                        effort_limit=spec.effort_limit,
                        stiffness=spec.stiffness,
                        damping=spec.damping,
                        armature=spec.armature,
                        frictionloss=spec.friction,
                    )
                    for spec in actuators.values()
                )
            ),
            collisions=(
                CollisionCfg(
                    geom_names_expr=(".*_collision",),
                    contype=1,
                    conaffinity=1,
                    disable_other_geoms=False,
                ),
            ),
            joint_names_simulation=JOINT_NAMES_SIMULATION,
            body_names_simulation=body_names,
            joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
            spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        )
        sensors = (
            ContactSensorCfg(
                name="contact_forces",
                primary=ContactMatch(
                    mode="subtree",
                    pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                    entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
                history_length=4,
            ),
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        )
        return AssetSpec(config=cfg, sensors=sensors)

    if backend == "isaaclab":
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import AssetSpec, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg

        actuator_cfgs = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[spec.joint_names_expr],
                effort_limit_sim=spec.effort_limit,
                velocity_limit_sim=spec.velocity_limit,
                stiffness=spec.stiffness,
                damping=spec.damping,
                friction=spec.friction,
                armature=spec.armature,
            )
            for name, spec in actuators.items()
        }
        cfg = ArticulationCfg(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(resolve_asset_reference(G1_URDF_REF_BY_MODE[mode])),
                fix_base=False,
                replace_cylinders_with_capsules=True,
                self_collision=True,
                make_instanceable=False,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0, damping=0
                    )
                ),
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=INIT_POS, joint_pos=INIT_JOINT_POS, joint_vel={".*": 0.0}
            ),
            actuators=actuator_cfgs,
            joint_names_simulation=JOINT_NAMES_SIMULATION,
            body_names_simulation=body_names,
            joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
            spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        )
        sensors = {
            "contact_forces": ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*", track_air_time=True, history_length=4
            ),
            "self_collision": ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/^(?!.*_ankle_roll_link.*$)(?!.*_toe_link$)(?!.*_wrist_yaw_link$).+$",
                track_air_time=True,
                history_length=4,
            ),
        }
        return AssetSpec(config=cfg, sensors=sensors)

    raise ValueError(f"Unsupported backend: {backend}")


for _mode in (5, 11, 13, 15):
    registry.register(
        "asset",
        f"g1-mode_{_mode}",
        lambda backend, mode=_mode: _build_g1_cfg(mode, backend),
    )
