from __future__ import annotations

from typing import Literal
import xml.etree.ElementTree as ET
from mjhub import resolve_asset_reference

from active_adaptation.registry import Registry

registry = Registry.instance()

H2_MJCF_REF = "hf://elijahgalahad/h2_model@beb532e8717b99816b93baace0c649a599538715/h2.xml"
H2_XML = resolve_asset_reference(H2_MJCF_REF)


def _parse_names() -> tuple[list[str], list[str]]:
    tree = ET.parse(H2_XML)
    root = tree.getroot()
    body_names: list[str] = []
    joint_names: list[str] = []
    for elem in root.iter():
        if elem.tag == "body":
            name = elem.attrib.get("name")
            if name and name not in body_names:
                body_names.append(name)
        elif elem.tag == "joint":
            name = elem.attrib.get("name")
            if name and name not in joint_names:
                joint_names.append(name)
    return body_names, joint_names


BODY_NAMES_SIMULATION, JOINT_NAMES_SIMULATION = _parse_names()

INIT_POS = (0.0, 0.0, 1.2)
INIT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    ".*": 0.0,
}

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)


def _reflected_inertia(rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
    return (
        rotor_inertias[0] * (gears[1] * gears[2]) ** 2
        + rotor_inertias[1] * gears[2] ** 2
        + rotor_inertias[2]
    )


ARMATURE_5020 = _reflected_inertia(ROTOR_INERTIAS_5020, GEARS_5020)
ARMATURE_7520_14 = _reflected_inertia(ROTOR_INERTIAS_7520_14, GEARS_7520_14)
ARMATURE_7520_22 = _reflected_inertia(ROTOR_INERTIAS_7520_22, GEARS_7520_22)
ARMATURE_4010 = _reflected_inertia(ROTOR_INERTIAS_4010, GEARS_4010)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _motor_actuator(joint_names_expr: tuple[str, ...], effort: float, armature: float):
    from mjlab.actuator import BuiltinPositionActuatorCfg

    return BuiltinPositionActuatorCfg(
        target_names_expr=joint_names_expr,
        effort_limit=effort,
        stiffness=_stiffness(armature),
        damping=_damping(armature),
        armature=armature,
        frictionloss=0.01,
    )


def make_mjlab_cfg():
    import mujoco
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    from active_adaptation.assets.asset_cfg import AssetSpec, EntityCfg

    def spec_fn():
        return mujoco.MjSpec.from_file(str(H2_XML))

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                _motor_actuator((".*_hip_roll_joint",), 360.0, ARMATURE_7520_22),
                _motor_actuator((".*_hip_pitch_joint",), 360.0, ARMATURE_7520_22),
                _motor_actuator((".*_hip_yaw_joint",), 360.0, ARMATURE_7520_14),
                _motor_actuator((".*_knee_joint",), 360.0, ARMATURE_7520_22),
                _motor_actuator((".*_ankle_roll_joint",), 19.0, ARMATURE_5020),
                _motor_actuator((".*_ankle_pitch_joint",), 66.88, 2.0 * ARMATURE_5020),
                _motor_actuator(("waist_yaw_joint",), 120.0, ARMATURE_7520_14),
                _motor_actuator(("waist_roll_joint", "waist_pitch_joint"), 180.0, 2.0 * ARMATURE_5020),
                _motor_actuator((".*_shoulder_pitch_joint",), 120.0, ARMATURE_5020),
                _motor_actuator((".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"), 54.0, ARMATURE_5020),
                _motor_actuator((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 25.0, ARMATURE_4010),
                _motor_actuator(("head_pitch_joint", "head_yaw_joint"), 50.0, ARMATURE_5020),
            ),
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=1,
                conaffinity=1,
                condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*": 1},
                priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
                friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
                disable_other_geoms=False,
            ),
        ),
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = (
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
            history_length=3,
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


def make_cfg(backend: Literal["mjlab"] | str):
    if backend != "mjlab":
        raise ValueError("H2 asset is currently implemented for mjlab only")
    return make_mjlab_cfg()


registry.register("asset", "mlite-h2", make_cfg)
