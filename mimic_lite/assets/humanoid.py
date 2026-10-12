"""Shared biped construction; robot dynamics and landmarks stay in each asset."""

from dataclasses import dataclass
from functools import cache

from mjhub import resolve_asset_reference


@dataclass(frozen=True)
class Motor:
    joints: tuple[str, ...]
    effort: float
    velocity: float
    stiffness: float
    damping: float
    armature: float = 0.0
    friction: float = 0.01


@cache
def load_model(reference):
    """Cache immutable model metadata only after this robot is requested."""
    import mujoco

    xml = resolve_asset_reference(reference)
    return xml, mujoco.MjModel.from_xml_path(str(xml))


def make_biped(
    backend,
    *,
    mjcf,
    urdf,
    pos,
    joint_pos,
    motors,
    root,
    feet,
    collisions=None,
    joint_names=None,
):
    if backend not in ("mjlab", "isaaclab"):
        raise ValueError(f"Biped asset does not support backend {backend!r}")
    import mujoco
    from active_adaptation.assets.asset_cfg import AssetSpec

    # Resolve only the robot requested by the caller, preserving HF mesh symlinks.
    xml, model = load_model(mjcf)
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    joints = joint_names or [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    if backend == "mjlab":
        from active_adaptation.assets.asset_cfg import EntityCfg
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.utils.spec_config import CollisionCfg

        cfg = EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_file(str(xml)),
            init_state=EntityCfg.InitialStateCfg(
                pos=pos, joint_pos=joint_pos, joint_vel={".*": 0.0}
            ),
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(
                    BuiltinPositionActuatorCfg(
                        target_names_expr=m.joints,
                        effort_limit=m.effort,
                        stiffness=m.stiffness,
                        damping=m.damping,
                        armature=m.armature,
                        frictionloss=m.friction,
                    )
                    for m in motors
                )
            ),
            collisions=(CollisionCfg(**collisions),) if collisions else (),
            joint_names_simulation=list(joints),
            body_names_simulation=bodies,
        )
        sensors = (
            ContactSensorCfg(
                name="contact_forces",
                primary=ContactMatch(mode="subtree", pattern=feet, entity="robot"),
                secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
                history_length=4,
            ),
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(mode="subtree", pattern=root, entity="robot"),
                secondary=ContactMatch(mode="subtree", pattern=root, entity="robot"),
                fields=("found", "force"),
                reduce="none",
                num_slots=1,
                history_length=4,
            ),
        )
    else:
        import isaaclab.sim as sim_utils
        from active_adaptation.assets.asset_cfg import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg

        cfg = ArticulationCfg(
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(resolve_asset_reference(urdf)),
                fix_base=False,
                replace_cylinders_with_capsules=True,
                self_collision=True,
                make_instanceable=False,
                force_usd_conversion=True,
                merge_fixed_joints=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0, damping=0
                    )
                ),
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos, joint_pos=joint_pos, joint_vel={".*": 0.0}
            ),
            actuators={
                f"motor_{i}": ImplicitActuatorCfg(
                    joint_names_expr=list(m.joints),
                    effort_limit_sim=m.effort,
                    velocity_limit_sim=m.velocity,
                    stiffness=m.stiffness,
                    damping=m.damping,
                    armature=m.armature,
                    friction=m.friction,
                )
                for i, m in enumerate(motors)
            },
            joint_names_simulation=list(joints),
            body_names_simulation=bodies,
        )
        sensors = {
            "contact_forces": ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                track_air_time=True,
                history_length=4,
            ),
            "self_collision": ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/^(?!.*ankle_roll_[Ll]ink$)(?!.*toe_link$)(?!.*wrist.*$).+$",
                history_length=4,
            ),
        }
    return AssetSpec(config=cfg, sensors=sensors)
