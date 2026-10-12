from __future__ import annotations

from active_adaptation.registry import Registry

registry = Registry.instance()

H2_MJCF_REF = (
    "hf://elijahgalahad/h2_model@97909bf17e6d6df6a82df229a92a9e7eb89acd8d/h2.xml"
)
H2_URDF_REF = H2_MJCF_REF.rsplit("/", 1)[0] + "/h2.urdf"

INIT_POS = (0.0, 0.0, 1.2)
INIT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
}

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)


def _reflected_inertia(
    rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]
) -> float:
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


def make_cfg(backend):
    from .humanoid import Motor, make_biped

    groups = (
        ((".*_hip_roll_joint",), 360.0, ARMATURE_7520_22),
        ((".*_hip_pitch_joint",), 360.0, ARMATURE_7520_22),
        ((".*_hip_yaw_joint",), 360.0, ARMATURE_7520_14),
        ((".*_knee_joint",), 360.0, ARMATURE_7520_22),
        ((".*_ankle_roll_joint",), 19.0, ARMATURE_5020),
        ((".*_ankle_pitch_joint",), 66.88, 2.0 * ARMATURE_5020),
        (("waist_yaw_joint",), 120.0, ARMATURE_7520_14),
        (("waist_roll_joint", "waist_pitch_joint"), 180.0, 2.0 * ARMATURE_5020),
        ((".*_shoulder_pitch_joint",), 120.0, ARMATURE_5020),
        (
            (
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            54.0,
            ARMATURE_5020,
        ),
        ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 25.0, ARMATURE_4010),
        (("head_pitch_joint", "head_yaw_joint"), 50.0, ARMATURE_5020),
    )
    return make_biped(
        backend,
        mjcf=H2_MJCF_REF,
        urdf=H2_URDF_REF,
        pos=INIT_POS,
        joint_pos=INIT_JOINT_POS,
        motors=[
            Motor(
                names,
                effort,
                1000.0,
                _stiffness(armature),
                _damping(armature),
                armature,
            )
            for names, effort, armature in groups
        ],
        root="pelvis",
        feet=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        collisions={
            "geom_names_expr": (".*_collision",),
            "contype": 1,
            "conaffinity": 1,
            "condim": {r"^(left|right)_foot[1-7]_collision$": 3, ".*": 1},
            "priority": {r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
            "friction": {r"^(left|right)_foot[1-7]_collision$": (0.6,)},
            "disable_other_geoms": False,
        },
    )


registry.register("asset", "mlite-h2", make_cfg)
