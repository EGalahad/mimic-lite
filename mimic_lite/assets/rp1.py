"""RP1 standard and 24-DoF assets; training gains from mimic-lite-rp1."""

from active_adaptation.registry import Registry

from .humanoid import Motor, make_biped

RP1_REPO = "hf://elijahgalahad/rp1_model@66e80fc75f358fb939a65df6ab2cc8826e9ad921"
INIT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": -0.2,
    "left_ankle_pitch_joint": -0.1,
    "left_shoulder_pitch_joint": 0.18,
    "left_shoulder_roll_joint": 0.18,
    "left_elbow_joint": -1.2,
    "right_hip_pitch_joint": 0.1,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.1,
    "right_shoulder_pitch_joint": -0.18,
    "right_shoulder_roll_joint": -0.18,
    "right_elbow_joint": 1.2,
}
MOTORS = (
    Motor((".*waist.*",), 141.7, 14.24192, 198.521, 15.798, 0.125715),
    Motor((".*_hip.*|.*_knee_joint",), 141.7, 14.24192, 148.891, 9.479, 0.125715),
    Motor(
        (".*_ankle_pitch_joint|.*_ankle_roll_joint",),
        70.6,
        21.88648,
        40.193,
        2.559,
        0.040724,
    ),
    Motor(
        (".*_shoulder_.*_joint|.*elbow.*|.*_wrist(?:_.*)?_joint",),
        35.3,
        21.88648,
        40.193,
        2.559,
        0.020362,
    ),
)


def make_cfg(backend, *, reduced=False):
    return make_biped(
        backend,
        mjcf=RP1_REPO
        + (
            "/mjcf/robot_24dof.mimic_lite.xml"
            if reduced
            else "/mjcf/robot.mimic_lite.xml"
        ),
        urdf=RP1_REPO + ("/urdf/rp1_24dof.urdf" if reduced else "/urdf/rp1.urdf"),
        pos=(0.0, 0.0, 0.78),
        joint_pos=INIT_JOINT_POS,
        motors=MOTORS,
        root="base_link",
        feet=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        collisions={
            "geom_names_expr": (".*collision",),
            "contype": 1,
            "conaffinity": 1,
            "condim": 4,
            "priority": 0,
            "disable_other_geoms": False,
        },
    )


registry = Registry.instance()
registry.register("asset", "mlite-rp1", make_cfg)
registry.register(
    "asset", "mlite-rp1-24dof", lambda backend: make_cfg(backend, reduced=True)
)
