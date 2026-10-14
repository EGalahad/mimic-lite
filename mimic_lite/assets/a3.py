"""AgiBot A3 parameters from Renforce-Dynamics/mimicA3."""

from active_adaptation.registry import Registry

from .humanoid import Motor, load_model, make_biped

# Official A3 T3.0 hands and inertia.
A3_MJCF_REF = "hf://elijahgalahad/a3_model@6657ece91a831e92038f74e50abc8ccb32b9790f/mjlab/a3_31dof_no_racket.xml"
A3_URDF_REF = "hf://elijahgalahad/a3_model@6657ece91a831e92038f74e50abc8ccb32b9790f/a3_no_racket.urdf"

INIT_JOINT_POS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.8,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "left_hip_yaw_joint": -0.0348,
    "left_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_hip_pitch_joint": -0.1311,
    "right_hip_roll_joint": -0.0056,
    "right_hip_yaw_joint": 0.0348,
    "right_knee_joint": 0.2468,
    "right_ankle_pitch_joint": -0.1204,
    "right_ankle_roll_joint": 0.0078,
}

MOTOR_GROUPS = [
    (("waist_yaw_joint",), 85.0, 3.0, 220.0),
    (("waist_roll_joint",), 50.0, 2.0, 46.0),
    (("waist_pitch_joint",), 50.0, 2.0, 118.0),
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 80.0, 3.0, 220.0),
    ((".*_hip_roll_joint",), 120.0, 4.0, 220.0),
    ((".*_knee_joint",), 250.0, 8.0, 320.0),
    ((".*_ankle_pitch_joint",), 50.0, 2.0, 118.2),
    ((".*_ankle_roll_joint",), 50.0, 2.0, 54.75),
    ((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"), 40.0, 3.0, 60.0),
    (
        (".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"),
        30.0,
        2.0,
        24.0,
    ),
    ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 20.0, 2.0, 6.0),
    (("head_yaw_joint", "head_pitch_joint"), 40.0, 2.0, 6.0),
]


def make_cfg(backend):
    mjcf = A3_MJCF_REF
    urdf = A3_URDF_REF
    import re

    _, model = load_model(mjcf)
    motors = []
    for names, stiffness, damping, effort in MOTOR_GROUPS:
        for i in range(model.njnt):
            name = model.joint(i).name
            if any(re.fullmatch(pattern, name) for pattern in names):
                dof = model.jnt_dofadr[i]
                motors.append(
                    Motor(
                        (name,),
                        effort,
                        1000.0,
                        stiffness,
                        damping,
                        float(model.dof_armature[dof]),
                        float(model.dof_frictionloss[dof]),
                    )
                )
    return make_biped(
        backend,
        mjcf=mjcf,
        urdf=urdf,
        pos=(0.0, 0.0, 1.06839),
        joint_pos=INIT_JOINT_POS,
        motors=motors,
        root="pelvis_link",
        feet=r"^(left_ankle_roll_Link|right_ankle_roll_Link)$",
    )


Registry.instance().register("asset", "mlite-a3", make_cfg)
