"""Run directly to validate the root-stage patches and body experiment matrix."""
from pathlib import Path

from hydra import compose, initialize_config_dir


def main():
    config = Path(__file__).resolve().parents[1] / "cfg"
    arms = {
        "control": (1.0, 0.3, 0.4),
        "weight2": (2.0, 0.3, 0.4),
        "tighter_sigma": (1.0, 0.15, 0.3),
        "wrist_roll": (1.0, 0.3, 0.4),
    }
    with initialize_config_dir(config_dir=str(config), version_base=None):
        for phase in ("train", "adapt", "finetune"):
            patch = "root_global_xyz" if phase == "train" else "root_window_xyz"
            for arm, (weight, pos_sigma, ori_sigma) in arms.items():
                patches = ["teacher_future_t16", patch]
                if arm == "wrist_roll":
                    patches.append("palm_wrist_roll")
                task = compose(overrides=[
                    "+task=tracking-base",
                    f"+task/patches=[{','.join(patches)}]",
                    f"task.reward.tracking.body_pos.weight={weight}",
                    f"task.reward.tracking.body_ori.weight={weight}",
                    f"task.reward.tracking.body_pos.sigma={pos_sigma}",
                    f"task.reward.tracking.body_ori.sigma={ori_sigma}",
                ]).task
                root = task.reward.tracking.root_pos
                assert root.body_names == ["torso_link"]
                assert root.weight == 0.5 and root.sigma == 0.3
                assert task.termination.root_pos_error.is_timeout is False
                assert task.termination.root_pos_error.threshold == 0.4
                assert task.termination.root_pos_error.min_steps == 25
                if phase == "train":
                    assert root._target_ == "mimic_lite.body_pos_exp"
                    assert root.history_steps is None
                else:
                    assert root._target_ == "mimic_lite.windowed_root_displacement_exp"
                    assert root.history_steps == [200]
                assert task.reward.tracking.body_pos.weight == weight
                assert task.reward.tracking.body_ori.weight == weight
                assert task.reward.tracking.body_pos.sigma == pos_sigma
                assert task.reward.tracking.body_ori.sigma == ori_sigma
                assert task.reward.tracking.body_linvel.weight == 0.5
                assert task.reward.tracking.body_angvel.weight == 0.5
                assert ".*_palm_link" in task.shared.reward_body_names
                assert ".*_wrist_yaw_link" in task.shared.obs_body_names
                assert (".*_wrist_roll_link" in task.shared.reward_body_names) == (arm == "wrist_roll")
                assert ".*_wrist_roll_link" not in task.shared.tracking_body_names
                assert ".*_wrist_roll_link" not in task.shared.obs_body_names
    print("Twelve phase/arm combinations passed; root target, timeout and body deltas verified")


if __name__ == "__main__":
    main()
