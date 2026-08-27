from __future__ import annotations

from types import SimpleNamespace
import unittest

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from mimic_lite_learning.fast_sac import FastSAC, FastSACConfig
from mimic_lite_learning.fast_td3 import FastTD3, FastTD3Config
from mimic_lite_learning.ppo import PPOConfig, PPOPolicy
from mimic_lite_learning.ppo_roa import PPOConfig as PPOROAConfig
from mimic_lite_learning.ppo_roa import PPOROA
from mimic_lite_learning.sac import SAC, SACConfig


class _FakeActionManager:
    def __init__(self, action_dim: int, joint_names: list[str]) -> None:
        self.action_dim = action_dim
        self.joint_names = joint_names


class _FakeEnv:
    def __init__(self, num_envs: int = 4, action_dim: int = 2) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            total_iters=1,
            reward={"tracking": {"_enabled_": True}},
        )
        self.action_manager = _FakeActionManager(
            action_dim=action_dim,
            joint_names=[f"joint_{i}" for i in range(action_dim)],
        )
        self.observation_spec = Composite(
            policy=Unbounded(shape=(num_envs, 5)),
            command=Unbounded(shape=(num_envs, 3)),
            command_short=Unbounded(shape=(num_envs, 2)),
            priv=Unbounded(shape=(num_envs, 7)),
            shape=(num_envs,),
        )
        self.action_spec = Composite(
            action=Bounded(low=-1.0, high=1.0, shape=(num_envs, action_dim)),
            shape=(num_envs,),
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1))

    def fake_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.zeros(self.num_envs, 5),
                "command": torch.zeros(self.num_envs, 3),
                "priv": torch.zeros(self.num_envs, 7),
                "done": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                "terminated": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                "is_init": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                "collector": torch.zeros(self.num_envs, 1),
                "next": {
                    "policy": torch.zeros(self.num_envs, 5),
                    "command": torch.zeros(self.num_envs, 3),
                    "priv": torch.zeros(self.num_envs, 7),
                    "reward": torch.zeros(self.num_envs, 1),
                    "stats": TensorDict({}, batch_size=[self.num_envs]),
                    "done": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                    "terminated": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                    "truncated": torch.zeros(self.num_envs, 1, dtype=torch.bool),
                    "discount": torch.ones(self.num_envs, 1),
                    "reward": torch.zeros(self.num_envs, 1),
                },
            },
            batch_size=[self.num_envs],
        )


class PolicyFactoryApiTest(unittest.TestCase):
    def test_algo_configs_instantiate_to_config_objects(self) -> None:
        cases = (
            (PPOConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)), PPOPolicy),
            (
                PPOROAConfig(
                    actor_hidden_dims=(8,),
                    critic_hidden_dims=(8,),
                    encoder_teacher_dims=(8,),
                    encoder_student_dims=(8,),
                    latent_dim=4,
                ),
                PPOROA,
            ),
            (
                SACConfig(
                    actor_hidden_dims=(8, 8, 8),
                    critic_hidden_dims=(8, 8, 8),
                    use_amp=False,
                ),
                SAC,
            ),
            (
                FastSACConfig(
                    action_space_mode="manual",
                    action_bounds={".*": [-1.0, 1.0]},
                    actor_hidden_dim=8,
                    critic_hidden_dim=8,
                    train_every=1,
                    buffer_size=8,
                    replay_batch_size=4,
                    warm_up_steps=1,
                    utd_ratio=1,
                    policy_frequency=1,
                    enable_value_probe=False,
                ),
                FastSAC,
            ),
            (
                FastTD3Config(
                    action_bounds={".*": [-1.0, 1.0]},
                    actor_hidden_dim=8,
                    critic_hidden_dim=8,
                    collect_steps=1,
                    buffer_size=8,
                    replay_batch_size=4,
                    warm_up_steps=1,
                    updates_per_step=1,
                    policy_frequency=1,
                    num_atoms=5,
                    v_min=-1.0,
                    v_max=1.0,
                ),
                FastTD3,
            ),
        )

        for cfg, policy_cls in cases:
            with self.subTest(config=type(cfg).__name__):
                instantiated = hydra.utils.instantiate(OmegaConf.structured(cfg))
                self.assertIsInstance(instantiated, type(cfg))
                self.assertIs(instantiated.get_class(), policy_cls)

    def test_from_env_builds_all_supported_policies(self) -> None:
        env = _FakeEnv()

        ppo = PPOPolicy.from_env(
            PPOConfig(
                actor_hidden_dims=(8,),
                critic_hidden_dims=(8,),
                compile=False,
                compile_rollout=False,
                compile_train_modules=False,
                in_keys=("command", "policy", "priv"),
                actor_in_keys=("policy", "command"),
            ),
            env,
            device="cpu",
        )
        self.assertIsInstance(ppo, PPOPolicy)

        ppo_roa = PPOROA.from_env(
            PPOROAConfig(
                actor_hidden_dims=(8,),
                critic_hidden_dims=(8,),
                encoder_teacher_dims=(8,),
                encoder_student_dims=(8,),
                latent_dim=4,
                in_keys=("command", "command_short", "policy", "priv"),
            ),
            env,
            device="cpu",
        )
        self.assertIsInstance(ppo_roa, PPOROA)

        sac = SAC.from_env(
            SACConfig(
                actor_hidden_dims=(8, 8, 8),
                critic_hidden_dims=(8, 8, 8),
                buffer_size=8,
                critic_batch_size=4,
                actor_batch_size=4,
                warm_up_steps=1,
                use_amp=False,
                distributional=False,
                use_correlated=False,
            ),
            env,
            device="cpu",
        )
        self.assertIsInstance(sac, SAC)

        fast_sac = FastSAC.from_env(
            FastSACConfig(
                action_space_mode="manual",
                action_bounds={".*": [-1.0, 1.0]},
                actor_hidden_dim=8,
                critic_hidden_dim=8,
                train_every=1,
                buffer_size=8,
                replay_batch_size=4,
                warm_up_steps=1,
                utd_ratio=1,
                policy_frequency=1,
                enable_value_probe=False,
            ),
            env,
            device="cpu",
        )
        self.assertIsInstance(fast_sac, FastSAC)

        fast_td3 = FastTD3.from_env(
            FastTD3Config(
                action_bounds={".*": [-1.0, 1.0]},
                actor_hidden_dim=8,
                critic_hidden_dim=8,
                collect_steps=1,
                buffer_size=8,
                replay_batch_size=4,
                warm_up_steps=1,
                updates_per_step=1,
                policy_frequency=1,
                num_atoms=5,
                v_min=-1.0,
                v_max=1.0,
            ),
            env,
            device="cpu",
        )
        self.assertIsInstance(fast_td3, FastTD3)


if __name__ == "__main__":
    unittest.main()
