from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from active_adaptation.learning.ppo.ppo_base import PPOBase
from mimic_lite_learning.ppo import PPOConfig, PPOPolicy


class PPOActorEncoderTest(unittest.TestCase):
    def test_roa_style_privileged_encoder_feeds_actor(self) -> None:
        policy = object.__new__(PPOPolicy)
        PPOBase.__init__(policy)
        policy.cfg = PPOConfig(
            use_actor_encoder=True,
            encoder_hidden_dims=(512, 512),
            latent_dim=256,
            encoder_in_keys=("priv",),
            actor_in_keys=("policy",),
            actor_hidden_dims=(1024, 512, 512),
            compile_train_modules=False,
        )
        policy.action_dim = 29
        policy.device = torch.device("cpu")

        actor = policy._build_actor(list(policy.cfg.actor_in_keys))
        data = TensorDict(
            {
                "policy": torch.randn(4, 518),
                "priv": torch.randn(4, 548),
            },
            batch_size=[4],
        )
        actor(data)
        policy.actor = actor

        self.assertEqual(data["_actor_encoder_feature"].shape, (4, 256))
        self.assertEqual(data["action"].shape, (4, 29))
        linear_shapes = [
            (module.in_features, module.out_features)
            for module in actor.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            linear_shapes,
            [
                (548, 512),
                (512, 512),
                (512, 256),
                (774, 1024),
                (1024, 512),
                (512, 512),
                (512, 29),
            ],
        )
        encoder_parameter_ids = {
            id(parameter) for parameter in policy._actor_encoder_parameters
        }
        actor_parameter_ids = {id(parameter) for parameter in actor.parameters()}
        body_parameter_ids = actor_parameter_ids - encoder_parameter_ids
        self.assertTrue(encoder_parameter_ids)
        self.assertTrue(body_parameter_ids)
        self.assertFalse(encoder_parameter_ids & body_parameter_ids)
        self.assertEqual(
            encoder_parameter_ids | body_parameter_ids,
            actor_parameter_ids,
        )
    def test_single_mlp_mode_uses_actor_inputs_directly(self) -> None:
        policy = object.__new__(PPOPolicy)
        PPOBase.__init__(policy)
        policy.cfg = PPOConfig(
            use_actor_encoder=False,
            actor_in_keys=("policy", "command"),
            actor_hidden_dims=(8,),
            compile_train_modules=False,
        )
        policy.action_dim = 2
        policy.device = torch.device("cpu")
        actor = policy._build_actor(list(policy.cfg.actor_in_keys))
        data = TensorDict(
            {"policy": torch.randn(2, 3), "command": torch.randn(2, 5)},
            batch_size=[2],
        )
        actor(data)

        self.assertNotIn("_actor_encoder_feature", data)
        self.assertEqual(data["_actor_input"].shape, (2, 8))
        self.assertEqual(data["action"].shape, (2, 2))

    def test_roa_student_encoder_uses_policy_and_command(self) -> None:
        policy = object.__new__(PPOPolicy)
        PPOBase.__init__(policy)
        policy.cfg = PPOConfig(
            in_keys=("command", "policy", "priv"),
            use_actor_encoder=True,
            encoder_hidden_dims=(8, 8),
            latent_dim=4,
            encoder_in_keys=("policy", "command"),
            actor_in_keys=("policy",),
            actor_hidden_dims=(8,),
            compile_train_modules=False,
        )
        policy.action_dim = 2
        policy.device = torch.device("cpu")

        actor = policy._build_actor(list(policy.cfg.actor_in_keys))
        data = TensorDict(
            {
                "policy": torch.randn(3, 5),
                "command": torch.randn(3, 7),
                "priv": torch.randn(3, 11),
            },
            batch_size=[3],
        )
        actor(data)

        self.assertEqual(data["_actor_encoder_input"].shape, (3, 12))
        self.assertEqual(data["_actor_encoder_feature"].shape, (3, 4))
        self.assertEqual(data["_actor_input"].shape, (3, 9))

    def test_default_mode_and_clip_are_simple_mlp_and_four(self) -> None:
        cfg = PPOConfig()
        self.assertFalse(cfg.use_actor_encoder)
        self.assertEqual(cfg.max_grad_norm, 4.0)
        self.assertFalse(cfg.separate_actor_encoder_grad_clip)

    def test_default_optimization_recipe_matches_promoted_ppo(self) -> None:
        cfg = PPOConfig()
        self.assertEqual((cfg.ppo_epochs, cfg.num_minibatches), (5, 8))
        self.assertEqual(cfg.actor_hidden_dims, (1024, 512, 512))
        self.assertEqual(cfg.critic_hidden_dims, (1024, 512, 512))
        self.assertEqual(
            (
                cfg.entropy_coef_start,
                cfg.entropy_coef_end,
                cfg.entropy_decay_start,
                cfg.entropy_decay_end,
            ),
            (0.008, 0.002, 500, 3500),
        )
        self.assertEqual(cfg.grad_sync_mode, "ddp")
        self.assertEqual(cfg.train_amp_dtype, "bf16")
        self.assertFalse(cfg.compile)
        self.assertFalse(cfg.compile_rollout)
        self.assertFalse(cfg.compile_train_modules)

    def test_separate_clip_requires_actor_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires use_actor_encoder"):
            PPOConfig(separate_actor_encoder_grad_clip=True)


if __name__ == "__main__":
    unittest.main()
