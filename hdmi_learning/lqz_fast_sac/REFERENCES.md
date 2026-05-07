# Fast-SAC references and migration decisions

Primary reference:

- Amazon FAR Holosoma repository: `amazon-far/holosoma`.
- Files reviewed:
  - `src/holosoma/holosoma/agents/fast_sac/fast_sac.py`
  - `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`
  - `src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py`
  - `src/holosoma/holosoma/config_values/algo.py`
  - `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
  - `src/holosoma/holosoma/config_values/loco/g1/experiment.py`

What was migrated:

- Tanh-squashed Gaussian actor with log-probability correction.
- Distributional twin critic with categorical support and C51-style projection.
- Soft Bellman target `r + gamma * (Q_target(s', a') - alpha * log pi(a'|s'))` implemented as reward shift before distributional projection.
- Automatic temperature tuning with `target_entropy = -action_dim * target_entropy_ratio`.
- FastSAC WBT defaults for G1 29-DoF from Holosoma: `num_envs=4096`, `num_learning_iterations=400000`, `gamma=0.99`, `num_steps=1`, `num_updates=4`, `num_atoms=501`, `policy_frequency=2`, `target_entropy_ratio=0.5`, `tau=0.05`, `v_min=-20`, `v_max=20`, `use_symmetry=False`.
- Large-batch, GPU replay-buffer style with one vectorized environment step per iteration.

What was adapted instead of copied:

- Holosoma wraps Isaac/MJWarp envs directly; this repo already exposes TorchRL `TensorDict` transitions through `env.step_and_maybe_reset`, so the implementation keeps the existing env/reward/VecNorm/action manager pipeline.
- Holosoma has its own empirical normalizers. This implementation uses the repository's existing VecNorm transform selected by `cfg.vecnorm`.
- Holosoma has optional symmetry augmentation and CNN encoders. These are not enabled by default because the current PPO context uses policy/priv/priv_critic TensorDict groups and the WBT FastSAC reference disables symmetry.
- Optimizer state/checkpoint layout follows this repository's PPO checkpoint convention rather than Holosoma's agent directory convention.
