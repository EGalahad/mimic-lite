# FlashSAC references and migration decisions

Primary reference:

- `Holiday-Robot/FlashSAC` GitHub repository.
- Paper: `FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control` (`arXiv:2604.04539`).
- Files reviewed:
  - `flash_rl/agents/flashSAC/update.py`
  - `flash_rl/agents/flashSAC/network.py`
  - `flash_rl/agents/flashSAC/layer.py`
  - `configs/agent/flashSAC.yaml`
  - repository README and project structure.

What was migrated:

- SAC base objective with tanh Gaussian actor, automatic temperature and clipped-double distributional critic.
- Categorical critic TD target with min-Q target distribution selection.
- `n_step` returns.
- Reward normalization and normalized critic value support (`v_min=-5`, `v_max=5` by default).
- Low update-to-data ratio: `num_updates=1` with large batch sizes for GPU simulators.
- Actor update period (`policy_frequency=2`).
- Stability constraints: gradient clipping plus row-wise linear weight normalization and parameter clipping after optimizer steps.

What was adapted instead of copied:

- The official FlashSAC network uses UnitLinear, UnitBatchNorm, UnitRMSNorm and compiled helper kernels. This repo extension uses the shared off-policy MLP/LayerNorm stack to stay compatible with the current PPO-style codebase, then adds row-wise weight normalization and gradient clipping to capture the core bounded-update mechanism without importing the full external framework.
- The official default config uses smaller model/batch sizes for generic GPU simulators. For 25--30 DoF whole-body tracking with thousands of vectorized envs, this implementation extrapolates to `hidden_dim=1024`, `batch_size=65536`, `buffer_size=1024`, while keeping `num_updates=1`.
- Replay buffer, checkpoint and logging follow the current repository's TensorDict/Hydra/WandB conventions.
