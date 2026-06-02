import torch
from tensordict import TensorDict
from typing import Optional, Tuple

from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    OBS_KEY,
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
)

DISCOUNT_KEY = ("next", "discount")


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        fake_tensordict: TensorDict,
        gamma: float,
        obs_keys: Tuple[str, ...] = (OBS_KEY,),
    ):
        self.max_size = max_size
        self.num_envs = fake_tensordict.shape[0]
        self.device = fake_tensordict.device
        self.gamma = float(gamma)
        self.obs_keys = tuple(obs_keys)
        self._current_size = 0
        self._td = fake_tensordict.expand(max_size, *fake_tensordict.shape).clone()
        self._storage_keys = tuple(fake_tensordict.keys(True, True))
        self._ptr = 0

    def push(self, tensordict: TensorDict):
        self._td[self._ptr] = tensordict.select(
            *self._storage_keys,
            strict=False,
        ).to(self.device)
        self._ptr = (self._ptr + 1) % self._td.shape[0]
        self._current_size = min(self._current_size + 1, self.max_size)
    
    append = push
    
    def last(self, steps: int) -> TensorDict:
        """
        Returns the last `steps` samples from the buffer.
        """
        assert len(self) >= steps, "Not enough samples in buffer"
        if self._ptr >= steps:
            samples = self._td[self._ptr - steps:self._ptr].clone()
        else:
            part1 = self._td[-(steps - self._ptr):]
            part2 = self._td[:self._ptr]
            samples = torch.cat([part1, part2], dim=0)
        assert samples.shape[0] == steps, "Not enough samples in buffer"
        return samples

    @property
    def num_samples(self):
        return self._td.shape[1] * len(self)

    def _valid_start_rows(self, steps: int) -> torch.Tensor:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}.")
        size = len(self)
        if size <= steps:
            return torch.empty(0, dtype=torch.long, device=self._td.device)
        if size < self.max_size:
            return torch.arange(size - steps, device=self._td.device)
        return (self._ptr + torch.arange(size - steps, device=self._td.device)) % size

    def _sample_rows_envs(self, batch_size: int, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_rows = self._valid_start_rows(steps)
        if start_rows.numel() == 0:
            raise RuntimeError(
                "Cannot sample replay transitions with bootstrap observations: "
                f"len={len(self)}, steps={steps}."
            )
        start_idx = torch.randint(
            0,
            start_rows.numel(),
            (batch_size,),
            device=self._td.device,
        )
        starts = start_rows[start_idx]
        envs = torch.randint(0, self.num_envs, (batch_size,), device=self._td.device)
        offsets = torch.arange(steps + 1, device=self._td.device).unsqueeze(1)
        rows = (starts.unsqueeze(0) + offsets) % self.max_size
        return rows, envs

    def _build_training_batch(
        self,
        rows: torch.Tensor,
        envs: torch.Tensor,
        steps: int,
    ) -> TensorDict:
        transitions = self._td[rows[:-1], envs]
        rewards = transitions[REWARD_KEY]
        done = transitions[DONE_KEY].bool()
        terminated = transitions[TERM_KEY].bool()

        batch_size = envs.shape[0]
        batch_indices = torch.arange(batch_size, device=self._td.device)
        done_flat = done.squeeze(-1)
        terminated_flat = terminated.squeeze(-1)

        has_done = done_flat.any(dim=0)
        first_done = done_flat.float().argmax(dim=0)
        last_transition = torch.where(
            has_done,
            first_done,
            torch.full_like(first_done, steps - 1),
        )

        gamma = torch.as_tensor(self.gamma, device=rewards.device, dtype=rewards.dtype)
        gammas = gamma ** torch.arange(steps, device=rewards.device, dtype=rewards.dtype)
        reward_shape = (steps,) + (1,) * (rewards.ndim - 1)
        cumulative_rewards = (rewards * gammas.reshape(reward_shape)).cumsum(dim=0)
        reward = cumulative_rewards[last_transition, batch_indices]

        terminated_at_bootstrap = terminated_flat[last_transition, batch_indices]
        discount = gamma.pow(last_transition.to(rewards.dtype) + 1.0).reshape(
            batch_size,
            1,
        )
        discount = discount * (~terminated_at_bootstrap).to(rewards.dtype).reshape(
            batch_size,
            1,
        )

        boundary_rows = rows.gather(0, last_transition.reshape(1, batch_size)).squeeze(0)
        next_rows = torch.where(has_done, boundary_rows, rows[-1])
        data = {
            key: transitions[0][key].clone()
            for key in self.obs_keys
        }
        data.update({
            ACTION_KEY: transitions[0][ACTION_KEY].clone(),
            "next": {
                key: self._td[next_rows, envs][key].clone()
                for key in self.obs_keys
            } | {
                "reward": reward,
                "discount": discount,
            },
        })
        if "loc" in transitions.keys():
            data["loc"] = transitions[0]["loc"].clone()

        return TensorDict(
            data,
            batch_size=[batch_size],
            device=self._td.device,
        )

    def sample(self, batch_size: int, steps: int=1) -> TensorDict:
        rows, envs = self._sample_rows_envs(batch_size, steps)
        return self._build_training_batch(rows, envs, steps)
    
    def sample_sequential(
        self,
        batch_size: int,
        steps: int=1,
        last_indices: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
        sequential_prob: float=0.0,
        sequential_offset: int=-1,
    ) -> Tuple[TensorDict, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample transitions with optional temporal correlation along the replay ring.

        Most indices are drawn i.i.d., like :meth:`sample`. With probability
        ``sequential_prob``, each element instead steps **backward** along the ring
        time index for the same env id, matching the direction rewards propagate
        under dynamic programming (earlier stored step = one step back in ring).

        Ring indices wrap with ``% len(self)``. After a minibatch, pass the returned
        ``(t, e)`` back as ``last_indices`` on the next call to chain segments.

        Args:
            batch_size: Number of processed training transitions.
            steps: Number of rewards to fold into each sampled transition.
            last_indices: Previous ``(t, e)`` tensors from this method, same length
                as ``batch_size``, or ``None`` for independent draws only.
            sequential_prob: In ``(0, 1]``, fraction of elements (in expectation)
                that reuse ``last_indices`` shifted by ``sequential_offset`` instead
                of a fresh random index. ``0`` disables chaining.
            sequential_offset: Added to ``last_indices[0]`` before wrapping; default
                ``-1`` is one step **backward** in ring time (reward backup direction).

        Returns:
            ``(samples, (t, e))`` — processed training batch and the time/env indices
            used for each row (for feeding back as ``last_indices``).
        """
        rows, e = self._sample_rows_envs(batch_size, steps)
        t = rows[0]
        
        if last_indices is not None and sequential_prob > 0.0:
            candidate_t = (last_indices[0] + sequential_offset) % self.max_size
            valid_rows = self._valid_start_rows(steps)
            candidate_valid = (candidate_t.unsqueeze(-1) == valid_rows).any(dim=-1)
            use_seq = (
                torch.rand(batch_size, device=self._td.device) <= sequential_prob
            ) & candidate_valid
            t = torch.where(use_seq, candidate_t, t)
            e = torch.where(use_seq, last_indices[1], e)
            offsets = torch.arange(steps + 1, device=self._td.device).unsqueeze(1)
            rows = (t.unsqueeze(0) + offsets) % self.max_size
        
        return self._build_training_batch(rows, e, steps), (t, e)

    def __len__(self):
        return self._current_size
