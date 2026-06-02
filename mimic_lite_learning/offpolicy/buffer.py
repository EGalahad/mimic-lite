from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import threading

import torch
from tensordict import TensorDictBase

from active_adaptation.learning.ppo.common import (
    DONE_KEY,
    REWARD_KEY,
    TERM_KEY,
)


BOOTSTRAP_KEY = "bootstrap"
ENV_ID_KEY = "_env_id"
N_STEP_REWARD_KEY = "_n_step_reward"
N_STEP_BOOTSTRAP_KEY = "_n_step_bootstrap"
N_STEP_DISCOUNT_KEY = "_n_step_discount"


class CudaPrefetchNStepReplayBuffer:
    def __init__(
        self,
        *,
        capacity_per_env: int,
        num_envs: int,
        n_step: int,
        batch_size: int,
        gamma: float,
        device: torch.device,
        prefetch: int = 2,
        compact: bool = True,
    ) -> None:
        self.capacity_per_env = int(capacity_per_env)
        self.num_envs = int(num_envs)
        self.n_step = int(n_step)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.prefetch = max(0, int(prefetch))
        self.compact = bool(compact)
        self.storage: TensorDictBase | None = None
        self.cursor = 0
        self.length = 0
        self.lock = threading.RLock()
        self.prefetch_queue: deque[Future[TensorDictBase]] = deque()
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="fast-sac-replay")
            if self.prefetch > 0
            else None
        )
        self.copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def __len__(self) -> int:
        return self.length * self.num_envs

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def __del__(self) -> None:
        self.close()

    def extend(self, data: TensorDictBase) -> None:
        data = data.detach()
        if data.device is not None and data.device.type != "cpu":
            data = data.cpu()
        if data.batch_dims == 1:
            data = data.unsqueeze(0)
        if data.batch_dims != 2 or data.batch_size[1] != self.num_envs:
            raise ValueError(
                "Expected replay data with batch size [T, num_envs] or [num_envs], "
                f"got {tuple(data.batch_size)} for num_envs={self.num_envs}."
            )

        with self.lock:
            if self.storage is None:
                sample = data[0]
                self.storage = sample.apply(
                    lambda value: torch.empty(
                        (self.capacity_per_env, *value.shape),
                        dtype=value.dtype,
                        device="cpu",
                    ),
                    batch_size=(self.capacity_per_env, *sample.batch_size),
                )

            num_steps = int(data.batch_size[0])
            start = 0
            while start < num_steps:
                write_count = min(num_steps - start, self.capacity_per_env - self.cursor)
                self.storage[self.cursor : self.cursor + write_count] = data[
                    start : start + write_count
                ]
                self.cursor = (self.cursor + write_count) % self.capacity_per_env
                self.length = min(self.capacity_per_env, self.length + write_count)
                start += write_count

        self._fill_prefetch()

    def sample(self) -> TensorDictBase:
        if self.executor is None:
            return self._sample_to_device()

        self._fill_prefetch()
        if not self.prefetch_queue:
            return self._sample_to_device()
        future = self.prefetch_queue.popleft()
        batch = future.result()
        self._fill_prefetch()
        return batch

    def _fill_prefetch(self) -> None:
        if self.executor is None or self.length < self.n_step:
            return
        while len(self.prefetch_queue) < self.prefetch:
            self.prefetch_queue.append(self.executor.submit(self._sample_to_device))

    def _sample_to_device(self) -> TensorDictBase:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        with self.lock:
            if self.storage is None or self.length < self.n_step:
                raise RuntimeError(
                    f"Not enough replay data to sample n_step={self.n_step}: "
                    f"length={self.length}."
                )
            length = self.length
            cursor = self.cursor
            oldest = cursor if length == self.capacity_per_env else 0
            starts = torch.randint(
                0,
                length - self.n_step + 1,
                (self.batch_size,),
                dtype=torch.long,
            )
            envs = torch.randint(
                0,
                self.num_envs,
                (self.batch_size, 1),
                dtype=torch.long,
            )
            offsets = torch.arange(self.n_step, dtype=torch.long).unsqueeze(0)
            time_idx = (oldest + starts.unsqueeze(1) + offsets) % self.capacity_per_env
            env_idx = envs.expand(-1, self.n_step)
            if self.compact:
                batch = self._sample_compact_locked(time_idx, env_idx)
            else:
                batch = self.storage[time_idx, env_idx]

        if self.device.type != "cuda":
            return batch.to(self.device)

        batch = batch.pin_memory()
        assert self.copy_stream is not None
        with torch.cuda.stream(self.copy_stream):
            batch = batch.to(self.device, non_blocking=True)
        self.copy_stream.synchronize()
        return batch

    def _sample_compact_locked(
        self,
        time_idx: torch.Tensor,
        env_idx: torch.Tensor,
    ) -> TensorDictBase:
        assert self.storage is not None
        batch_size, n_step = time_idx.shape
        env_flat = env_idx[:, 0]
        rewards = self.storage[REWARD_KEY][time_idx, env_idx]
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)
        rewards = rewards.squeeze(-1)
        dones = self.storage[DONE_KEY][time_idx, env_idx].bool().squeeze(-1)
        terminated = self.storage[TERM_KEY][time_idx, env_idx].bool().squeeze(-1)

        adjusted_rewards = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap_discount = torch.zeros(batch_size, dtype=rewards.dtype)
        bootstrap_idx = torch.zeros(batch_size, dtype=torch.long)
        discount = torch.ones(batch_size, dtype=rewards.dtype)
        active = torch.ones(batch_size, dtype=torch.bool)

        for step_idx in range(n_step):
            step_active = active
            if not step_active.any():
                break

            adjusted_rewards = adjusted_rewards + torch.where(
                step_active,
                discount * rewards[:, step_idx],
                torch.zeros_like(adjusted_rewards),
            )
            can_bootstrap = step_active & ~terminated[:, step_idx]
            next_discount = discount * self.gamma
            boundary = can_bootstrap & dones[:, step_idx]
            if boundary.any():
                bootstrap_idx = torch.where(
                    boundary,
                    torch.full_like(bootstrap_idx, step_idx),
                    bootstrap_idx,
                )
                bootstrap_discount = torch.where(boundary, next_discount, bootstrap_discount)
                bootstrap = torch.where(boundary, torch.ones_like(bootstrap), bootstrap)

            active = can_bootstrap & ~dones[:, step_idx]
            discount = torch.where(active, next_discount, discount)

        if active.any():
            bootstrap_idx = torch.where(
                active,
                torch.full_like(bootstrap_idx, n_step - 1),
                bootstrap_idx,
            )
            bootstrap_discount = torch.where(active, discount, bootstrap_discount)
            bootstrap = torch.where(active, torch.ones_like(bootstrap), bootstrap)

        batch_idx = torch.arange(batch_size)
        compact = self.storage[time_idx[:, 0], env_flat].copy()
        compact.set(
            BOOTSTRAP_KEY,
            self.storage[BOOTSTRAP_KEY][time_idx[batch_idx, bootstrap_idx], env_flat],
        )
        compact.set(N_STEP_REWARD_KEY, adjusted_rewards.unsqueeze(-1))
        compact.set(N_STEP_BOOTSTRAP_KEY, bootstrap.unsqueeze(-1))
        compact.set(N_STEP_DISCOUNT_KEY, bootstrap_discount.unsqueeze(-1))
        return compact.unsqueeze(1)

