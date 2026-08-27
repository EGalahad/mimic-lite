from __future__ import annotations

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.envs.mdp.terminations.base import Termination


class DeferredObservation(Observation):
    def __init__(self, *args, functional: bool = False, **kwargs) -> None:
        super().__init__(functional=functional)
        self._deferred_args = args
        self._deferred_kwargs = kwargs
        self._deferred_ready = False

    def _initialize(self, env) -> None:
        super()._initialize(env)
        if not self._deferred_ready:
            self._initialize_impl(*self._deferred_args, **self._deferred_kwargs)
            self._deferred_ready = True

    def _initialize_impl(self, *args, **kwargs) -> None:
        return None


class DeferredReward(Reward):
    def __init__(
        self,
        *args,
        weight: float,
        enabled: bool = True,
        track_var: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(weight=weight, enabled=enabled, track_var=track_var)
        self._deferred_args = args
        self._deferred_kwargs = kwargs
        self._deferred_ready = False

    def _initialize(self, env) -> None:
        super()._initialize(env)
        if not self._deferred_ready:
            self._initialize_impl(*self._deferred_args, **self._deferred_kwargs)
            self._deferred_ready = True

    def _initialize_impl(self, *args, **kwargs) -> None:
        return None


class DeferredTermination(Termination):
    def __init__(
        self,
        *args,
        is_timeout: bool = False,
        enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(is_timeout=is_timeout, enabled=enabled)
        self._deferred_args = args
        self._deferred_kwargs = kwargs
        self._deferred_ready = False

    def _initialize(self, env) -> None:
        super()._initialize(env)
        if not self._deferred_ready:
            self._initialize_impl(*self._deferred_args, **self._deferred_kwargs)
            self._deferred_ready = True

    def _initialize_impl(self, *args, **kwargs) -> None:
        return None
