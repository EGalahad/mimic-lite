from active_adaptation.learning.ppo.ppo_base import PPOBase


def _ppo_from_env(cls, cfg, env, device):
    runtime_env = getattr(env, "base_env", env)
    return cls(
        cfg=cfg,
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        reward_spec=env.reward_spec,
        device=device,
        env=runtime_env,
    )


if not hasattr(PPOBase, "from_env"):
    PPOBase.from_env = classmethod(_ppo_from_env)


from . import ppo
from . import ppo_roa
from . import sac
from . import fast_sac
from . import fast_td3
