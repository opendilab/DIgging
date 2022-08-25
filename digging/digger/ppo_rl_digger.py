from typing import Dict, Optional
import torch.nn as nn

from .rl_digger import RLDigger
from .base_digger import DIGGER_REGISTRY
from ding.policy import PPOPolicy, PPOOffPolicy
from ding.worker import EpisodeReplayBuffer
from ding.utils import deep_merge_dicts


@DIGGER_REGISTRY.register('ppo_rl')
class PPODigger(RLDigger):
    """
    RL searching digger of on-policy PPO algorithm.

    :param Dict cfg: user config
    :param BaseSpace search_space: searching space of digger
    :param nn.Module model: NN models within the RL policy, defaults to None
    """

    config = dict(
        samples_per_iteration=128,
        max_iterations=100,
        env=dict(),
        policy=dict(collect=dict(collector=dict(), ), learn=dict(learner=dict(), )),
    )

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            model: Optional[nn.Module()] = None,
    ) -> None:
        cfg.policy = deep_merge_dicts(PPOPolicy.default_config(), cfg.policy)
        policy = PPOPolicy(cfg.policy, model)
        super().__init__(cfg, search_space, policy, None)


@DIGGER_REGISTRY.register('ppo_offpolicy_rl')
class PPOOffPolicyDigger(RLDigger):
    """
    RL searching digger of off-policy PPO algorithm.

    :param Dict cfg: user config
    :param BaseSpace search_space: searching space of digger
    :param nn.Module model: NN models within the RL policy, defaults to None
    """

    config = dict(
        samples_per_iteration=128,
        max_iterations=100,
        env=dict(),
        policy=dict(
            collect=dict(collector=dict(), ),
            learn=dict(learner=dict(), ),
            other=dict(replay_buffer=dict(), ),
        ),
    )

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            model: Optional[nn.Module()] = None,
    ) -> None:
        cfg.policy = deep_merge_dicts(PPOOffPolicy.default_config(), cfg.policy)
        cfg.policy.other.replay_buffer = deep_merge_dicts(
            EpisodeReplayBuffer.default_config(), cfg.policy.other.replay_buffer
        )
        policy = PPOOffPolicy(cfg.policy, model)
        replay_buffer = EpisodeReplayBuffer(cfg.policy.other.replay_buffer)
        super().__init__(cfg, search_space, policy, replay_buffer)
