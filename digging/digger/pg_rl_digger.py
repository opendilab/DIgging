from typing import Any, Dict
import torch.nn as nn

from .rl_digger import RLDigger
from .base_digger import DIGGER_REGISTRY
from ding.policy import PPOPGPolicy
from ding.utils import deep_merge_dicts


@DIGGER_REGISTRY.register('pg_rl')
class PolicyGradientDigger(RLDigger):
    """
    RL digger of Policy Gradient algorithm.

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
            model: nn.Module() = None,
    ) -> None:
        cfg.policy = deep_merge_dicts(PPOPGPolicy.default_config(), cfg.policy)
        policy = PPOPGPolicy(cfg.policy, model)
        super().__init__(cfg, search_space, policy, None)
