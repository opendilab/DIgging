import pytest
import numpy as np

from digging.digger import PolicyGradientDigger, PPODigger, PPOOffPolicyDigger
from digging.model import RNNDiscretePGModel, RNNVACModel
from digging.problem import DiscreteSpace, ContinuousSpace


def target_discrete(x):
    x = x / 4 * 2 - 1
    return -np.sum((x - 0.5) ** 2)


@pytest.mark.unittest
class TestPGEngine:

    def test_discrete(self, make_rl_digging_config):
        cfg = make_rl_digging_config
        space = DiscreteSpace((5, 5), dtype=np.int64)
        model = RNNDiscretePGModel(5, 5, 128, 2, head_hidden_size=128)
        digger = PolicyGradientDigger(cfg, space, model)
        res = digger.search(target_discrete)
        print(res)


@pytest.mark.unittest
class TestPPODigger:

    def test_discrete(self, make_rl_digging_config):
        cfg = make_rl_digging_config
        space = DiscreteSpace((5, 5), dtype=np.int64)
        model = RNNVACModel(5, 5, 2, critic_head_hidden_size=128)
        digger = PPODigger(cfg, space, model)
        res = digger.search(target_discrete)
        print(res)

    def test_continuous(self, make_rl_digging_config):
        cfg = make_rl_digging_config
        cfg.env['obs_shape'] = (2,)
        cfg.policy['action_space'] = 'continuous'
        space = ContinuousSpace(2, low=0, high=1, dtype=np.float32)
        model = RNNVACModel(2, 1, 2, critic_head_hidden_size=128, action_space='continuous')
        digger = PPODigger(cfg, space, model)
        res = digger.search(target_discrete)
        print(res)


@pytest.mark.unittest
class TestPPOOffPolicyDigger:

    def test_discrete(self, make_rl_digging_config):
        cfg = make_rl_digging_config
        space = DiscreteSpace((5, 5), dtype=np.int64)
        model = RNNVACModel(5, 5, 2, critic_head_hidden_size=128)
        digger = PPOOffPolicyDigger(cfg, space, model)
        res = digger.search(target_discrete)
        print(res)
