import pytest
import numpy as np

from digging.digger import RandomDigger
from digging.problem import DiscreteSpace, ContinuousSpace, DictSpace


def target_discrete(x):
    x = x / 4 * 2 - 1
    return -np.sum((x - 0.5) ** 2)


def target_continuous(x):
    return -np.sum((x - 0.5) ** 2)


def target_dict(x):
    x0, x1 = x['x0'], x['x1']
    return target_discrete(x0) + target_continuous(x1)


@pytest.mark.unittest
class TestRandomDigger:

    config = dict()

    def test_discrete(self):
        space = DiscreteSpace([3, 4])
        engine = RandomDigger(self.config, space)
        engine.search(target_discrete)
        sample, score = engine.best
        assert sample.shape == (2, )
        assert 0 <= sample[0] < 3
        assert 0 <= sample[1] < 4

    def test_continuous(self):
        space = ContinuousSpace((2, 3), low=0)
        engine = RandomDigger(self.config, space)
        engine.search(target_continuous)
        sample, score = engine.best
        assert sample.shape == (2, 3)
        assert (sample > 0).all()

    def test_dict_space(self):
        space = DictSpace(x0=DiscreteSpace(5), x1=ContinuousSpace(5))
        engine = RandomDigger(self.config, space)
        engine.search(target_dict)
        sample, score = engine.best
        assert sample['x0'].shape == (1, )
        assert sample['x1'].shape == (5, )
