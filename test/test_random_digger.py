import pytest
import numpy as np

from digging.digger.random_digger import RandomDigger
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

    def test_common(self):
        space = DiscreteSpace(5)
        digger = RandomDigger(self.config, space)
        assert digger.space == space
        digger = RandomDigger(self.config, space, random_state=1)
        digger = RandomDigger(self.config, space, random_state=np.random.RandomState(2))
        samples = digger.propose(10)
        assert digger._start
        assert samples.shape == (10, 1)
        digger.reset()
        assert not digger._start
        samples = np.asarray([[4], [0]])
        scores = np.asarray([target_discrete(s) for s in samples])
        digger.update_score(samples, scores)
        best = digger.provide_best()
        assert best['sample'] == [4, ]
        assert best['score'] == -0.25

    def test_discrete(self):
        space = DiscreteSpace([3, 4])
        digger = RandomDigger(self.config, space)
        digger.search(target_discrete)
        best = digger.provide_best()
        sample, score = best['sample'], best['score']
        assert sample.shape == (2, )
        assert 0 <= sample[0] < 3
        assert 0 <= sample[1] < 4

    def test_continuous(self):
        space = ContinuousSpace((2, 3), low=0)
        digger = RandomDigger(self.config, space)
        digger.search(target_continuous)
        best = digger.provide_best()
        sample, score = best['sample'], best['score']
        assert sample.shape == (2, 3)
        assert (sample > 0).all()

    def test_dict_space(self):
        space = DictSpace(x0=DiscreteSpace(5), x1=ContinuousSpace(5))
        digger = RandomDigger(self.config, space)
        digger.search(target_dict)
        best = digger.provide_best()
        sample, score = best['sample'], best['score']
        assert sample['x0'].shape == (1, )
        assert sample['x1'].shape == (5, )
