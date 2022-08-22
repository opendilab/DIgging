import numpy as np
import pytest

from digging.digger import BayesianOptimizationDigger
from digging.problem import ContinuousSpace


def target_function(x):
    return -sum((x - 0.5) ** 2)


@pytest.mark.unittest
class TestBOEngine:

    config = dict()
    space = ContinuousSpace(shape=5, low=0, high=1)

    def test_iteration(self):
        digger = BayesianOptimizationDigger(self.config, self.space)
        for i in range(10):
            ans = digger.propose()
            scores = []
            for sample in ans:
                scores.append(target_function(sample))
            digger.update_score(ans, np.asarray(scores))
        print(digger.best)

    def test_search(self):
        config = {'max_iterations': 10}
        digger = BayesianOptimizationDigger(config, self.space)
        digger.search(target_function)
        print(digger.best)
