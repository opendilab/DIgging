import numpy as np
import pytest

from digging.digger import GeneticAlgorithmDigger
from digging.problem import DiscreteSpace


def target_function(x):
    x = x / 4 * 2 - 1
    return -sum((x - 0.5) ** 2)


@pytest.mark.unittest
class TestGADigger:

    config = dict()
    space = DiscreteSpace([5, 5, 5, 5, 5])

    def test_search(self):
        digger = GeneticAlgorithmDigger(self.config, self.space)
        digger.search(target_function)
        print(digger.provide_best())

    def test_iteration(self):
        digger = GeneticAlgorithmDigger(self.config, self.space)

        for i in range(50):
            ans = digger.propose(20)
            scores = []
            for seq in ans:
                scores.append(target_function(seq))
            digger.update_score(ans, np.array(scores))
        print(digger.provide_best())
