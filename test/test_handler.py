import pytest
import numpy as np

from digging.problem import ProblemHandler, ContinuousSpace, DiscreteSpace
from digging.problem.space import DictSpace


def target_function(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    return -sum((x - 0.5) ** 2)


@pytest.mark.unittest
class TestProblemHandler:

    def test_common(self):
        space = DictSpace(x=DiscreteSpace(5), y=ContinuousSpace(5))
        handler = ProblemHandler(space)
        assert handler.space == space
        assert len(handler) == 0
        sample = {
            'x': np.array([0], dtype=np.int64),
            'y': np.array([0, 0, 0, 0, 0], dtype=np.float32)
        }
        score = target_function(sample)
        handler.update_data([sample], np.asarray([score]))
        assert len(handler) == 1
        scores = handler.get_cached_score([sample])
        assert scores[0] == -0.25

    def test_best(self):
        space = ContinuousSpace(1)
        handler = ProblemHandler(space)
        x = np.asarray([[0.2], [0.5], [1]], dtype=np.float32)
        scores = [target_function({'x': xi, 'y': None}) for xi in x]
        print(x, scores)
        handler.update_data(x, np.asarray(scores))
        best_x, best_y = handler.provide_best()
        assert best_y == 0
