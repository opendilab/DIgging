import pytest
import numpy as np

from digging.problem import ProblemHandler, ContinuousSpace


@pytest.mark.unittest
class TestProblemHandler:

    def test_common(self, make_dict_space, make_dict_target_function):
        space = make_dict_space
        target_func = make_dict_target_function
        handler = ProblemHandler(space)
        assert handler.space == space
        assert len(handler) == 0
        sample = {
            'x': np.array([0, 1], dtype=np.int64),
            'y': np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        }
        score = target_func(sample)
        handler.update_data([sample], np.asarray([score]))
        assert len(handler) == 1
        scores = handler.get_cached_score([sample])
        assert scores[0] == -0.5

    def test_complex(self, make_tuple_in_dict_space, make_tuple_in_dict_target_function):
        space = make_tuple_in_dict_space
        target_func = make_tuple_in_dict_target_function
        handler = ProblemHandler(space)
        sample = space.sample()
        score = target_func(sample)
        handler.update_data([sample], np.asarray([score]))
        assert len(handler) == 1
        scores = handler.get_cached_score([sample])
        assert scores[0] < 0

    def test_best(self, make_dict_target_function):
        target_func = make_dict_target_function
        space = ContinuousSpace(1)
        handler = ProblemHandler(space)
        x = np.asarray([[0.2], [0.5], [1]], dtype=np.float32)
        scores = [target_func({'x': xi, 'y': None}) for xi in x]
        handler.update_data(x, np.asarray(scores))
        best_dict = handler.best
        assert best_dict['score'] == 0
