import pytest
import numpy as np

from digging.problem import DiscreteSpace, ContinuousSpace, TupleSpace, DictSpace



@pytest.fixture(scope='function')
def make_tuple_space():
    space_a = DiscreteSpace([2, 4])
    space_b = ContinuousSpace((3, 3))
    space = TupleSpace(space_a, space_b)
    return space


@pytest.fixture(scope='function')
def make_dict_space():
    space_a = DiscreteSpace([3, 4])
    space_b = ContinuousSpace((2, 3))
    space = DictSpace(x=space_a, y=space_b)
    return space


@pytest.fixture(scope='function')
def make_dict_target_function():
    def target_func(data_dict):
        x = data_dict['x']
        y = data_dict['y']
        return -sum((x - 0.5) ** 2)
    return target_func


@pytest.fixture(scope='function')
def make_tuple_in_tuple_space():
    space_a = DiscreteSpace([2, 4])
    space_b = ContinuousSpace((3, 3))
    space_1 = TupleSpace(space_a, space_b)
    space_c = ContinuousSpace(5, np.array([-5, -4, -3, -2, -1]), 0)
    space = TupleSpace(space_1, space_c)
    return space


@pytest.fixture(scope='function')
def make_same_shape_tuple_space():
    space_a = DiscreteSpace([2, 4])
    space_b = ContinuousSpace((2, 3))
    space = TupleSpace(space_a, space_b)
    return space


@pytest.fixture(scope='function')
def make_tuple_in_dict_space():
    space_1 = DiscreteSpace([2, 4])
    space_a = ContinuousSpace((3, 3))
    space_b = DiscreteSpace([5], dtype=np.uint8)
    space_2 = TupleSpace(space_a, space_b)
    space = DictSpace(discrete=space_1, tuple=space_2)
    return space


@pytest.fixture(scope='function')
def make_tuple_in_dict_target_function():
    def target_func(data_dict):
        dis = data_dict['discrete']
        tup = data_dict['tuple']
        x, y = tup
        return -np.sum((x - 0.5)**2 + (y - 0.5)**2)
    return target_func


@pytest.fixture(scope='function')
def make_dict_in_dict_space():
    space_a = DiscreteSpace([2, 4])
    space_b = ContinuousSpace((3, 3))
    space_1 = DictSpace(a=space_a, b=space_b)
    space_2 = DiscreteSpace([5], dtype=np.uint8)
    space = DictSpace(dict=space_1, discrete=space_2)
    return space
