import pytest
import numpy as np

from digging.problem import DiscreteSpace, ContinuousSpace, TupleSpace, DictSpace


@pytest.mark.unittest
class TestDiscreteSpace():

    def test_single_value(self):
        space = DiscreteSpace(3)
        assert len(space) == 1
        assert space.nshape == 3
        x = space.sample()
        assert space.shape == x.shape
        assert 0 <= x < 3

    def test_single_array(self):
        space = DiscreteSpace([3])
        assert len(space) == 1
        assert space.nshape == 3
        x = space.sample()
        assert 0 <= x < 3
        assert space.shape == x.shape

    def test_array(self):
        space = DiscreteSpace([3, 4, 5])
        assert len(space) == 3
        assert (space.nshape == [3, 4, 5]).all()
        x = space.sample()
        assert space.shape == x.shape
        assert (x < np.array([3, 4, 5])).all()
        assert (x >= np.array([0, 0, 0])).all()
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x == y).all()
        x = space.sample_single(1)
        assert 0 <= x < 4


@pytest.mark.unittest
class TestContinuousSpace():

    def test_bound(self):
        space = ContinuousSpace(5, np.array([-5, -4, -3, -2, -1]), 0)
        assert len(space) == 5
        assert space.shape == [5]
        assert space.bounds.shape == (5, 2)
        x = space.sample()
        assert (x < 0).all()
        assert (x > np.array([-5, -4, -3, -2, -1])).all()

    def test_array(self):
        space = ContinuousSpace([3, 4])
        assert not space._bounded_above.any() and not space._bounded_below.any()
        assert len(space) == 12
        assert (space.shape == (3, 4)).all()
        x = space.sample()
        assert x.shape == (3, 4)
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x == y).all()


@pytest.mark.unittest
class TestTupleSpace():
    space_a = DiscreteSpace([2, 4])
    space_b = ContinuousSpace((3, 3))
    space_1 = TupleSpace(space_a, space_b)

    def test_common(self):
        assert len(self.space_1) == 2
        assert self.space_1.shape[0] == (2, ) and (self.space_1.shape[1] == (3, 3)).all()
        assert (self.space_1.nshape[0] == (2, 4)).all() and (self.space_1.nshape[1] == (3, 3)).all()

        x = self.space_1.sample()
        assert x[0].shape == (2, ) and x[1].shape == (3, 3)
        y = self.space_1.convert_to_sample(self.space_1.convert_to_data(x))
        assert (x[0] == y[0]).all() and (x[1] == y[1]).all()

    space_c = ContinuousSpace(5, np.array([-5, -4, -3, -2, -1]), 0)
    space_2 = TupleSpace(space_1, space_c)

    def test_tuple_in_tuple(self):
        assert len(self.space_2) == 2
        assert self.space_2.shape[0][0] == (2, ) and (self.space_2.shape[0][1] == (3, 3)).all()
        assert self.space_2.shape[1] == (5)
        x = self.space_2.sample()
        y = self.space_2.convert_to_sample(self.space_2.convert_to_data(x))
        assert (x[0][0] == y[0][0]).all()
        assert (x[0][1] == y[0][1]).all()
        assert (x[1] == y[1]).all()

    space_d = ContinuousSpace((2, 3))
    space_3 = TupleSpace(space_a, space_d)

    def test_same_shape(self):
        assert len(self.space_3) == 2
        x = self.space_3.sample()
        y = self.space_3.convert_to_sample(self.space_3.convert_to_data(x))
        assert (x[0] == y[0]).all()
        assert (x[1] == y[1]).all()


@pytest.mark.unittest
class TestDictSpace():

    def test_tuple_in_dict(self):
        space_1 = DiscreteSpace([2, 4])
        space_a = ContinuousSpace((3, 3))
        space_b = DiscreteSpace([5], dtype=np.uint8)
        space_2 = TupleSpace(space_a, space_b)
        space = DictSpace(discrete=space_1, tuple=space_2)
        x = space.sample()
        assert set(x.keys()) == set(('discrete', 'tuple'))
        assert x['discrete'].shape == (2, )
        assert x['tuple'].shape == (2, )
        assert x['tuple'][0].shape == (3, 3)
        assert x['tuple'][1].dtype == np.uint8
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x['discrete'] == y['discrete']).all()
        assert (x['tuple'][0] == y['tuple'][0]).all()
        assert (x['tuple'][1] == y['tuple'][1]).all()
