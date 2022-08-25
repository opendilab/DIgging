import pytest
import numpy as np

from digging.problem import DiscreteSpace, ContinuousSpace


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

    def test_common(self, make_tuple_space):
        space = make_tuple_space
        assert len(space) == 2
        assert space.shape[0] == (2, ) and (space.shape[1] == (3, 3)).all()
        assert (space.nshape[0] == (2, 4)).all() and (space.nshape[1] == (3, 3)).all()

        x = space.sample()
        assert x[0].shape == (2, ) and x[1].shape == (3, 3)
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x[0] == y[0]).all() and (x[1] == y[1]).all()

    def test_tuple_in_tuple(self, make_tuple_in_tuple_space):
        space = make_tuple_in_tuple_space
        assert len(space) == 2
        assert space.shape[0][0] == (2, ) and (space.shape[0][1] == (3, 3)).all()
        assert space.shape[1] == (5)
        x = space.sample()
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x[0][0] == y[0][0]).all()
        assert (x[0][1] == y[0][1]).all()
        assert (x[1] == y[1]).all()


    def test_same_shape(self, make_same_shape_tuple_space):
        space = make_same_shape_tuple_space
        assert len(space) == 2
        x = space.sample()
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x[0] == y[0]).all()
        assert (x[1] == y[1]).all()


@pytest.mark.unittest
class TestDictSpace():

    def test_common(self, make_dict_space):
        space = make_dict_space
        x = space.sample()
        assert set(x.keys()) == set(('x', 'y'))
        assert x['x'].shape == (2, )
        assert x['y'].shape == (2, 3)
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x['x'] == y['x']).all()
        assert (x['y'] == y['y']).all()

    def test_tuple_in_dict(self, make_tuple_in_dict_space):
        space = make_tuple_in_dict_space
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


    def test_dict_in_dict(self, make_dict_in_dict_space):
        space = make_dict_in_dict_space
        x = space.sample()
        assert set(x.keys()) == set(('discrete', 'dict'))
        assert x['discrete'].shape == (1, )
        assert len(x['dict']) == 2
        assert x['dict']['b'].shape == (3, 3)
        assert x['dict']['a'].dtype == np.int64
        y = space.convert_to_sample(space.convert_to_data(x))
        assert (x['discrete'] == y['discrete']).all()
        assert (x['dict']['a'] == y['dict']['a']).all()
        assert (x['dict']['b'] == y['dict']['b']).all()
