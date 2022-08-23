from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
import numpy as np


class BaseSpace(ABC):
    r"""
    Abstract class of `space`.

    :param List shape: The shape of space, defaults to None.
    :param Type dtype: The type of space, defaults to None.
    :param Any random_state: The random method for space to generate sample, defaults to None.
    """

    def __init__(self, shape: List = None, dtype: Type = None, random_state: Any = None) -> None:
        self._shape = None if shape is None else np.array(shape)
        self._dtype = None if dtype is None else np.dtype(dtype)
        if random_state is None:
            self._random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self._random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        else:
            raise TypeError(f'Unknown random state type - {repr(random_state)}.')

    @abstractmethod
    def sample(self) -> Any:
        r"""
        Get a random sample from the space.

        :return Any: The random sample.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_data(sample: Any) -> np.ndarray:
        r"""
        Convert a sample into data array.

        :param Any sample: The sample to be convert.
        :return np.ndarray: The converted data array.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_sample(data: np.ndarray) -> Any:
        r"""
        Convert data into original sample.

        :param np.ndarray data: The data array to be convert.
        :return Any: The original sample.
        """

    @abstractmethod
    def create_empty(self) -> np.ndarray:
        r"""
        Create an empty space for the initialization of data pool

        :return np.ndarray: The empty array.
        """
        raise NotImplementedError

    @property
    def shape(self) -> np.ndarray:
        r"""
        Shape array of the space. It should be equal to the `shape` of a sample in space.
        """
        return self._shape

    @property
    def dtype(self) -> Type:
        r"""
        Data type of the samples in searching space.
        """
        return self._dtype


class DiscreteSpace(BaseSpace):
    """
    Discrete search space. It is configured by a input value or a list.
    The element of input at each position determinates the discrete
    space for each dim.

    :param List shape: List of space shape, defaults to None.
    :param Type dtype: Data type of samples, defaults to np.int64.
    :param Any random_state: Random state of sampler, defaults to None.

    :Examples:

    >>> space = DiscreteSpace((2, 3, 4))
    >>> space.shape
    array([3, ])
    >>> x = space.sample()
    >>> x
    array([1, 1, 0])
    """

    def __init__(self, shape: List = None, dtype: Type = np.int64, random_state: Any = None) -> None:
        assert (np.array(shape) > 0).all(), "shape must be positive"
        if np.isscalar(shape):
            shape = [shape]
        self._nshape = np.asarray(shape, dtype=dtype)
        shape = self._nshape.shape

        super().__init__(shape, dtype, random_state)

    def sample(self) -> np.ndarray:
        sample = []
        for n in self._nshape:
            sample.append(self._random_state.randint(n))
        sample = np.array(sample).astype(self._dtype)
        return sample

    def sample_single(self, position: int) -> int:
        """
        Get a random value at the position of the discrete space.

        :param int position: The position in the space to sample.
        :return int: The random value at position.
        """
        assert position < len(self._nshape)
        return self._random_state.randint(self._nshape[position])

    def convert_to_data(self, sample: Any, flatten: bool = False) -> np.ndarray:
        return np.asarray(sample, dtype=self._dtype)

    def convert_to_sample(self, data: np.ndarray) -> np.ndarray:
        return data

    def create_empty(self) -> np.ndarray:
        return np.empty(shape=(0, *self._shape), dtype=self._dtype)

    def get_log_title(self, max_cols: int = 2) -> Dict:
        if len(self) > max_cols:
            return [str(num) for num in range(max_cols)]
        else:
            return [str(num) for num in range(len(self))]

    def get_log_data(self, data: np.ndarray, max_cols: int = 2) -> np.ndarray:
        if len(self) > max_cols:
            return data[:max_cols]
        else:
            return data

    def __len__(self) -> int:
        return self._nshape.size

    @property
    def nshape(self) -> np.ndarray:
        """
        List of the space shape.
        """
        return self._nshape


class ContinuousSpace(BaseSpace):
    """
    Continuous search space. The space is defined by its shape and lower/higher bounds, with continuous value.
    Users can set the lower and higher boundary of each dimension of the space, otherwise the value will be
    infinite at these dimensions. Each dimension can have different boundary types and values. If so, the shape
    of boundary argument must be the same as space's shape.

    :param List shape: Shape of the space.
    :param float low: The lower bound of space, defaults to None.
    :param float high: The higher bound of space, defaults to None.
    :param Type dtype: Data type of space, defaults to np.float32.
    :param Any random_state: Random space of sampler, defaults to None.

    :Example:

    >>> space = ContinuousSpace(shape=(2, 3), low=0, high=((5, 5, 5), (8, 8, 8)))
    >>> space.shape
    array([2, 3])
    >>> space.sample()
    array([[2.7791994, 4.216622 , 3.0280395],
       [4.2073464, 7.35141  , 2.5952444]], dtype=float32)
    """

    def __init__(
            self,
            shape: List,
            low: float = None,
            high: float = None,
            dtype: Type = np.float32,
            random_state: Any = None
    ) -> None:
        assert dtype is not None, "dtype must be explicitly provided."
        if np.isscalar(shape):
            shape = [shape]

        super().__init__(shape, dtype, random_state)

        if low is not None:
            assert (np.isscalar(low) or low.shape == self._shape), \
                "low.shape doesn't match provided shape"
            if np.isscalar(low):
                low = np.full(self._shape, low)
        else:
            low = np.full(self._shape, -np.inf)
        if high is not None:
            assert (np.isscalar(high) or high.shape == self._shape), \
                "low.shape doesn't match provided shape"
            if np.isscalar(high):
                high = np.full(self._shape, high)
        else:
            high = np.full(self._shape, np.inf)

        self._low = low.astype(dtype)
        self._high = high.astype(dtype)

        self._bounded_below = -np.inf < self._low
        self._bounded_above = np.inf > self._high

    def sample(self) -> np.ndarray:
        r"""
        Get a random sample from the space. unbounded dimension will be sampled from a normal distribution,
        dimensions with one bound will be sampled from a exponential distribution. Others will be sampled
        from a exponential distribution.

        :return np.ndarray: The random sample.
        """
        sample = np.empty(self._shape)

        unbounded = ~self._bounded_below & ~self._bounded_above
        upp_bounded = ~self._bounded_below & self._bounded_above
        low_bounded = self._bounded_below & ~self._bounded_above
        bounded = self._bounded_below & self._bounded_above

        sample[unbounded] = np.random.normal(size=unbounded[unbounded].shape)
        sample[low_bounded] = (np.random.exponential(size=low_bounded[low_bounded].shape) + self._low[low_bounded])

        sample[upp_bounded] = (-np.random.exponential(size=upp_bounded[upp_bounded].shape) + self._high[upp_bounded])

        sample[bounded] = np.random.uniform(
            low=self._low[bounded], high=self._high[bounded], size=bounded[bounded].shape
        )

        return sample.astype(self._dtype)

    def convert_to_data(self, sample: Any, flatten: bool = False) -> np.ndarray:
        data = np.asarray(sample)
        if flatten:
            data = data.flatten()
        return data

    def convert_to_sample(self, data: np.ndarray) -> np.ndarray:
        sample = data.reshape(self.shape)
        return sample

    def create_empty(self) -> np.ndarray:
        return np.empty(shape=(0, *self._shape), dtype=self._dtype)

    def __len__(self) -> int:
        r"""
        Return the product of dim in shape.
        """
        return np.prod(self._shape)

    @property
    def nshape(self) -> np.ndarray:
        return self._shape

    @property
    def bounds(self) -> np.ndarray:
        return np.array(list(zip(self._low, self._high)))


class TupleSpace(BaseSpace):
    r"""
    Space tuple consists of one or more kinds of space.

    :param List space_list: The provided space list.

    :Example:

    >>> s1 = DiscreteSpace(3)
    >>> s2 = ContinuousSpace(2)
    >>> space = HybridSpace(s1, s2)
    >>> space.shape
    [array([1]), array([2])]
    >>> space.sample()
    [array(1), array([-0.35141  , 1.5952444], dtype=float32)]
    """

    def __init__(self, *space_list: List) -> None:
        self._spaces = []
        for space in space_list:
            self._spaces.append(space)

    def sample(self) -> np.ndarray:
        r"""
        Get am array of random samples from each space.

        :return np.ndarray: Sample array.
        """
        sample = np.zeros(shape=(len(self._spaces)), dtype=object)
        for i, space in enumerate(self._spaces):
            sample[i] = space.sample()
        return np.asarray(sample)

    def convert_to_data(
            self,
            sample: Any,
    ) -> np.ndarray:
        data = np.zeros(shape=(len(self._spaces)), dtype=object)
        for i, space in enumerate(self._spaces):
            data[i] = space.convert_to_data(sample[i])
        return np.asarray(data)

    def convert_to_sample(self, data: np.ndarray) -> np.ndarray:
        sample = np.zeros(shape=(len(self._spaces)), dtype=object)
        for i in range(len(self._spaces)):
            sample[i] = self._spaces[i].convert_to_sample(data[i])
        return np.asarray(sample)

    def create_empty(self) -> np.ndarray:
        return np.empty(shape=(0, len(self._spaces)), dtype=object)

    def __len__(self) -> int:
        r"""
        Get the num of spaces.
        """
        return len(self._spaces)

    @property
    def nshape(self) -> List:
        r"""
        Get a shape list of each space.
        """
        return [space.nshape for space in self._spaces]

    @property
    def shape(self) -> List:
        r"""
        Get a shape list of each space.
        """
        return [space.shape for space in self._spaces]


class DictSpace(BaseSpace):

    def __init__(self, **space_dict: Dict) -> None:
        self._keys = sorted(space_dict.keys())
        self._spaces = [space_dict[key] for key in self._keys]

    def sample(self) -> Dict:
        r"""
        Get a dict of random samples from each space.

        :return Dict: Sample dict.
        """
        return {k: self._spaces[i].sample() for i, k in enumerate(self._keys)}

    def convert_to_data(self, sample: Dict) -> np.ndarray:
        r"""
        Convert a sample dict into data array.

        :param Dict sample: The sample to be convert.
        :return np.ndarray: The converted data array.
        """
        assert set(sample) == set(self._keys)
        data = np.zeros(shape=(len(self._keys)), dtype=object)
        for i, key in enumerate(self._keys):
            data[i] = self._spaces[i].convert_to_data(sample[key])
        return np.asarray(data)

    def convert_to_sample(self, data: np.ndarray) -> Dict:
        r"""
        Convert data into sample dict.

        :param np.ndarray data: The data array to be convert.
        :return Dict: The original sample.
        """
        assert len(data) == len(self._keys)
        return {key: self._spaces[i].convert_to_sample(data[i]) for i, key in enumerate(self._keys)}

    def create_empty(self) -> np.ndarray:
        return np.empty(shape=(0, len(self._spaces)), dtype=object)

    def __len__(self) -> int:
        r"""
        Get the num of spaces.
        """
        return len(self._keys)

    @property
    def nshape(self) -> Dict:
        return {k: self._spaces[i].nshape for i, k in enumerate(self._keys)}

    @property
    def shape(self) -> Dict:
        return {k: self._spaces[i].shape for i, k in enumerate(self._keys)}
