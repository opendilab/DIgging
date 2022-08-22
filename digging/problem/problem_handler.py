import numpy as np
from typing import Any, Callable, List, Tuple

from .space import BaseSpace
from .utils import DataPool


class ProblemHandler():
    r"""
    This class is used to provide a uniformed interface for the digger to interact with.
    It can store all the searched samples and scores together with the best ones. Users
    can provide a convert function to convert sample into data array if the sample is not
    in a standard array type.

    :param BaseSpace search_space: Search space of handler.
    :param Callable sample_converter: Convert func to convert samples to data, defaults to None
    """

    def __init__(
            self,
            search_space: BaseSpace,
    ) -> None:
        self._search_space = search_space
        self._data_pool = DataPool(search_space)
        self._best_sample = None
        self._best_score = None

    def _convert_to_data(self, samples: Any) -> np.ndarray:
        data = []
        for sample in samples:
            data.append(self._search_space.convert_to_data(sample))
        data = np.asarray(data)
        return data

    def update_data(self, samples: Any, scores: np.ndarray) -> None:
        r"""
        Update a batch of samples and scores. If not stored already, they will be added into data pool.
        Best score and sample will be stored.

        :param Any samples: Samples to be updated
        :param np.ndarray scores: Scores to be updated
        """
        #assert samples.shape[0] == scores.shape[0], "sample and score must have the same num"
        data_array = self._convert_to_data(samples)
        self._data_pool.update(data_array, scores)
        max_idx = np.argmax(scores)
        if self._best_score is None or scores[max_idx] > self._best_score:
            self._best_sample = samples[max_idx]
            self._best_score = scores[max_idx]

    def get_cached_score(self, samples: Any) -> np.ndarray:
        r"""
        Get scores of a batch of sample from stored scores. All the samples must be searched or updated
        before. Otherwise the score will be `None`.

        :param Any samples: Samples to get scores.
        :return List: Scores of all provided samples stored as `List` for there will be `None` elements.
        """
        data_array = self._convert_to_data(samples)
        return self._data_pool.get_scores(data_array)

    def provide_best(self) -> Tuple[Any, float]:
        r"""
        Get the currently best sample and its score in the data pool.

        :return Tuple[np.ndarray, float]: best sample and its score
        """
        return self._best_sample, self._best_score

    def __len__(self) -> int:
        return len(self._data_pool)

    @property
    def space(self) -> BaseSpace:
        return self._search_space

    @property
    def best_score(self) -> float:
        """
        Get the currently best score only.
        """
        if len(self._data_pool) == 0:
            return -np.inf
        return self._best_score

    def reset(self) -> None:
        """
        Reset the handler. Clear data pool.
        """
        self._data_pool.clear()
        self._best_sample = None
        self._best_score = None
