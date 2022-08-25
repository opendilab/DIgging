from typing import Any, Tuple
import numpy as np
import copy

from .space import BaseSpace


def make_hashable(x):
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            x_ = x.copy()
            x = []
            for xi in x_:
                x += make_hashable(xi)
        else:
            x = x.flatten()
    return tuple(map(float, x))


class DataPool():

    def __init__(self, data_space: BaseSpace) -> None:
        self._all_data = data_space.create_empty()
        self._scores = np.empty(shape=(0, ))
        self._data_dict = {}

    def update(self, data_array: np.ndarray, scores: np.ndarray) -> None:
        new_idx = []
        for i in range(data_array.shape[0]):
            data, score = data_array[i], scores[i]
            if make_hashable(data) not in self._data_dict:
                self._data_dict[make_hashable(data)] = score
                new_idx.append(i)
        self._all_data = np.concatenate([self._all_data, data_array[new_idx]], axis=0)
        self._scores = np.concatenate([self._scores, scores[new_idx]], axis=0)
        self._best_idx = np.argmax(self._scores)

    def get_scores(self, data_array: np.ndarray) -> np.ndarray:
        scores = []
        for data in data_array:
            try:
                score = self._data_dict[make_hashable(data)]
            except KeyError:
                print(f"key {data} not found in cache!")
                score = None
            scores.append(score)
        return scores

    def __len__(self) -> int:
        return len(self._data_dict)

    def clear(self) -> None:
        self._all_data = np.empty(shape=(0, *self._all_data.shape[1:]), dtype=self._all_data.dtype)
        self._scores = np.empty(shape=(0, ))
        self._data_dict.clear()

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._all_data, self._scores

    @property
    def best(self) -> Tuple[np.ndarray, float]:
        if self._best_idx is None:
            raise ValueError("sample pool is empty!")
        return self._all_data[self._best_idx], self._scores[self._best_idx]
