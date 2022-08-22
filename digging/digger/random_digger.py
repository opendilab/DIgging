import numpy as np
from typing import Any, Callable, Dict, Tuple

from .base_digger import BaseDigger, DIGGER_REGISTRY
from digging.problem import ProblemHandler


@DIGGER_REGISTRY.register('random')
class RandomDigger(BaseDigger):
    config = dict(num_sample=100, )

    def __init__(self, cfg: Dict, search_space: "BaseSpace", random_state: Any = None) -> None:  # noqa
        super().__init__(cfg, search_space, random_state)
        self._handler = ProblemHandler(search_space)

    def reset(self) -> None:
        self._handler.clear()

    def search(self, target_func: Callable) -> Tuple[Any, float]:
        samples = self.propose(self._cfg.num_sample)
        scores = []
        for sample in samples:
            score = target_func(sample)
            scores.append(score)
        scores = np.asarray(scores)
        self.update_score(samples, scores)
        return self.best

    def propose(self, sample_num: int) -> np.ndarray:
        samples = []
        for _ in range(sample_num):
            samples.append(self._search_space.sample())
        return np.asarray(samples)

    def update_score(self, samples: Any, scores: np.ndarray) -> None:
        self._handler.update_data(samples, scores)

    @property
    def best(self) -> Tuple[Any, float]:
        return self._handler.provide_best()
