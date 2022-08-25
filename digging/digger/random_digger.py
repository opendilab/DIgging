import numpy as np
from typing import Any, Callable, Dict, Tuple

from .base_digger import BaseDigger, DIGGER_REGISTRY
from digging.problem import ProblemHandler
from digging.utils.event import DiggingEvent


@DIGGER_REGISTRY.register('random')
class RandomDigger(BaseDigger):
    r"""
    Random digger of provided searching space. The random candidates comes from the :func:`sample`
    method of the space.

    :param Dict cfg: user config
    :param BaseSpace search_space: searching space of digger
    :param Any random_state: the random state to set the random seed or state of the digger. If the
        value is an integer, it is used as the seed for creating a ``numpy.random.RandomState``.
        Otherwise, the random state provided it is used. When set to None, an unseeded random state
        is generated. Defaults to None
    """
    config = dict(num_sample=100, )

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            random_state: Any = None
    ) -> None:
        super().__init__(cfg, search_space, random_state)
        self._handler = ProblemHandler(search_space)
        self._start = False

    def reset(self) -> None:
        r"""
        Reset the digger by clearing the digging queue and renew the Gaussian Process Regressor.
        """
        self.call_event(DiggingEvent.END)
        self._start = False
        self._handler.clear()

    def search(self, target_func: Callable) -> Tuple[Any, float]:
        self._apply_default_logger()
        self.call_event(DiggingEvent.START)
        self._start = True
        samples = self.propose(self._cfg.num_sample)
        scores = []
        for sample in samples:
            score = target_func(sample)
            scores.append(score)
        scores = np.asarray(scores)
        self.update_score(samples, scores)
        return self.provide_best()

    def propose(self, sample_num: int) -> np.ndarray:
        if not self._start:
            self._start = True
            self.call_event(DiggingEvent.START)
        samples = []
        for _ in range(sample_num):
            samples.append(self._search_space.sample())
        return np.asarray(samples)

    def update_score(self, samples: Any, scores: np.ndarray) -> None:
        r"""
        Update new samples and provided scores into data pool.

        :param Any samples: samples
        :param np.ndarray scores: scores
        """
        self._handler.update_data(samples, scores)
        self.call_event(DiggingEvent.STEP)

    @property
    def latest(self) -> Dict[str, Any]:
        r"""
        Return the latest sample and score updated into data pool.
        """
        all_data = self._handler.get_all_data()
        return {'sample': self._search_space.convert_to_sample(all_data[0][-1]), 'score': all_data[1][-1]}

    def provide_best(self) -> Dict[str, Any]:
        r"""
        Return the current best sample and score stored in data pool.
        """
        return self._handler.best
