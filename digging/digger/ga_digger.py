import numpy as np
import copy
from typing import Any, Callable, Dict, Sequence, Tuple

from .base_digger import BaseDigger, DIGGER_REGISTRY
from digging.problem import ProblemHandler
from digging.utils.event import DiggingEvent


@DIGGER_REGISTRY.register('ga')
class GeneticAlgorithmDigger(BaseDigger):
    r"""
    The Genetic Algorithm digger. The digger is usually used in discrete space.

    :param Dict cfg: user config
    :param BaseSpace search_space: searching space of digger
    :param Any random_state: the random state to set the random seed or state of the digger. If the
        value is an integer, it is used as the seed for creating a ``numpy.random.RandomState``.
        Otherwise, the random state provided it is used. When set to None, an unseeded random state
        is generated. Defaults to None
    """

    config = dict(
        cross_rate=0.1,
        mutate_rate=0.1,
        max_generation=100,
        population_size=100,
    )

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            random_state: Any = None
    ) -> None:
        super().__init__(cfg, search_space, random_state)
        self._cross_rate = self._cfg.cross_rate
        self._mutate_rate = self._cfg.mutate_rate

        self._handler = ProblemHandler(search_space)
        self._first_generation = True

    def reset(self) -> None:
        r"""
        Reset the digger and clear all current generations.
        """
        self.call_event(DiggingEvent.END)
        self._handler.clear()
        self._first_generation = True

    def search(self, target_func: Callable) -> Tuple[np.ndarray, float]:
        r"""
        The complete digging pipeline of GA. The first generation is randomly generated and the following
        generations is proposed by standard Genetic Algorithms. It returns best sample together with its score.

        :param Callable target_func: target function
        :return Tuple[np.ndarray, float]: the best sample and score
        """
        self._apply_default_logger()
        for i in range(self._cfg.max_generation):
            pop = self.propose(self._cfg.population_size)
            scores = np.asarray([target_func(p) for p in pop])
            self.update_score(pop, scores)
        return self.best

    def propose(self, candidate_num: int) -> np.ndarray:
        r"""
        Propose ``sample_num`` numbers of sample candidates by getting a new generation.

        :param int sample_num: number of samples, defaults to 1
        :return np.ndarray: sample candidates
        """
        if self._first_generation:
            self.call_event(DiggingEvent.START)
            samples = [self._search_space.sample() for _ in range(candidate_num)]
            candidates = np.vstack([self._search_space.sample() for _ in range(candidate_num)])
            self._first_generation = False
        else:
            samples, scores = self._handler.get_all_data()
            norm_scores = scores - np.min(scores)
            parents_idx = self._random_state.choice(
                np.arange(len(self._handler)), candidate_num, True, p=norm_scores / np.sum(norm_scores)
            )
            parents = samples[parents_idx]
            children = self._crossover(parents, parents.copy())
            children = self._mutation(children)
            candidates = np.vstack(children)
        return candidates

    def _crossover(self, seqs: np.ndarray, pops: np.ndarray) -> np.ndarray:
        for i in range(len(seqs)):
            if self._random_state.rand() < self._cross_rate:
                j = self._random_state.randint(0, len(seqs), size=1)
                cross_points = self._random_state.randint(0, 2, len(self._search_space)).astype(bool)
                seqs[i, cross_points] = pops[j, cross_points]
        return seqs

    def _mutation(self, seqs: np.ndarray) -> np.ndarray:
        for seq in seqs:
            for point in range(len(self._search_space)):
                if self._random_state.rand() < self._mutate_rate:
                    seq[point] = self._random_state.randint(self._search_space.nshape[point])
        return seqs

    def update_score(self, sequences: np.ndarray, scores: np.ndarray) -> None:
        r"""
        Update a set of samples and scores.

        :param np.ndarray sequences: samples
        :param np.ndarray scores: scores
        """
        self._handler.update_data(sequences, scores)
        self.call_event(DiggingEvent.STEP)

    @property
    def latest(self) -> Dict[str, Any]:
        r"""
        Return the latest sample and score updated into data pool.
        """
        all_data = self._handler.get_all_data()
        return {'sample': self._search_space.convert_to_sample(all_data[0][-1]), 'score': all_data[1][-1]}

    @property
    def best(self) -> Dict[str, Any]:
        r"""
        Return the current best sample and score stored in data pool.
        """
        if self._first_generation:
            return
        return self._handler.provide_best()
