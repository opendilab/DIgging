import numpy as np
from typing import Any, Dict, Callable, Tuple, List

from .base_digger import BaseDigger
from digging.problem import ProblemEnv
from digging.utils.event import DiggingEvent
from ding.utils import deep_merge_dicts
from ding.worker import BaseLearner, SampleSerialCollector


class RLDigger(BaseDigger):
    """
    Base class of all RL diggers. It defines a standard searching pipeline of RL
    algorithms with provided RL policies and workers i.e. ``collector`` and ``replay_buffer``.
    It uses a problem relevant environment to handle the interactions of RL.

    :param Dict cfg: user config
    :param BaseSpace search_space: searching space of digger
    :param Policy policy: RL policy class in ``ding``
    :param Any replay_buffer: replay buffer of RL. If `None``, the RL will be on-policy,
        defaults to None
    """

    config = dict(
        samples_per_iteration=128,
        max_iterations=100,
        env=dict(),
        policy=dict(collect=dict(collector=dict(), ), learn=dict(learner=dict(), )),
    )

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            policy: "Policy",  # noqa
            replay_buffer: Any = None,
    ) -> None:
        cfg.policy.collect.collector = deep_merge_dicts(
            SampleSerialCollector.default_config(), cfg.policy.collect.collector
        )
        cfg.policy.learn.learner = deep_merge_dicts(BaseLearner.default_config(), cfg.policy.learn.learner)
        super().__init__(cfg, search_space)

        self._policy = policy
        self._collector = None
        self._learner = BaseLearner(cfg.policy.learn.learner, self._policy.learn_mode)
        self._replay_buffer = replay_buffer

        self._max_iterations = self._cfg.max_iterations
        self._samples_per_iteration = self._cfg.samples_per_iteration
        self._start = False
        self._best = None

    def reset(self) -> None:
        r"""
        Reset RL digger by clear the replay buffer and delete collector.
        """
        self.call_event(DiggingEvent.END)
        self._start = False
        self._collector = None
        if self._replay_buffer is not None:
            self._replay_buffer.clear()

    def propose(self, sample_num: int) -> Any:
        r"""
        Propose ``sample_num`` numbers of sample candidates by running ``collector`` to interact with
        the environment.

        :param int sample_num: number of samples, defaults to 1
        :return Any: sample candidates
        """
        assert self._collector is not None, "no collector in digger"
        if not self._start:
            self._start = True
            self.call_event(DiggingEvent.START)
        new_data = self._collector.collect(sample_num, train_iter=self._learner.train_iter)
        return new_data

    def search(self, target_func: Callable) -> Tuple[np.ndarray, int]:
        r"""
        The complete digging pipeline of RL. It will make a problem relevant environment according to target
        function and run the propose-update loops to generate samples and update models. It returns best sample
        together with its score.

        :param Callable target_func: target function
        :return Tuple[np.ndarray, float]: the best sample and score
        """
        env = ProblemEnv(self._cfg.env, self._search_space, target_func)
        env.set_episode_len(self._samples_per_iteration // env.env_num)
        if self._collector is None:
            self._collector = SampleSerialCollector(self._cfg.policy.collect.collector, env, self._policy.collect_mode)
        else:
            self._collector.reset_env(env)

        for i in range(self._max_iterations):
            new_data = self.propose(self._samples_per_iteration)
            self.update_score(new_data)
            self._best = env.best
        return self.provide_best()

    def update_score(self, train_data: List) -> None:
        r"""
        Update a set of samples and scores by running ``learner`` to update the RL model according to config.

        :param List train_data: the new samples.
        """
        if self._replay_buffer is not None:
            self._replay_buffer.push(train_data, cur_collector_envstep=self._collector.envstep)
            for j in range(self._cfg.policy.learn.update_per_collect):
                train_data = self._replay_buffer.sample(self._cfg.policy.learn.batch_size, self._learner.train_iter)
                self._learner.train(train_data, self._collector.envstep)
            if self._learner.policy.get_attribute('priority'):
                self._replay_buffer.update(self._learner.priority_info)
        else:
            self._learner.train(train_data, self._collector.envstep)
        self.call_event(DiggingEvent.STEP)

    def provide_best(self) -> Dict[str, Any]:
        r"""
        Return the current best sample and score stored in data pool.
        """
        return self._best
