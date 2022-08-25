import gym
import copy
import numpy as np
from easydict import EasyDict
from typing import Any, Callable, Dict, Optional, Tuple

from .data_pool import DataPool
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import deep_merge_dicts
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register("problem_env")
class ProblemEnv(BaseEnv):

    config = dict(
        episode_len=float("inf"),
        init_type='zero',
        empty_obs=False,
        obs_shape=(1, ),
        batch_num=1,
    )

    reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)

    def __init__(
            self,
            cfg: Dict,
            search_space: "BaseSpace",  # noqa
            target_func: Callable,
    ) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self._empty_obs = self._cfg.empty_obs
        self._obs_shape = self._cfg.obs_shape
        self._search_space = search_space
        self._data_pool = DataPool(search_space)
        self._target_func = target_func
        self._state = None

        self._episode_len = self._cfg.episode_len
        assert self._cfg.init_type in ['zero', 'random'], self._cfg.init_type
        self._batch_num = self._cfg.batch_num
        self._step_count = 0

    @property
    def ready_obs(self) -> Dict:
        r"""
        Interface to get observations of current step.

        :return Dict: observations for each ``env_id``
        """
        obs = {}
        if self._state is None:
            return obs
        for env_id in range(self._batch_num):
            if self._empty_obs:
                obs[env_id] = np.zeros(self._obs_shape, dtype=self._search_space.dtype)
            else:
                obs[env_id] = self._state[env_id]
        return obs

    @property
    def env_num(self) -> int:
        r"""
        The batched num of samples in one step
        """
        return self._batch_num

    def set_episode_len(self, episode_len: int) -> None:
        self._episode_len = min(episode_len, self._episode_len)

    def launch(self) -> None:
        r"""
        The first ``reset`` called by ``ding`` workers.
        """
        self.reset()

    def reset(self, starting_point: Optional[np.ndarray] = None) -> None:
        r"""
        Reset all batch of samples with provided sample or random/zero ones.

        :param Optional[np.ndarray] starting_point: provided starting samples, defaults to None
        """
        self._step_count = 0
        if starting_point is None:
            obs_list = []
            for i in range(self._batch_num):
                if self._cfg.init_type == 'zero':
                    single_obs = np.zeros(shape=self._search_space.shape, dtype=np.float32)
                elif self._cfg.init_type == 'random':
                    single_obs = self._search_space.sample()
                obs_list.append(single_obs)
            self._state = np.array(obs_list, dtype=self._search_space.dtype)
        else:
            self._state = starting_point.astype(self._search_space.dtype)

    def step(self, actions: Dict) -> Dict[int, BaseEnvTimestep]:
        r"""
        Run one step of a batch of actions. It will git scores of all actions from target functions
        and return its scores.

        :param Dict actions: batch of actions
        :return Dict[int, BaseEnvTimestep]: The env timestep of each batch of samples.
        """
        env_ids = actions.keys()
        self._state = np.asarray(list(actions.values())).astype(np.float32)
        scores = np.apply_along_axis(self._target_func, axis=0, arr=self._state)
        self._data_pool.update(self._state, scores)
        self._step_count += 1
        done = self._step_count > self._episode_len
        timesteps = {}
        for env_id in env_ids:
            if self._empty_obs:
                obs = np.zeros(self._obs_shape, dtype=self._search_space.dtype)
            else:
                obs = self._state[env_id]
            reward = scores[env_id]
            info = {'final_eval_reward': scores[env_id]}
            timesteps[env_id] = BaseEnvTimestep(obs, [reward], done, info)
        return timesteps

    def close(self) -> None:
        r"""
        Clear the data pool and close the env.
        """
        self._data_pool.clear()
        return

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        r"""
        Set random seed and dynamic seed

        :param int seed: seed value
        :param bool dynamic_seed: whether to use dynamic seed, defaults to True
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def best(self) -> Tuple[np.ndarray, float]:
        r"""
        Return the current best sample and score.
        """
        return {'sample': self._data_pool.best[0], 'score': self._data_pool.best[1]}

    @property
    def best_score(self) -> float:
        r"""
        Get the currently best score only.
        """
        if len(self._data_pool) == 0:
            return -np.inf
        self._data_pool.best[1]

    def __repr__(self) -> str:
        return "DIgging Problem Env"

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
