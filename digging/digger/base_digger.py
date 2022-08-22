from abc import ABC, abstractmethod, abstractproperty
from easydict import EasyDict
import copy
import numpy as np
from typing import Any, Dict, Tuple

from ding.utils.registry import Registry
from ding.utils.default_helper import deep_merge_dicts


class BaseDigger(ABC):

    config = dict()

    def __init__(self, cfg: Dict, search_space: "BaseSpace", random_state: Any = None) -> None: # noqa
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg

        self._search_space = search_space
        if random_state is None:
            self._random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self._random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        else:
            raise TypeError(f'Unknown random state type - {repr(random_state)}.')

    @abstractmethod
    def reset(self, handler: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def propose(self, sample_num: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update_score(self, samples: np.ndarray, scores: np.ndarray) -> None:
        raise NotImplementedError

    @abstractproperty
    def best(self) -> Tuple[Any, float]:
        raise NotImplementedError

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)


DIGGER_REGISTRY = Registry()
