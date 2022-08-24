from abc import ABC, abstractmethod, abstractproperty
from easydict import EasyDict
import copy
import numpy as np
from typing import Any, Dict, Tuple, TypeVar, Callable, Union

from ding.utils.registry import Registry
from ding.utils.default_helper import deep_merge_dicts
from digging.utils.event import DiggingEvent
from digging.utils.logger import get_logger

DIGGER_REGISTRY = Registry()
EventType = TypeVar("EventType")
SubscriberType = TypeVar('SubscriberType')


class BaseDigger(ABC):

    config = dict()

    def __init__(self, cfg: Dict, search_space: "BaseSpace", random_state: Any = None) -> None:  # noqa
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

        self._events: Dict[EventType, Dict[Tuple[str, int], Callable]] = {e: {} for e in DiggingEvent}
        self._subscribers: Dict[Tuple[str, int], SubscriberType] = {}

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

    def subscribe(
            self, event: EventType, subscriber: SubscriberType, callback: Union[Callable, str, None] = None
    ) -> None:
        if callback is None:
            callback = getattr(subscriber, 'update')
        elif isinstance(callback, str):
            callback = getattr(subscriber, callback)
        if not callable(callback):
            raise TypeError(f'Callback should be callable, but {repr(callback)} found.')
        subscriber_id = _get_object_id(subscriber)
        self._events[event][subscriber_id] = callback
        self._subscribers[subscriber_id] = subscriber

    def unsubscribe(self, event: EventType, subscriber: SubscriberType) -> None:
        try:
            del self._events[event][_get_object_id(subscriber)]
        except KeyError:
            raise KeyError(subscriber)

    def apply_logger(self, logger_name: str, *args, **kwargs) -> None:
        _logger = get_logger(logger_name)(*args, **kwargs)
        for event in DiggingEvent.__members__.values():
            self.subscribe(event, _logger)

    def _apply_default_logger(self) -> None:
        if not any([subs for subs in self._events.values()]):
            self.apply_logger('screen', 2)

    def call_event(self, event: EventType) -> None:
        for _, callback in self._events[event].items():
            callback(event, self)

    @property
    def space(self):
        return self._search_space


def _get_object_id(obj) -> Tuple[str, int]:
    try:
        return 'hash', hash(obj)
    except TypeError:
        return 'id', id(obj)
