import numpy as np
from typing import Any

from .base_digger import BaseDigger, DIGGER_REGISTRY
from digging.problem import ProblemHandler


@DIGGER_REGISTRY.register()
class RandomDigger():
    pass
