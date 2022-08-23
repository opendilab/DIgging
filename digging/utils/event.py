from enum import IntEnum, auto
import enum_tools


#@enum_tools.document_enum
class DiggingEvent(IntEnum):
    r"""
    Event during digging.
    """
    START = auto()
    STEP = auto()
    SKIP = auto()
    END = auto()
