from enum import Enum


class OverlapTypeEnum(Enum):
    DOUBLE = 'Double (Top + Bottom)'
    SLIDING_WINDOW = 'Sliding Window (Top)'