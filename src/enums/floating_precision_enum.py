from enum import Enum

import numpy as np


class FloatPrecisionPointEnum(Enum):
    FP32 = 'FP32'
    FP16 = 'FP16'

    @property
    def dtype(self):
        dtype_map = {
            FloatPrecisionPointEnum.FP16: np.float16,
            FloatPrecisionPointEnum.FP32: np.float32,
        }
        return dtype_map[self]