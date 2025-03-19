from enum import Enum

import numpy as np
import torch


class FloatPrecisionPointEnum(Enum):
    FP32 = 'FP32'
    FP16 = 'FP16'

    @property
    def numpy_dtype(self):
        dtype_map = {
            FloatPrecisionPointEnum.FP16: np.float16,
            FloatPrecisionPointEnum.FP32: np.float32,
        }
        return dtype_map[self]

    @property
    def torch_dtype(self):
        dtype_map = {
            FloatPrecisionPointEnum.FP16: torch.float16,
            FloatPrecisionPointEnum.FP32: torch.float32,
        }
        return dtype_map[self]