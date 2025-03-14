from typing import Tuple, List, Optional

import numpy as np

from src.enums.floating_precision_enum import FloatPrecisionPointEnum


def embedding_types_checking(
        embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]],
        float_precision: FloatPrecisionPointEnum
):
    dtype_map: dict[FloatPrecisionPointEnum, any] = {
        FloatPrecisionPointEnum.FP16: np.float16,
        FloatPrecisionPointEnum.FP32: np.float32,
    }
    expected_float_type = dtype_map[float_precision]



    # Poprawne sprawdzenie typów, uwzględniając możliwość wartości None
    if embeddings[0] is not None:
        if not isinstance(embeddings[0], np.ndarray):
            raise TypeError(f"Dense embeddings bad type! Expected np.ndarray or None, got {type(embeddings[0])}")

        # Dodatkowe sprawdzenia
        if embeddings[0].dtype != expected_float_type:
            raise TypeError(f"Dense embeddings bad dtype! Expected {expected_float_type}, got {embeddings[0].dtype}")

        if embeddings[0].ndim != 2:
            raise TypeError(f"Dense embeddings bad shape! Expected 2D array, got {embeddings[0].ndim}D")

        if embeddings[0].shape[0] == 0:
            raise ValueError("Dense embeddings is empty! Expected non-empty array")

        expected_dim = 384  # Dostosuj do swojego modelu
        if embeddings[0].shape[1] != expected_dim:
            raise TypeError(f"Dense embeddings bad dimension! Expected {expected_dim}, got {embeddings[0].shape[1]}")

        print(f'✅ Dense Type: {type(embeddings[0])}, dtype: {embeddings[0].dtype}, shape: {embeddings[0].shape}')
    else:
        print(f'✅ Dense Type: {type(embeddings[0])}')

    if embeddings[1] is not None:
        if not isinstance(embeddings[1], list):
            raise TypeError(
                f"Sparse embeddings bad type! Expected List[Dict[str, float]] or None, got {type(embeddings[1])}")
        for i, d in enumerate(embeddings[1]):
            if not isinstance(d, dict):
                raise TypeError(f"Sparse embeddings bad type! Item {i} is {type(d)}, expected dict")
            for k, v in d.items():
                if not isinstance(k, str):
                    raise TypeError(f"Sparse embeddings bad type! Key {k} in item {i} is {type(k)}, expected str")
                if not isinstance(v, expected_float_type):
                    raise TypeError(f"Sparse embeddings bad type! Value {v} in item {i} is {type(v)}, expected float")
        print(f'✅ Sparse Type: {type(embeddings[1])}')
    else:
        print(f'✅ Sparse Type: {type(embeddings[1])}')

    if embeddings[2] is not None:
        if not isinstance(embeddings[2], list) or not all(isinstance(arr, np.ndarray) for arr in embeddings[2]):
            raise TypeError(
                f"Colbert embeddings bad type! Expected List[np.ndarray] or None, got {type(embeddings[2])}")
    else:
        print(f'✅ Colbert Type: {type(embeddings[2])}')