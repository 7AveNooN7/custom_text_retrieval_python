from typing import Tuple, List, Optional

import numpy as np


def embedding_types_checking(embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]):
    # Poprawne sprawdzenie typów, uwzględniając możliwość wartości None
    if embeddings[0] is not None and not isinstance(embeddings[0], np.ndarray):
        raise TypeError(f"Dense embeddings bad type! Expected np.ndarray or None, got {type(embeddings[0])}")
    else:
        print(f'✅ Dense Type: {type(embeddings[0])}')

    if embeddings[1] is not None:
        if not isinstance(embeddings[1], list) or not all(
                isinstance(d, dict) and all(isinstance(k, str) and isinstance(v, float) for k, v in d.items()) for d in
                embeddings[1]):
            raise TypeError(
                f"Sparse embeddings bad type! Expected List[Dict[str, float]] or None, got {type(embeddings[1])}")
    else:
        print(f'✅ Sparse Type: {type(embeddings[1])}')

    if embeddings[2] is not None:
        if not isinstance(embeddings[2], list) or not all(isinstance(arr, np.ndarray) for arr in embeddings[2]):
            raise TypeError(
                f"Colbert embeddings bad type! Expected List[np.ndarray] or None, got {type(embeddings[2])}")
    else:
        print(f'✅ Colbert Type: {type(embeddings[2])}')