from enum import Enum

class EmbeddingType(Enum):
    DENSE = "Dense"
    SPARSE = "Sparse"
    COLBERT = "ColBERT"

    @staticmethod
    def list():
        """Zwraca listę wszystkich wartości enuma"""
        return [e.value for e in EmbeddingType]
