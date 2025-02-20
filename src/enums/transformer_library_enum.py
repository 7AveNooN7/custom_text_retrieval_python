from enum import Enum

class TransformerLibrary(Enum):
    FlagEmbedding = "FlagEmbedding"
    SentenceTransformers = "SentenceTransformers"

    @staticmethod
    def list():
        """Zwraca listę wszystkich wartości enuma"""
        return [e.value for e in TransformerLibrary]