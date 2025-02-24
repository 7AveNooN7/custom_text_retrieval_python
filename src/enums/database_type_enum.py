from enum import Enum
from typing import Type, List

from src.config import CHROMA_DB_FOLDER, LANCE_DB_FOLDER
from src.enums.embedding_type_enum import EmbeddingType
from src.models.vector_database_info import ChromaVectorDatabase, LanceVectorDatabase, VectorDatabaseInfo

class DatabaseType(Enum):
    CHROMA_DB = (
        "ChromaDB",
        ChromaVectorDatabase,
        [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT],
        1,
        CHROMA_DB_FOLDER
    )
    LANCE_DB = (
        "LanceDB",
        LanceVectorDatabase,
        [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT],
        3,
        LANCE_DB_FOLDER
    )



    def __init__(self, display_name: str, db_class: Type["VectorDatabaseInfo"], supported_embeddings: List[EmbeddingType], simultaneous_embeddings: int, db_folder: str):
        self._display_name: str = display_name
        self._db_class: Type["VectorDatabaseInfo"] = db_class
        self._supported_embeddings: List[EmbeddingType] = supported_embeddings
        self._simultaneous_embeddings: int = simultaneous_embeddings
        self._db_folder: str = db_folder

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def db_class(self) -> Type["VectorDatabaseInfo"]:
        return self._db_class

    @property
    def storage_supported_embeddings(self) -> List[EmbeddingType]:
        return self._supported_embeddings

    @property
    def simultaneous_embeddings(self) -> int:
        return self._simultaneous_embeddings

    @property
    def db_folder(self) -> str:
        """Zwraca ścieżkę do folderu bazy danych."""
        return self._db_folder

    def __str__(self):
        return self.display_name

    @classmethod
    def from_display_name(cls, display_name: str):
        """Konwersja stringa na DatabaseType"""
        for db in cls:
            if db.display_name == display_name:
                return db
        raise ValueError(f"Niepoprawna wartość: {display_name}")