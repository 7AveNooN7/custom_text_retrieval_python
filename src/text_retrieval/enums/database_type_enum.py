from enum import Enum
from typing import Type, List, Dict

from src.config import CHROMA_DB_FOLDER, LANCE_DB_FOLDER, SQLITE_FOLDER
from src.text_retrieval.enums.embedding_type_enum import EmbeddingType
from src.text_retrieval.models.vector_database_info import ChromaVectorDatabase, LanceVectorDatabase, VectorDatabaseInfo, \
    SqliteVectorDatabase


class DatabaseFeature(Enum):
    LANCEDB_FULL_TEXT_SEARCH = "LanceDB Full Text Search"
    LANCEDB_HYBRID_SEARCH = "LanceDB Hybrid Search"
    HNSW_INDEX = "hnsw_index"
    FILTERING = "filtering"
    MULTI_MODAL = "multi_modal"
    COMPRESSION = "compression"
    RERANKING = "reranking"

class DatabaseType(Enum):
    CHROMA_DB = (
        "ChromaDB",
        CHROMA_DB_FOLDER,
        ChromaVectorDatabase,
        [EmbeddingType.DENSE],
        1,
        True,
        {},
        []
    )
    LANCE_DB = (
        "LanceDB",
        LANCE_DB_FOLDER,
        LanceVectorDatabase,
        [EmbeddingType.DENSE],
        1,
        True,
        {
            DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value: {
                "use_tantivy": True
            }
        },
        [
            [EmbeddingType.DENSE.value, DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value]
        ]
    )
    SQLITE = (
        "SQLite",
        SQLITE_FOLDER,
        SqliteVectorDatabase,
        [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT],
        3,
        False,
        {},
        []
    )


    def __init__(
            self,
            display_name: str,
            db_folder: str,
            db_class: Type["VectorDatabaseInfo"],
            supported_embeddings: List[EmbeddingType],
            simultaneous_embeddings: int,
            database_search: bool,
            features: Dict[str, dict],
            hybrid_search: List[List]
    ):
        self._display_name: str = display_name
        self._db_folder: str = db_folder
        self._db_class: Type["VectorDatabaseInfo"] = db_class
        self._supported_embeddings: List[EmbeddingType] = supported_embeddings
        self._simultaneous_embeddings: int = simultaneous_embeddings
        self._database_search: bool = database_search
        self._features: Dict[str, dict] = features if features is not None else {}
        self._hybrid_search: List[List] = hybrid_search

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def db_folder(self) -> str:
        """Zwraca ścieżkę do folderu bazy danych."""
        return self._db_folder

    @property
    def db_class(self) -> Type["VectorDatabaseInfo"]:
        return self._db_class

    @property
    def supported_embeddings(self) -> List[EmbeddingType]:
        return self._supported_embeddings

    @property
    def simultaneous_embeddings(self) -> int:
        return self._simultaneous_embeddings

    @property
    def database_search(self) -> bool:
        return self._database_search

    @property
    def features(self) -> Dict[str, dict]:
        """Zwraca zbiór obsługiwanych funkcji przez bazę danych."""
        return self._features

    @property
    def hybrid_search(self) -> List[List]:
        """Zwraca zbiór obsługiwanych funkcji przez bazę danych."""
        return self._hybrid_search

    def __str__(self):
        return self.display_name

    @classmethod
    def from_display_name(cls, display_name: str):
        """Konwersja stringa na DatabaseType"""
        for db in cls:
            if db.display_name == display_name:
                return db
        raise ValueError(f"Niepoprawna wartość: {display_name}")