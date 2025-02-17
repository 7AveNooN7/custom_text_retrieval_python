from enum import Enum
from typing import Type

from src.models.vector_database_info import ChromaVectorDatabase, LanceVectorDatabase, VectorDatabaseInfo


class DatabaseType(Enum):
    CHROMA_DB = ("ChromaDB", ChromaVectorDatabase)
    LANCE_DB = ("LanceDB", LanceVectorDatabase)

    def __init__(self, display_name: str, db_class: Type[VectorDatabaseInfo]):
        self.display_name = display_name
        self.db_class = db_class  # Powiązana klasa

    def create_instance(self, *args, **kwargs) -> VectorDatabaseInfo:
        """Tworzy instancję odpowiadającej klasy bazy danych"""
        return self.db_class(*args, **kwargs)
