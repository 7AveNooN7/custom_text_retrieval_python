import os
from typing import List, Dict

import chromadb
from src.enums.embedding_type_enum import EmbeddingType


class VectorDatabaseInfo:
    def __init__(self, *, database_name: str, embedding_model_name: str, embedding_types: List[EmbeddingType], chunk_size: int, chunk_overlap: int, files_paths: List[str]):
        self.database_name = database_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.files_paths = files_paths
        self.embedding_types=embedding_types

    def to_dict(self) -> Dict:
        """Konwertuje obiekt do słownika (dict) i zamienia Enum na stringi dla JSON."""
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.files_paths,
            "embedding_types": [et.value for et in self.embedding_types]  # Enum -> string
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Tworzy obiekt klasy na podstawie słownika (dict) i zamienia stringi na Enum."""
        return cls(
            database_name=data.get("database_name", "N/A"),
            embedding_model_name=data.get("embedding_model_name", "N/A"),
            chunk_size=data.get("chunk_size", 0),
            chunk_overlap=data.get("chunk_overlap", 0),
            files_paths=data.get("files_paths", []),
            embedding_types=[EmbeddingType(et) for et in data.get("embedding_types", [])],  # String -> Enum
        )

    @property
    def file_count(self) -> int:
        """Getter zwracający liczbę plików w files_paths."""
        return len(self.files_paths)


    @classmethod
    def get_database_type(cls) -> "DatabaseType":
        from src.enums.database_type_enum import DatabaseType
        """Zwraca odpowiedni typ bazy danych na podstawie klasy."""
        for db_type in DatabaseType:
            if db_type.db_class == cls:
                return db_type
        raise ValueError(f"Brak odpowiedniego typu bazy danych dla klasy: {cls.__name__}")


class ChromaVectorDatabase(VectorDatabaseInfo):

    # CHROMA NIE OBSŁUGUJE LIST W METADATA WIEC TRZEBA ZAMIENIC LISTY NA STRINGI
    def create_metadata_specific_for_database(self) -> dict:
        metadata_dict = self.to_dict()
        metadata_dict["files_paths"] = ", ".join(metadata_dict["files_paths"])
        metadata_dict["embedding_types"] = ", ".join(et.value for et in metadata_dict["embedding_types"])
        return metadata_dict

    @classmethod
    def get_saved_databases_from_drive_as_instances(cls) -> dict:
        saved_databases = {}
        database_folder = cls.get_database_type().db_folder
        for db_folder_name in os.listdir(database_folder):
            db_path = os.path.join(database_folder, db_folder_name)
            if not os.path.isdir(db_path):
                continue
            chroma_client = chromadb.PersistentClient(path=db_path)
            try:
                collection = chroma_client.get_or_create_collection(name=db_folder_name)
                metadata = collection.metadata or {}
                chroma_vector_instance = cls.from_specific_database_metadata(metadata=metadata)
                database_name = chroma_vector_instance.database_name
                saved_databases[database_name] = chroma_vector_instance
            finally:
                del chroma_client  # Zamknięcie klienta po każdej iteracji

        return saved_databases

    @classmethod
    def from_specific_database_metadata(cls, *, metadata: Dict):
        return cls(
            database_name=metadata.get("database_name", "N/A"),
            embedding_model_name=metadata.get("embedding_model_name", "N/A"),
            chunk_size=metadata.get("chunk_size", 0),
            chunk_overlap=metadata.get("chunk_overlap", 0),
            files_paths=metadata.get("files_paths", "N/A").split(', '),
            embedding_types=[EmbeddingType(et.strip()) for et in metadata.get("embedding_types", "N/A").split(", ") if et],
        )



class LanceVectorDatabase(VectorDatabaseInfo):
    supported_embeddings = [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT]
    simultaneous_embeddings: int = 3