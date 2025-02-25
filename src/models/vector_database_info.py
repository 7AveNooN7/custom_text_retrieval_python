import json
import os
from typing import List, Dict, Any, Tuple

import chromadb
import lancedb
from chromadb.api.types import IncludeEnum
from tqdm import tqdm
import pyarrow as pa
import pandas as pd

from src.config import CHROMA_DB_FOLDER
from src.embedding_model_utils import load_embedding_model
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary


class VectorDatabaseInfo:
    def __init__(self, *, database_name: str, embedding_model_name: str, embedding_types: List[EmbeddingType], chunk_size: int, chunk_overlap: int, files_paths: List[str], transformer_library: TransformerLibrary):
        self.database_name: str = database_name
        self.embedding_model_name: str = embedding_model_name
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.files_paths: List[str] = files_paths
        self.embedding_types: List[EmbeddingType] = embedding_types
        self.transformer_library: TransformerLibrary = transformer_library

    string_separator = '|'

    def to_dict(self) -> Dict:
        """Konwertuje obiekt do słownika (dict) i zamienia Enum na stringi dla JSON."""
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.files_paths,
            "embedding_types": [et.value for et in self.embedding_types], # Enum -> string
            "transformer_library": self.transformer_library.display_name
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
            transformer_library=TransformerLibrary.from_display_name(data.get("transformer_library"))
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
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.string_separator.join(self.files_paths),
            "embedding_types": self.string_separator.join(et.value for et in self.embedding_types),  # Enum -> string
            "transformer_library": self.transformer_library.display_name
        }

    @classmethod
    def get_saved_databases_from_drive_as_instances(cls) -> dict[str, Any]:
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
                # TU JEST PROBLEM
                chroma_vector_instance = cls.from_specific_database_metadata(metadata=metadata)
                database_name = chroma_vector_instance.database_name
                saved_databases[database_name] = chroma_vector_instance
            finally:
                del chroma_client  # Zamknięcie klienta po każdej iteracji

        return saved_databases

    @classmethod
    def from_specific_database_metadata(cls, *, metadata: Dict):
        print(f'tu blad: {metadata.get("embedding_types", "N/A")}')
        return cls(
            database_name=metadata.get("database_name", "N/A"),
            embedding_model_name=metadata.get("embedding_model_name", "N/A"),
            chunk_size=metadata.get("chunk_size", 0),
            chunk_overlap=metadata.get("chunk_overlap", 0),
            files_paths=metadata.get("files_paths", "N/A").split(cls.string_separator),
            embedding_types=[EmbeddingType(et.strip()) for et in metadata.get("embedding_types", "N/A").split(cls.string_separator) if et],
            transformer_library=TransformerLibrary.from_display_name(metadata.get("transformer_library", "N/A"))
        )

    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[dict], hash_id: List[str], embeddings: Tuple[List, List, List]):
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(
            name=self.database_name,
            metadata=self.create_metadata_specific_for_database()
        )

        embeddings_types = self.embedding_types
        embedding_type = embeddings_types[0]
        if embedding_type == EmbeddingType.DENSE:
            index = 0
        elif embedding_type == EmbeddingType.SPARSE:
            index = 1
        elif embedding_type == EmbeddingType.COLBERT:
            index = 2


        if text_chunks:
            for i in tqdm(range(len(text_chunks)), desc="📥 Tworzenie bazy danych"):
                collection.add(
                    ids=[hash_id[i]],
                    embeddings=[embeddings[0][i]], # MOZE PRZYJAC TYLKO DENSE
                    documents=[text_chunks[i]],
                    metadatas=[chunks_metadata[i]]
                )

        del chroma_client

    def perform_search(self, *, query: str, top_k: int):
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)

        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=self.database_name)

        transformer_library = self.transformer_library
        query_embedding: List = transformer_library.generate_embeddings([query], self)[0] # TYLKO DENSE INDEX 0

        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        sorted_results = sorted(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
            key=lambda x: x[2]
        )

        # Budujemy odpowiedź tekstową
        response = ""
        for doc, meta, dist in sorted_results:
            response += (
                f"📄 Plik: {meta['source']} "
                f"(fragment {meta['fragment_id']}, dystans: {dist:.4f}, model: {self.embedding_model_name})\n"
                f"{doc}\n\n"
            )

        return response

    def retrieve_from_database(self) -> tuple[List, List, List, tuple[List, List, List]]:
        # Ścieżka do bazy danych
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)

        # Połączenie z bazą
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=self.database_name)

        # Pobieranie wszystkich danych
        results = collection.get(
            include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.embeddings],
        )

        # Przypisanie danych do zmiennych
        text_chunks = results["documents"]
        chunks_metadata = results["metadatas"]
        hash_id = results["ids"]
        dense_embeddings = results["embeddings"]

        del chroma_client

        return text_chunks, chunks_metadata, hash_id, (dense_embeddings, [], [])


class LanceVectorDatabase(VectorDatabaseInfo):

    @classmethod
    def get_saved_databases_from_drive_as_instances(cls) -> dict[str, Any]:
        saved_databases = {}
        database_folder = cls.get_database_type().db_folder
        for db_folder_name in os.listdir(database_folder):
            db_path = os.path.join(database_folder, db_folder_name)
            metadata_path = os.path.join(db_path, "metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_dict = json.load(f)
            except FileNotFoundError:
                print("Plik metadata.json nie istnieje.")
            except json.JSONDecodeError:
                print("Błąd: Plik nie jest poprawnym JSON-em.")

            lance_vector_instance = cls.from_dict(data=metadata_dict)
            database_name = lance_vector_instance.database_name
            saved_databases[database_name] = lance_vector_instance


        return saved_databases

    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[dict], hash_id: List[str], embeddings: Tuple[List, List, List]):
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)
        os.makedirs(db_path, exist_ok=True)
        lance_db = lancedb.connect(db_path)

        all_records = []

        for i in tqdm(range(len(text_chunks)), desc="📥 Tworzenie bazy danych"):
            all_records.append({
                EmbeddingType.DENSE.value: embeddings[0][i],
                #EmbeddingType.SPARSE.value: embeddings[1][i], LANCEDB NIE WSPIERA SPARSE I COLBERT BEKA
                #EmbeddingType.COLBERT: embeddings[2][i],
                'text': text_chunks[i],
                'source': chunks_metadata[i]['source'],
                "fragment_id": chunks_metadata[i]["fragment_id"],
                'hash_id': hash_id[i]
            })

        # Determine embedding dimension
        embedding_dim = len(all_records[0][EmbeddingType.DENSE.value])
        # Define schema with FixedSizeList for embedding
        schema = pa.schema([
            pa.field(EmbeddingType.DENSE.value, pa.list_(pa.float32(), embedding_dim)),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("fragment_id", pa.int32()),
            pa.field("hash_id", pa.string())
        ])

        df = pd.DataFrame(all_records)

        database_name = self.database_name

        if database_name in lance_db.table_names():
            lance_db.drop_table(database_name)
        lance_db.create_table(database_name, data=df, schema=schema)

        metadata_string = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        with open(os.path.join(db_path, "metadata.json"), "w", encoding="utf-8") as f:
            f.write(metadata_string)

        del lance_db

    def retrieve_from_database(self) -> tuple[List, List, List, tuple[List, List, List]]:
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)

        row_count = table.count_rows()
        print(f'row_count: {row_count}')
        # Odczytanie danych z tabeli jako DataFrame
        df = table.to_lance().to_table().to_pandas() # BUG W LANCE DB ZE TYLKO 10 REKORDOW ZWRACA

        # Rozpakowanie danych do zmiennych
        text_chunks: List = df['text'].tolist()
        chunks_metadata: List = [{'source': row['source'], 'fragment_id': row['fragment_id']} for _, row in df.iterrows()]
        dense_embeddings = [list(embedding) for embedding in df[EmbeddingType.DENSE.value]]
        hash_id = df['hash_id'].tolist()

        del lance_db

        return text_chunks, chunks_metadata, hash_id, (dense_embeddings, [], [])

    def perform_search(self, *, query: str, top_k: int):
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)

        transformer_library = self.transformer_library
        query_embedding: List = transformer_library.generate_embeddings([query], self)[0]  # TYLKO DENSE INDEX 0

        results = (
            table.search(query_embedding, EmbeddingType.DENSE.value)
            .limit(top_k)
            .select(["text", "source", "fragment_id"])
            .to_df()
        )

        print(f'results1: {results}')

        if "score" in results.columns:
            results = results.sort_values("score", ascending=True)
        elif "distance" in results.columns:
            results = results.sort_values("distance", ascending=True)

            # Budujemy odpowiedź
        response = ""
        for idx, row in results.iterrows():
            if "_distance" in row:
                score_val = f"distance: {row['_distance']:.4f}"
            elif "score" in row:
                score_val = f"(score: {row['score']:.4f})"
            else:
                score_val = ""
            response += (
                f"📄 Plik: {row['source']} "
                f"(fragment {row['fragment_id']}, {score_val}, model: {self.embedding_model_name})\n"
                f"{row['text']}\n\n"
            )
        return response

