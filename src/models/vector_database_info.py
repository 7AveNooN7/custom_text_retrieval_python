from __future__ import annotations
import json
import os
import pickle
import sqlite3
import time
from typing import List, Dict, Any, Tuple, Optional
import chromadb
import lancedb
import numpy as np
from chromadb.api.types import IncludeEnum
from lancedb.rerankers import RRFReranker
from overrides import overrides
from tqdm import tqdm
import pyarrow as pa
import pandas as pd


from src.enums.embedding_type_enum import EmbeddingType
from src.enums.floating_precision_enum import FloatPrecisionPointEnum
from src.enums.overlap_type import OverlapTypeEnum
from src.enums.text_segmentation_type_enum import TextSegmentationTypeEnum
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.chunk_metadata_model import ChunkMetadataModel
from abc import ABC, abstractmethod

class VectorDatabaseInfo:
    def __init__(
            self, *,
            database_name: str,
            embedding_model_name: str,
            embedding_types: List[EmbeddingType],
            float_precision: FloatPrecisionPointEnum,
            segmentation_type: TextSegmentationTypeEnum,
            preserve_whole_sentences: bool,
            exceed_limit: bool,
            overlap_type: OverlapTypeEnum,
            chunk_size: int,
            chunk_overlap: int,
            files_paths: List[str],
            transformer_library: TransformerLibrary,
            features: Dict[str, dict] = None
    ):
        self.database_name: str = database_name
        self.embedding_model_name: str = embedding_model_name
        self.segmentation_type: TextSegmentationTypeEnum = segmentation_type
        self.preserve_whole_sentences: bool = preserve_whole_sentences
        self.exceed_limit: bool = exceed_limit
        self.overlap_type: OverlapTypeEnum = overlap_type
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.files_paths: List[str] = files_paths
        self.embedding_types: List[EmbeddingType] = embedding_types
        self.float_precision: FloatPrecisionPointEnum = float_precision
        self.transformer_library: TransformerLibrary = transformer_library
        self.features: Dict[str, dict] = features if features is not None else {}

    string_separator = '|'

    def to_dict(self) -> Dict:
        """Konwertuje obiekt do słownika (dict) i zamienia Enum na stringi dla JSON."""
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "segmentation_type": self.segmentation_type.value,
            "preserve_whole_sentences": json.dumps(self.preserve_whole_sentences),
            "exceed_limit": json.dumps(self.exceed_limit),
            "overlap_type": self.overlap_type.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.files_paths,
            "embedding_types": [et.value for et in self.embedding_types], # Enum -> string
            "float_precision": self.float_precision.value,
            "transformer_library": self.transformer_library.display_name,
            "features": self.features
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Tworzy obiekt klasy na podstawie słownika (dict) i zamienia stringi na Enum."""
        return cls(
            database_name=data.get("database_name", "N/A"),
            embedding_model_name=data.get("embedding_model_name", "N/A"),
            segmentation_type=TextSegmentationTypeEnum(data.get("segmentation_type")),
            preserve_whole_sentences=json.loads(data.get("preserve_whole_sentences")),
            exceed_limit=json.loads(data.get("exceed_limit")),
            overlap_type=OverlapTypeEnum(data.get("overlap_type")),
            chunk_size=data.get("chunk_size", 0),
            chunk_overlap=data.get("chunk_overlap", 0),
            files_paths=data.get("files_paths", []),
            embedding_types=[EmbeddingType(et) for et in data.get("embedding_types", [])],  # String -> Enum
            float_precision=FloatPrecisionPointEnum(data.get("float_precision")),
            transformer_library=TransformerLibrary.from_display_name(data.get("transformer_library")),
            features=data.get("features")
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

    def perform_search(self, query, top_k, vector_choices, features_choices):
        pass

    def retrieve_from_database(self):
        pass

    def create_new_database(self, text_chunks, chunks_metadata, embeddings):
        pass


class ChromaVectorDatabase(VectorDatabaseInfo):

    @classmethod
    def get_saved_databases_from_drive_as_instances(cls) -> dict[str, ChromaVectorDatabase]:
        saved_databases: dict[str, ChromaVectorDatabase] = {}
        database_folder = cls.get_database_type().db_folder
        for db_folder_name in os.listdir(database_folder):
            db_path = os.path.join(database_folder, db_folder_name)
            if not os.path.isdir(db_path):
                continue
            chroma_client = chromadb.PersistentClient(path=db_path)

            collection = chroma_client.get_or_create_collection(name=db_folder_name)
            metadata = collection.metadata or {}

            chroma_vector_instance = cls.from_specific_database_metadata(metadata=metadata)
            database_name = chroma_vector_instance.database_name
            saved_databases[database_name] = chroma_vector_instance

            del chroma_client  # Zamknięcie klienta po każdej iteracji

        return saved_databases

    def create_metadata_specific_for_database(self) -> dict:
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "segmentation_type": self.segmentation_type.value,
            "preserve_whole_sentences": json.dumps(self.preserve_whole_sentences),
            "exceed_limit": json.dumps(self.exceed_limit),
            "overlap_type": self.overlap_type.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.string_separator.join(self.files_paths),
            "embedding_types": self.string_separator.join(et.value for et in self.embedding_types),  # Enum -> string
            "float_precision": self.float_precision.value,
            "transformer_library": self.transformer_library.display_name,
            "features": json.dumps(self.features)
        }

    @classmethod
    def from_specific_database_metadata(cls, *, metadata: Dict) -> ChromaVectorDatabase:
        return cls(
            database_name=metadata.get("database_name", "N/A"),
            embedding_model_name=metadata.get("embedding_model_name", "N/A"),
            segmentation_type=TextSegmentationTypeEnum(metadata.get("segmentation_type")),
            preserve_whole_sentences=json.loads(metadata.get("preserve_whole_sentences")),
            exceed_limit=json.loads(metadata.get("exceed_limit")),
            overlap_type=OverlapTypeEnum(metadata.get("overlap_type")),
            chunk_size=metadata.get("chunk_size", 0),
            chunk_overlap=metadata.get("chunk_overlap", 0),
            files_paths=metadata.get("files_paths", "N/A").split(cls.string_separator),
            embedding_types=[EmbeddingType(et.strip()) for et in metadata.get("embedding_types", "N/A").split(cls.string_separator) if et],
            float_precision=FloatPrecisionPointEnum(metadata.get("float_precision")),
            transformer_library=TransformerLibrary.from_display_name(metadata.get("transformer_library", "N/A")),
            features=json.loads(metadata.get("features"))
        )

    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[ChunkMetadataModel], embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]):
        # CHROMA NIE OBSŁUGUJE LIST W METADATA WIEC TRZEBA ZAMIENIC LISTY NA STRINGI
        print('ChromaDB: Create Database')
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
            for i in tqdm(range(len(text_chunks)), desc="📥 ChromaDB: Tworzenie bazy danych"):
                collection.add(
                    ids=[chunks_metadata[i].hash_id],
                    embeddings=[embeddings[0][i]], # MOZE PRZYJAC TYLKO DENSE
                    documents=[text_chunks[i]],
                    metadatas=[chunks_metadata[i].to_dict()]
                )

        del chroma_client

    def perform_search(self, *, query: str, top_k: int, vector_choices: List[str], features_choices: List[str]):
        print('ChromaDB: Search')
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)

        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=self.database_name)

        transformer_library = self.transformer_library
        query_embedding: np.ndarray = transformer_library.generate_embeddings([query], self)[0] # TYLKO DENSE INDEX 0

        results = collection.query(query_embeddings=query_embedding, n_results=top_k)

        result_text: List[str] = results[IncludeEnum.documents.value][0]
        result_chunks_metadata: List[ChunkMetadataModel] = [
            ChunkMetadataModel.from_dict(meta_dict) for meta_dict in results[IncludeEnum.metadatas.value][0]
        ]
        result_scores: List[float] = results[IncludeEnum.distances.value][0]


        return result_text, result_chunks_metadata, result_scores

    def retrieve_from_database(self) -> tuple[List[str], List[ChunkMetadataModel], tuple[np.ndarray, List, List]]:
        print('ChromaDB: Retrieve')
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
        text_chunks = results[IncludeEnum.documents.value]
        chunks_metadata = results[IncludeEnum.metadatas.value]
        chunks_metadata_models: List[ChunkMetadataModel] = [ChunkMetadataModel.from_dict(meta) for meta in chunks_metadata]
        dense_embeddings = results[IncludeEnum.embeddings.value]
        dense_embeddings = np.array(dense_embeddings, dtype=self.float_precision.numpy_dtype)

        del chroma_client

        return text_chunks, chunks_metadata_models, (dense_embeddings, [], [])


class LanceVectorDatabase(VectorDatabaseInfo):
    TEXT_COLUMN = "text"
    METADATA_COLUMN = "metadata"

    dtype_pa_map: dict[FloatPrecisionPointEnum, any] = {
        FloatPrecisionPointEnum.FP16: pa.float16(),
        FloatPrecisionPointEnum.FP32: pa.float32(),
    }

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
                continue
            except json.JSONDecodeError:
                print("Błąd: Plik nie jest poprawnym JSON-em.")
                continue

            lance_vector_instance = cls.from_dict(data=metadata_dict)
            database_name = lance_vector_instance.database_name
            saved_databases[database_name] = lance_vector_instance


        return saved_databases


    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[ChunkMetadataModel], embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]):
        print('LanceDB: Create Database')
        def wait_for_index(target_table, index_name):
            poll_interval = 10
            while True:
                indices = target_table.list_indices()
                # print(f'indices: {indices}')
                # print(f'stats: {target_table.index_stats(index_name)}')

                if indices and any(index.name == index_name for index in indices):
                    break
                #print(f"⏳ Waiting for {index_name} to be ready...")
                time.sleep(poll_interval)

            #print(f"✅ {index_name} is ready!")

        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)
        os.makedirs(db_path, exist_ok=True)
        lance_db = lancedb.connect(db_path)

        all_records = []

        for i in tqdm(range(len(text_chunks)), desc="📥 LanceDB: Tworzenie bazy danych"):
            all_records.append({
                EmbeddingType.DENSE.value: embeddings[0][i],
                self.TEXT_COLUMN: text_chunks[i],
                self.METADATA_COLUMN: json.dumps(chunks_metadata[i].to_dict())
            })


        # Sprawdzenie pierwszego rekordu, aby automatycznie określić schemat
        first_record = all_records[0]


        # Automatyczna definicja schema, TO ZOSTAJE self.dtype_pa_map[self.float_precision] bo trzeba typ pa a nie np
        schema_fields = [
            pa.field(EmbeddingType.DENSE.value, pa.list_(self.dtype_pa_map[self.float_precision], len(first_record[EmbeddingType.DENSE.value]))),
            pa.field(self.TEXT_COLUMN, pa.string()),
            pa.field(self.METADATA_COLUMN, pa.string())
        ]

        # Tworzenie schema na podstawie dynamicznych pól
        schema = pa.schema(schema_fields)

        df = pd.DataFrame(all_records)

        database_name = self.database_name

        if database_name in lance_db.table_names():
            lance_db.drop_table(database_name)

        table = lance_db.create_table(database_name, data=df, schema=schema)

        from src.enums.database_type_enum import DatabaseFeature
        if DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in self.features:
            if self.features[DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value]['use_tantivy']:
                table.create_fts_index(self.TEXT_COLUMN, use_tantivy=True, tokenizer_name="en_stem")
            else:
                table.create_fts_index(self.TEXT_COLUMN, use_tantivy=False, tokenizer_name="en_stem")
                wait_for_index(table, f"{self.TEXT_COLUMN}_idx")

        metadata_string = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        with open(os.path.join(db_path, f"{self.METADATA_COLUMN}.json"), "w", encoding="utf-8") as f:
            f.write(metadata_string)

        del lance_db

    @overrides
    def retrieve_from_database(self) -> tuple[List[str], List[ChunkMetadataModel], tuple[np.ndarray, List[ChunkMetadataModel], List]]:
        print('LanceDB: Retrieve')
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)

        # Odczytanie danych z tabeli jako DataFrame
        df = table.to_lance().to_table().to_pandas() # BUG W LANCE DB ZE TYLKO 10 REKORDOW ZWRACA

        # Rozpakowanie danych do zmiennych
        text_chunks: List[str] = df[self.TEXT_COLUMN].tolist()
        chunks_metadata: List[ChunkMetadataModel] = [
            ChunkMetadataModel.from_dict(json.loads(row)) for row in df[self.METADATA_COLUMN]
        ]
        dense_embeddings = np.stack(df[EmbeddingType.DENSE.value].tolist(), axis=0).astype(self.float_precision.numpy_dtype)

        del lance_db

        return text_chunks, chunks_metadata, (dense_embeddings, [], [])

    def perform_search(self, *, query: str, top_k: int, vector_choices: List[str], features_choices: List[str]):
        print('LanceDB Search')
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)

        transformer_library = self.transformer_library
        query_embedding: np.ndarray = transformer_library.generate_embeddings([query], self)[0]  # TYLKO DENSE INDEX 0

        from src.enums.database_type_enum import DatabaseFeature
        if EmbeddingType.DENSE.value in vector_choices and DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in features_choices:
            print('LanceDB HYBRID SEARCH')
            reranker = RRFReranker()
            results = (
                table.search(vector_column_name=EmbeddingType.DENSE.value, query_type="hybrid", fts_columns=self.TEXT_COLUMN)
                .vector(query_embedding)
                .text(query)
                .limit(top_k)
                .select([self.TEXT_COLUMN, self.METADATA_COLUMN])
                .rerank(reranker)
                .to_pandas()
            )
        elif EmbeddingType.DENSE.value in vector_choices:
            print('LanceDB VECTOR SEARCH')
            results = (
                table.search(query=query_embedding, vector_column_name=EmbeddingType.DENSE.value, query_type="vector")
                .limit(top_k)
                .select([self.TEXT_COLUMN, self.METADATA_COLUMN])
                .to_pandas()
            )
        elif DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in features_choices:
            print('LanceDB FULL TEXT SEARCH')
            results = (
                table.search(query=query, query_type="fts", fts_columns=self.TEXT_COLUMN)
                .limit(top_k)
                .select([self.TEXT_COLUMN, self.METADATA_COLUMN])
                .to_pandas()
            )


        # if "_relevance_score" in results.columns:
        #     results = results.sort_values("_relevance_score", ascending=False)
        # elif "_distance" in results.columns:
        #     results = results.sort_values("_distance", ascending=True)
        # elif "_score" in results.columns:
        #     results = results.sort_values("_score", ascending=False)

        result_text: List[str] = results[self.TEXT_COLUMN].tolist()
        result_chunks_metadata: List[ChunkMetadataModel] =  [
            ChunkMetadataModel.from_dict(json.loads(row)) for row in results[self.METADATA_COLUMN].tolist()
        ]

        score_columns = ["_relevance_score", "_distance", "_score"]
        for col in score_columns:
            if col in results.columns:
                result_scores = results[col].tolist()
                break
        else:
            result_scores = [-99] * len(result_text)


        return result_text, result_chunks_metadata, result_scores

class SqliteVectorDatabase(VectorDatabaseInfo):
    TEXT_COLUMN = "text"
    METADATA_COLUMN = "metadata"

    @classmethod
    def get_saved_databases_from_drive_as_instances(cls) -> Dict[str, Any]:
        saved_databases = {}
        database_folder = cls.get_database_type().db_folder

        # Przeszukiwanie folderu w poszukiwaniu plików .db
        for db_file_name in os.listdir(database_folder):
            if not db_file_name.endswith(".db"):
                continue  # Pomijamy pliki, które nie są bazami SQLite

            db_path = os.path.join(database_folder, db_file_name)
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    # Wczytanie metadanych z tabeli metadata
                    cursor.execute("SELECT metadata_json FROM metadata LIMIT 1")
                    metadata_row = cursor.fetchone()
                    if metadata_row is None:
                        print(f"Brak metadanych w bazie: {db_file_name}")
                        continue

                    metadata_json = metadata_row[0]
                    metadata_dict = json.loads(metadata_json)

            except sqlite3.Error as e:
                print(f"Błąd połączenia z bazą {db_file_name}: {e}")
                continue
            except json.JSONDecodeError:
                print(f"Błąd: Metadane w {db_file_name} nie są poprawnym JSON-em.")
                continue

            # Tworzenie instancji na podstawie metadanych
            sqlite_vector_instance = cls.from_dict(data=metadata_dict)
            database_name = sqlite_vector_instance.database_name
            saved_databases[database_name] = sqlite_vector_instance

        return saved_databases

    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[ChunkMetadataModel], embeddings: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]):

        database_folder = self.get_database_type().db_folder
        os.makedirs(database_folder, exist_ok=True)
        db_path = os.path.join(database_folder, f"{self.database_name}.db")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Tworzenie tabeli dla embeddingów
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.database_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {self.TEXT_COLUMN} TEXT,
                    {self.METADATA_COLUMN} TEXT,
                    {EmbeddingType.DENSE.value} BLOB,
                    {EmbeddingType.SPARSE.value} BLOB,
                    {EmbeddingType.COLBERT.value} BLOB,
                    {EmbeddingType.COLBERT.value}_shape TEXT
                )
            ''')

            # Tworzenie tabeli dla metadanych
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metadata_json TEXT
                )
            ''')

            # Wstawianie embeddingów
            for i in tqdm(range(len(text_chunks)), desc="📥 SQLite: Tworzenie bazy danych"):
                # Dense
                dense_blob = None
                if embeddings[0] is not None and embeddings[0].size > 0 and i < len(embeddings[0]):
                    dense_blob = embeddings[0][i].tobytes()

                # Sparse
                sparse_blob = None
                if embeddings[1] is not None and i < len(embeddings[1]):
                    sparse_blob = pickle.dumps({k: self.float_precision.numpy_dtype(v) for k, v in embeddings[1][i].items()})

                # ColBERT
                colbert_blob = None
                colbert_shape = None
                if embeddings[2] is not None and i < len(embeddings[2]):
                    colbert_blob = embeddings[2][i].astype(self.float_precision.numpy_dtype).tobytes()
                    colbert_shape = json.dumps(embeddings[2][i].shape)


                cursor.execute(f'''
                    INSERT INTO {self.database_name} (
                    {self.TEXT_COLUMN}, 
                    {self.METADATA_COLUMN},
                    {EmbeddingType.DENSE.value}, 
                    {EmbeddingType.SPARSE.value}, 
                    {EmbeddingType.COLBERT.value}, 
                    {EmbeddingType.COLBERT.value}_shape
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (text_chunks[i], json.dumps(chunks_metadata[i].to_dict()), dense_blob, sparse_blob, colbert_blob, colbert_shape))

            # Wstawianie metadanych po pętli
            metadata_json = json.dumps(self.to_dict())
            cursor.execute('INSERT INTO metadata (metadata_json) VALUES (?)', (metadata_json,))

            # Automatyczny commit przy wyjściu z bloku 'with'

    def retrieve_from_database(self) -> Tuple[
        List[str], List[ChunkMetadataModel], Tuple[np.ndarray, List[dict[str, float]], List[np.ndarray]]]:
        db_path = os.path.join(self.get_database_type().db_folder, f'{self.database_name}.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            embedding_types_set = set(self.embedding_types)  # Zbiór dla O(1) lookup
            has_dense = EmbeddingType.DENSE in embedding_types_set
            has_sparse = EmbeddingType.SPARSE in embedding_types_set
            has_colbert = EmbeddingType.COLBERT in embedding_types_set

            # Wczytanie embeddingów
            cursor.execute(f'''
                SELECT {self.TEXT_COLUMN}, 
                       {self.METADATA_COLUMN},
                       {EmbeddingType.DENSE.value}, 
                       {EmbeddingType.SPARSE.value}, 
                       {EmbeddingType.COLBERT.value}, 
                       {EmbeddingType.COLBERT.value}_shape 
                FROM {self.database_name}
            ''')
            rows = cursor.fetchall()

            # Inicjalizacja list wynikowych
            text_chunks: List[str] = []
            chunks_metadata: List[ChunkMetadataModel] = []
            dense_vectors_list: Optional[List[np.ndarray]] = [] if has_dense else None
            sparse_embeddings: Optional[List[Dict[str, float]]] = [] if has_sparse else None
            colbert_embeddings: Optional[List[np.ndarray]] = [] if has_colbert else None

            # Przetwarzanie danych z progresbarem
            for row in tqdm(rows, desc="Przetwarzanie wierszy z bazy"):
                # Tekst
                text_chunks.append(row[0])

                # Metadane
                chunks_metadata.append(ChunkMetadataModel.from_dict(json.loads(row[1])))

                # Dense
                if has_dense:
                    dense_vector = np.frombuffer(row[2], dtype=self.float_precision.numpy_dtype)
                    dense_vectors_list.append(dense_vector)

                # Sparse
                if has_sparse:
                    sparse_vector = pickle.loads(row[3])
                    sparse_embeddings.append(sparse_vector)

                # ColBERT
                if has_colbert:
                    colbert_shape_tuple = tuple(json.loads(row[5]))
                    colbert_vector = np.frombuffer(row[4], dtype=self.float_precision.numpy_dtype).reshape(
                        colbert_shape_tuple)
                    colbert_embeddings.append(colbert_vector)

            # Konwersja listy dense_vectors_list na np.ndarray
            dense_embeddings = None
            if has_dense and dense_vectors_list:
                dense_embeddings = np.stack(dense_vectors_list)

            return text_chunks, chunks_metadata, (dense_embeddings, sparse_embeddings, colbert_embeddings)

