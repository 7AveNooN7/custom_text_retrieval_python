import json
import os
import sqlite3
import time
from typing import List, Dict, Any, Tuple
import gradio as gr
import chromadb
import lancedb
import numpy as np
from chromadb.api.types import IncludeEnum
from lancedb.rerankers import RRFReranker
from tqdm import tqdm
import pyarrow as pa
import pandas as pd

from src.config import CHROMA_DB_FOLDER
from src.embedding_model_utils import load_embedding_model
from src.enums.embedding_type_enum import EmbeddingType
from src.enums.transformer_library_enum import TransformerLibrary


class VectorDatabaseInfo:
    def __init__(
            self, *,
            database_name: str,
            embedding_model_name: str,
            embedding_types: List[EmbeddingType],
            chunk_size: int, chunk_overlap: int,
            files_paths: List[str],
            transformer_library: TransformerLibrary,
            features: Dict[str, dict] = None
    ):
        self.database_name: str = database_name
        self.embedding_model_name: str = embedding_model_name
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.files_paths: List[str] = files_paths
        self.embedding_types: List[EmbeddingType] = embedding_types
        self.transformer_library: TransformerLibrary = transformer_library
        self.features: Dict[str, dict] = features if features is not None else {}

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
            "transformer_library": self.transformer_library.display_name,
            "features": self.features
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


class ChromaVectorDatabase(VectorDatabaseInfo):


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

    def create_metadata_specific_for_database(self) -> dict:
        return {
            "database_name": self.database_name,
            "embedding_model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "files_paths": self.string_separator.join(self.files_paths),
            "embedding_types": self.string_separator.join(et.value for et in self.embedding_types),  # Enum -> string
            "transformer_library": self.transformer_library.display_name,
            "features": json.dumps(self.features)
        }

    @classmethod
    def from_specific_database_metadata(cls, *, metadata: Dict):
        return cls(
            database_name=metadata.get("database_name", "N/A"),
            embedding_model_name=metadata.get("embedding_model_name", "N/A"),
            chunk_size=metadata.get("chunk_size", 0),
            chunk_overlap=metadata.get("chunk_overlap", 0),
            files_paths=metadata.get("files_paths", "N/A").split(cls.string_separator),
            embedding_types=[EmbeddingType(et.strip()) for et in metadata.get("embedding_types", "N/A").split(cls.string_separator) if et],
            transformer_library=TransformerLibrary.from_display_name(metadata.get("transformer_library", "N/A")),
            features=json.loads(metadata.get("features"))
        )

    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[dict], hash_ids: List[str], embeddings: Tuple[np.ndarray, List[dict[str, float]], List[np.ndarray]]):
        # CHROMA NIE OBSŁUGUJE LIST W METADATA WIEC TRZEBA ZAMIENIC LISTY NA STRINGI

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
                    ids=[hash_ids[i]],
                    embeddings=[embeddings[0][i]], # MOZE PRZYJAC TYLKO DENSE
                    documents=[text_chunks[i]],
                    metadatas=[chunks_metadata[i]]
                )

        del chroma_client

    def perform_search(self, *, query: str, top_k: int, vector_choices: List[str], features_choices: List[str]):
        database_folder = self.get_database_type().db_folder
        db_path = os.path.join(database_folder, self.database_name)

        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name=self.database_name)

        transformer_library = self.transformer_library
        query_embedding: np.ndarray = transformer_library.generate_embeddings([query], self)[0] # TYLKO DENSE INDEX 0

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
                continue
            except json.JSONDecodeError:
                print("Błąd: Plik nie jest poprawnym JSON-em.")
                continue

            lance_vector_instance = cls.from_dict(data=metadata_dict)
            database_name = lance_vector_instance.database_name
            saved_databases[database_name] = lance_vector_instance


        return saved_databases


    def create_new_database(self, *, text_chunks: List[str], chunks_metadata: List[dict], hash_ids: List[str], embeddings: Tuple[np.ndarray, List[dict[str, float]], List[np.ndarray]]):
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
                'text': text_chunks[i],
                EmbeddingType.DENSE.value: embeddings[0][i],
                'source': chunks_metadata[i]['source'],
                'fragment_id': chunks_metadata[i]["fragment_id"],
                'hash_id': hash_ids[i]
            })

        # Determine embedding dimension
        embedding_dense_dim = len(all_records[0][EmbeddingType.DENSE.value])
        # Define schema with FixedSizeList for embedding
        schema = pa.schema([
            pa.field(EmbeddingType.DENSE.value, pa.list_(pa.float32(), embedding_dense_dim)),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("fragment_id", pa.int32()),
            pa.field("hash_id", pa.string())
        ])

        df = pd.DataFrame(all_records)

        database_name = self.database_name

        if database_name in lance_db.table_names():
            lance_db.drop_table(database_name)

        table = lance_db.create_table(database_name, data=df, schema=schema)

        from src.enums.database_type_enum import DatabaseFeature
        if DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in self.features:
            if self.features[DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value]['use_tantivy']:
                table.create_fts_index("text", use_tantivy=True, tokenizer_name="en_stem")
            else:
                table.create_fts_index("text", use_tantivy=False, tokenizer_name="en_stem")
                wait_for_index(table, f"text_idx")

        metadata_string = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        with open(os.path.join(db_path, "metadata.json"), "w", encoding="utf-8") as f:
            f.write(metadata_string)

        del lance_db

    def retrieve_from_database(self) -> tuple[List, List, List, tuple[List, List, List]]:
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)

        row_count = table.count_rows()
        # Odczytanie danych z tabeli jako DataFrame
        df = table.to_lance().to_table().to_pandas() # BUG W LANCE DB ZE TYLKO 10 REKORDOW ZWRACA

        # Rozpakowanie danych do zmiennych
        text_chunks: List = df['text'].tolist()
        chunks_metadata: List = [{'source': row['source'], 'fragment_id': row['fragment_id']} for _, row in df.iterrows()]
        dense_embeddings = [list(embedding) for embedding in df[EmbeddingType.DENSE.value]]
        hash_id = df['hash_id'].tolist()

        del lance_db

        return text_chunks, chunks_metadata, hash_id, (dense_embeddings, [], [])

    def perform_search(self, *, query: str, top_k: int, vector_choices: List[str], features_choices: List[str]):
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name)
        lance_db = lancedb.connect(db_path)
        table = lance_db.open_table(self.database_name)


        transformer_library = self.transformer_library
        query_embedding: List = transformer_library.generate_embeddings([query], self)[0]  # TYLKO DENSE INDEX 0

        from src.enums.database_type_enum import DatabaseFeature
        #if EmbeddingType.DENSE in
        if EmbeddingType.DENSE.value in vector_choices and DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in features_choices:
            print('HYBRID SEARCH')
            reranker = RRFReranker()
            results = (
                table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(query)
                .limit(top_k)
                .select(["text", "source", "fragment_id"])
                .rerank(reranker)
                .to_df()
            )
        elif EmbeddingType.DENSE.value in vector_choices:
            print('VECTOR SEARCH')
            results = (
                table.search(query_embedding, EmbeddingType.DENSE.value, query_type="vector")
                .limit(top_k)
                .select(["text", "source", "fragment_id"])
                .to_df()
            )
        elif DatabaseFeature.LANCEDB_FULL_TEXT_SEARCH.value in features_choices:
            print('FTS SEARCH')
            results = (
                table.search(query, query_type="fts")
                .limit(top_k)
                .select(["text", "source", "fragment_id"])
                .to_df()
            )


        if "_relevance_score" in results.columns:
            results = results.sort_values("_relevance_score", ascending=False)
        elif "_distance" in results.columns:
            results = results.sort_values("_distance", ascending=True)
        elif "_score" in results.columns:
            results = results.sort_values("_score", ascending=False)

            # Budujemy odpowiedź
        response = ""
        for idx, row in results.iterrows():
            if "_relevance_score" in row:
                score_val = f"relevance score: {row['_relevance_score']:.4f}"
            elif "_distance" in row:
                score_val = f"distance: {row['_distance']:.4f}"
            elif "_score" in row:
                score_val = f"score: {row['_score']:.4f}"
            else:
                score_val = ""
            response += (
                f"📄 File: {row['source']} "
                f"(fragment {row['fragment_id']}, {score_val}, model: {self.embedding_model_name})\n"
                f"{row['text']}\n\n"
            )
        return response

class SqliteVectorDatabase(VectorDatabaseInfo):

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

    def create_new_database(
            self,
            *,
            text_chunks: List[str],
            chunks_metadata: List[dict],
            hash_ids: List[str],
            embeddings: Tuple[np.ndarray, List[dict[str, float]], List[np.ndarray]]
    ):
        def convert_sparse_to_json(sparse_dict):
            return {key: float(value) for key, value in sparse_dict.items()}  # np.float16 -> float

        print(f'dense3: {type(embeddings[0])}')
        print(f'sparse3: {type(embeddings[1])}')
        print(f'colbert3: {type(embeddings[2])}')

        database_folder = self.get_database_type().db_folder
        os.makedirs(database_folder, exist_ok=True)
        db_path = os.path.join(database_folder, f"{self.database_name}.db")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Tworzenie tabeli dla embeddingów
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.database_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    text TEXT,
                    fragment_id INT,
                    hash_id TEXT,
                    {EmbeddingType.DENSE.value} BLOB,
                    {EmbeddingType.SPARSE.value} TEXT,
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
                dense_blob = embeddings[0][i].tobytes() if embeddings[0].size > 0 else None
                sparse_json = json.dumps(convert_sparse_to_json(embeddings[1][i])) if embeddings[1] else None
                colbert_blob = embeddings[2][i].tobytes() if embeddings[2] else None
                colbert_shape = json.dumps(embeddings[2][i].shape) if embeddings[2] else None

                source = chunks_metadata[i]['source']
                text = text_chunks[i]
                fragment_id = chunks_metadata[i]['fragment_id']
                hash_id = hash_ids[i]

                cursor.execute(f'''
                    INSERT INTO {self.database_name} (source, text, fragment_id, hash_id, 
                    {EmbeddingType.DENSE.value}, {EmbeddingType.SPARSE.value}, 
                    {EmbeddingType.COLBERT.value}, {EmbeddingType.COLBERT.value}_shape)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (source, text, fragment_id, hash_id, dense_blob, sparse_json, colbert_blob, colbert_shape))

            # Wstawianie metadanych po pętli
            metadata_json = json.dumps(self.to_dict())
            cursor.execute('INSERT INTO metadata (metadata_json) VALUES (?)', (metadata_json,))

            # Automatyczny commit przy wyjściu z bloku 'with'

    def retrieve_from_database(self) -> Tuple[List, List, List, Tuple[np.ndarray, List[dict[str, float]], List[np.ndarray]]]:
        db_path = os.path.join(self.get_database_type().db_folder, self.database_name, f'{self.database_name}.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Wczytanie metadanych (opcjonalne, ale zostawiam dla spójności)
            cursor.execute("SELECT metadata_json FROM metadata LIMIT 1")
            metadata_row = cursor.fetchone()
            # metadata = json.loads(metadata_row[0]) if metadata_row else {}  # Nie używamy, ale zostawiam

            # Wczytanie embeddingów
            cursor.execute(f'''
                SELECT source, text, fragment_id, hash_id, 
                {EmbeddingType.DENSE.value}, {EmbeddingType.SPARSE.value}, 
                {EmbeddingType.COLBERT.value}, {EmbeddingType.COLBERT.value}_shape 
                FROM {self.database_name}
            ''')
            rows = cursor.fetchall()

            # Inicjalizacja list wynikowych
            text_chunks: List[str] = []
            chunks_metadata: List[Dict[str, str]] = []
            hash_ids: List[str] = []
            dense_vectors_list: List[np.ndarray] = []  # Tymczasowa lista na wektory dense
            sparse_embeddings: List[Dict[str, float]] = []
            colbert_embeddings: List[np.ndarray] = []

            # Przetwarzanie danych
            for row in rows:
                source, text, fragment_id, hash_id, dense_blob, sparse_json, colbert_blob, colbert_shape = row

                # Tekst
                text_chunks.append(text)

                # Metadane
                chunks_metadata.append({"source": source, "fragment_id": fragment_id})

                # Hash
                hash_ids.append(hash_id)

                # Embeddingi z obsługą NULL
                # Dense
                if dense_blob is not None:
                    dense_vector = np.frombuffer(dense_blob, dtype=np.float32)
                    dense_vectors_list.append(dense_vector)
                else:
                    dense_vectors_list.append(np.zeros(dense_dim, dtype=np.float32))  # Zerowy wektor dla NULL

                # Sparse
                if sparse_json is not None:
                    sparse_vector = json.loads(sparse_json)  # dict
                    sparse_embeddings.append(sparse_vector)
                else:
                    sparse_embeddings.append(None)

                # ColBERT
                if colbert_blob is not None and colbert_shape is not None:
                    colbert_shape_tuple = tuple(json.loads(colbert_shape))
                    colbert_vector = np.frombuffer(colbert_blob, dtype=np.float32).reshape(colbert_shape_tuple)
                    colbert_embeddings.append(colbert_vector)
                else:
                    colbert_embeddings.append(None)

                # Konwersja listy dense_vectors_list na np.ndarray
            if dense_vectors_list:
                dense_embeddings = np.stack(dense_vectors_list)  # Tworzy tablicę (n, d)
            else:
                dense_embeddings = np.array([], dtype=np.float32)  # Pusta tablica, jeśli brak danych

            # Zwracanie w formacie tuple[List, List, List, tuple[List, List, List]]
            return text_chunks, chunks_metadata, hash_ids, (dense_embeddings, sparse_embeddings, colbert_embeddings)

