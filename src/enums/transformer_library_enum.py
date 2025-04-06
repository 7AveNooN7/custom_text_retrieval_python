import json
import os
import shutil
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.config import MODEL_FOLDER
from src.enums.embedding_type_enum import EmbeddingType
from FlagEmbedding import FlagModel, BGEM3FlagModel
from huggingface_hub import snapshot_download

from src.enums.floating_precision_enum import FloatPrecisionPointEnum
from src.models.chunk_metadata_model import ChunkMetadataModel

_model_cache = OrderedDict()  # Zachowuje kolejność dodawania
MAX_CACHE_SIZE = 3  # Maksymalna liczba modeli w cache

def _get_cached_model(transformer_library: "TransformerLibrary", model_name: str, float_precision: FloatPrecisionPointEnum):
    key = (transformer_library.display_name, model_name, float_precision)
    if key not in _model_cache:
        model_path = TransformerLibrary.get_embedding_model_path(model_name)
        if transformer_library == TransformerLibrary.FlagEmbedding:
            model = BGEM3FlagModel(model_path, use_fp16=(float_precision == FloatPrecisionPointEnum.FP16))
        elif transformer_library == TransformerLibrary.SentenceTransformers:
            model = SentenceTransformer(model_path)
            if float_precision== FloatPrecisionPointEnum.FP16:
                model = model.half()
        else:
            raise ValueError(f"Nieobsługiwana biblioteka: {transformer_library}")
        if len(_model_cache) >= MAX_CACHE_SIZE:
            _model_cache.popitem(last=False)  # Usuwa najstarszy element
        _model_cache[key] = model
    return _model_cache[key]


class TransformerLibrary(Enum):
    FlagEmbedding = (
        "FlagEmbedding",
        [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT]
    )
    SentenceTransformers = (
        "SentenceTransformers",
        [EmbeddingType.DENSE]
    )

    def __init__(self, display_name: str, supported_embeddings: List[EmbeddingType]):
        self.display_name: str = display_name
        self.supported_embeddings: List[EmbeddingType] = supported_embeddings


    @classmethod
    def from_display_name(cls, display_name: str):
        """Konwersja stringa na DatabaseType"""
        for db in cls:
            if db.display_name == display_name:
                return db
        raise ValueError(f"Niepoprawna wartość: {display_name}")

    @staticmethod
    def list() -> List[str]:
        """Zwraca listę wszystkich wartości enuma"""
        return [e.display_name for e in TransformerLibrary]

    @staticmethod
    def get_embedding_model_path(model_name: str) -> str:
        """
        Ładuje model Sentence Transformers, szukając folderu, w którym
        wartość "model_name" w metadata.json pasuje do model_instance.name.
        """
        target_model_name = model_name
        selected_model_path = None

        # Przeszukujemy MODEL_FOLDER w poszukiwaniu pasującego modelu
        for model_folder in os.scandir(MODEL_FOLDER):
            if model_folder.is_dir():
                metadata_json_path = os.path.join(model_folder.path, "metadata.json")

                if os.path.isfile(metadata_json_path):
                    try:
                        with open(metadata_json_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            if metadata.get("model_name") == target_model_name:
                                selected_model_path = model_folder.path
                                break  # Znaleźliśmy pasujący model, przerywamy pętlę
                    except Exception as e:
                        print(f"⚠️ Błąd podczas odczytu metadata.json w {model_folder.name}: {e}")

        if not selected_model_path:
            raise FileNotFoundError(f"❌ Nie znaleziono modelu '{target_model_name}' w katalogu {MODEL_FOLDER}")


        return selected_model_path

    @staticmethod
    def is_sentence_transformer_model(target_dir: str, model_name: str) -> dict:
        print('⚙️ SentenceTransformers library Test')
        basic_dict = {
            TransformerLibrary.SentenceTransformers: []
        }
        # Przykładowe zdania do testu
        question = ["Where is number one"]
        text = [
            "one, two, three",
            "four, five, six"
        ]

        try:
            results = []
            for i in range(2):
                print(f"--- Iteracja {i + 1} ---")
                if i > 0:
                    safe_model_dir: str = str(model_name.replace("/", "_"))
                    target_temp_dir = str(os.path.join(MODEL_FOLDER, 'temp', safe_model_dir))
                    if os.path.exists(target_temp_dir):
                        shutil.rmtree(target_temp_dir)
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=target_temp_dir
                    )
                    model = SentenceTransformer(target_temp_dir)
                else:
                    model = SentenceTransformer(target_dir)

                # Kodowanie pytania i tekstów źródłowych osobno
                question_embedding = model.encode(question, convert_to_tensor=True)
                text_embeddings = model.encode(text, convert_to_tensor=True)

                # Obliczanie podobieństwa kosinusowego
                similarities = util.cos_sim(question_embedding, text_embeddings)
                results.append(similarities)

                print(f"{similarities}")

            if torch.equal(results[0], results[1]):
                print("✅ Model obsługuj SentenceTransformers.")
                basic_dict[TransformerLibrary.SentenceTransformers].append(EmbeddingType.DENSE)
                return basic_dict
            else:
                print("❌ Model nie obsługuje SentenceTransformers - błędne wyniki.")
                return {}


        except Exception as e:
            print("❌ Model nie obsługuje SentenceTransformers - wystąpił błąd.")
            print(f'Błąd: {e}')
            return {}

    @staticmethod
    def is_flag_embedding_model(target_dir: str, model_name: str) -> dict:
        """
        Sprawdza, jakie typy osadzeń obsługuje model, wykonując dwa testy
        (dwukrotnie ładuje model, by sprawdzić czy wyniki są spójne).

        Wykorzystuje enum EmbeddingType, aby określić typy embeddingów.
        """
        print('⚙️ FlagEmbedding Library Test')
        basic_dict = {
            TransformerLibrary.FlagEmbedding: []
        }
        # Inicjujemy słownik wyników dla każdego typu embeddings
        results = {
            EmbeddingType.DENSE: False,
            EmbeddingType.SPARSE: False,
            EmbeddingType.COLBERT: False
        }

        question = ["Where is number one"]
        text = [
            "one, two, three",
            "four, five, six"
        ]

        def run_embedding_tests(iteration: int):
            """
            Wykonuje właściwe testy dla różnych rodzajów embeddings.
            iteration - numer iteracji (0 lub 1)
                        - przy iteration > 0 pobieramy model do 'temp'
                        - w innym wypadku używamy oryginalnego modelu
            """
            try:
                if iteration > 0:
                    safe_model_dir: str = str(model_name.replace("/", "_"))
                    target_temp_dir = os.path.join(MODEL_FOLDER, 'temp', safe_model_dir)
                    if os.path.exists(target_temp_dir):
                        shutil.rmtree(target_temp_dir)

                    snapshot_download(
                        repo_id=model_name,
                        local_dir=target_temp_dir
                    )
                    model = BGEM3FlagModel(target_temp_dir)
                else:
                    model = BGEM3FlagModel(target_dir)

                output = {}
                # Dense embeddings
                try:
                    print('1️⃣ Dense embeddings test')
                    embeddings_1 = model.encode(question, batch_size=12, max_length=8192)['dense_vecs']
                    embeddings_2 = model.encode(text)['dense_vecs']
                    dense_similarity = embeddings_1 @ embeddings_2.T
                    output['dense_similarity'] = dense_similarity
                    print(f'vector: {dense_similarity}')
                except Exception as e:
                    print(f"❌ Dense: Błąd - {e}")

                # Sparse embeddings
                try:
                    print('2️⃣ Sparse embeddings test')
                    question_encoded = model.encode(
                        question,
                        return_dense=False,
                        return_sparse=True,
                        return_colbert_vecs=False
                    )
                    text_encoded = model.encode(
                        text,
                        return_dense=False,
                        return_sparse=True,
                        return_colbert_vecs=False
                    )
                    # Zwróćmy same identyfikatory (dla porównania)
                    tokens_from_question = model.convert_id_to_token(question_encoded['lexical_weights'])
                    tokens_from_text = model.convert_id_to_token(text_encoded['lexical_weights'])
                    output['tokens_from_question'] = tokens_from_question
                    print(f'question tokens: {tokens_from_question}')
                    output['tokens_from_text'] = tokens_from_text
                    print(f'text tokens: {tokens_from_text}')
                    # compute the scores via lexical matching
                    lexical_scores_1 = model.compute_lexical_matching_score(
                        question_encoded['lexical_weights'][0],
                        text_encoded['lexical_weights'][0]
                    )
                    lexical_scores_2 = model.compute_lexical_matching_score(
                        question_encoded['lexical_weights'][0],
                        text_encoded['lexical_weights'][1]
                    )
                    output['lexical_scores_1'] = lexical_scores_1
                    output['lexical_scores_2'] = lexical_scores_2
                    print(f'score 1: {lexical_scores_1}')
                    print(f'score 2: {lexical_scores_2}')
                except Exception as e:
                    print(f"❌ Sparse: Błąd - {e}")

                # ColBERT embeddings
                try:
                    print('3️⃣ ColBERT embeddings test')
                    question_encoded = model.encode(
                        question,
                        return_dense=False,
                        return_sparse=False,
                        return_colbert_vecs=True
                    )
                    text_encoded = model.encode(
                        text,
                        return_dense=False,
                        return_sparse=False,
                        return_colbert_vecs=True
                    )
                    colbert_score_1 = model.colbert_score(
                        question_encoded['colbert_vecs'][0],
                        text_encoded['colbert_vecs'][0]
                    )
                    colbert_score_2 = model.colbert_score(
                        question_encoded['colbert_vecs'][0],
                        text_encoded['colbert_vecs'][1]
                    )
                    print(f'score 1: {colbert_score_1}')
                    print(f'score 2: {colbert_score_2}')
                    output['colbert_score_1'] = colbert_score_1
                    output['colbert_score_2'] = colbert_score_2
                except Exception as e:
                    print(f"❌ ColBERT: Błąd - {e}")

            except Exception as e:
                print(f"Błąd podczas testów osadzeń: {e}")
                return None

            return output

        try:
            results_list = []
            for i in range(2):
                print(f"--- Iteracja {i + 1} ---")
                res = run_embedding_tests(i)
                results_list.append(res)

            # Jeżeli w obu iteracjach (results_list[0], results_list[1]) mamy dane, to je porównujemy
            if results_list[0] and results_list[1]:

                # 1. DENSE
                #   - sprawdzamy dense_similarity
                if "dense_similarity" in results_list[0] and "dense_similarity" in results_list[1]:
                    try:
                        if np.array_equal(
                                results_list[0]["dense_similarity"],
                                results_list[1]["dense_similarity"]
                        ):
                            results[EmbeddingType.DENSE] = True
                        else:
                            results[EmbeddingType.DENSE] = False
                    except Exception as e:
                        print(f"Błąd porównania dense_similarity: {e}")
                        results[EmbeddingType.DENSE] = False

                # 2. SPARSE
                #   - sprawdzamy tokens_from_question, tokens_from_text, lexical_scores_1, lexical_scores_2
                def is_effectively_empty_tokens(value):
                    """
                    Zwraca True, jeśli 'value' jest pusty lub w środku nic nie ma (np. [ {}, {} ]).
                    """
                    if not value:
                        # np. pusta lista lub None
                        return True

                    # Jeśli to np. lista typu [ {}, {}, ... ]
                    # uznajemy, że pusta = wszystkie elementy są puste
                    if isinstance(value, list):
                        # sprawdźmy, czy każdy element jest pusty (np. puste dict, puste listy, "")
                        return all(not bool(elem) for elem in value)

                    # Jeżeli nie jest listą, a np. stringiem lub dict-em,
                    # to za "puste" uznajemy bool(value) == False
                    # (czyli empty string, pusty dict itp.)
                    return not bool(value)

                sparse_ok = True
                for sparse_key in [
                    "tokens_from_question",
                    "tokens_from_text",
                    "lexical_scores_1",
                    "lexical_scores_2"
                ]:
                    if sparse_key in results_list[0] and sparse_key in results_list[1]:
                        try:
                            # Dodatkowe sprawdzenie pustych tokenów (dotyczy tylko 'tokens_from_*'):
                            if sparse_key in ("tokens_from_question", "tokens_from_text"):
                                if (is_effectively_empty_tokens(results_list[0][sparse_key])
                                        or is_effectively_empty_tokens(results_list[1][sparse_key])):
                                    print(f"Błąd: {sparse_key} jest efektywnie pusty w jednej z iteracji!")
                                    sparse_ok = False

                            # Następnie porównanie wartości z iteracji 0 i 1
                            if isinstance(results_list[0][sparse_key], np.ndarray):
                                if not np.array_equal(
                                        results_list[0][sparse_key],
                                        results_list[1][sparse_key]
                                ):
                                    sparse_ok = False
                            elif isinstance(results_list[0][sparse_key], (int, float)):
                                if results_list[0][sparse_key] != results_list[1][sparse_key]:
                                    sparse_ok = False
                            else:
                                # Porównanie list / stringów / innych obiektów
                                if results_list[0][sparse_key] != results_list[1][sparse_key]:
                                    sparse_ok = False

                        except Exception as e:
                            print(f"Błąd porównania {sparse_key}: {e}")
                            sparse_ok = False
                    else:
                        sparse_ok = False

                results[EmbeddingType.SPARSE] = sparse_ok

                # 3. COLBERT
                #   - sprawdzamy colbert_score_1, colbert_score_2
                colbert_ok = True
                for colbert_key in ["colbert_score_1", "colbert_score_2"]:
                    if colbert_key in results_list[0] and colbert_key in results_list[1]:
                        try:
                            if results_list[0][colbert_key] != results_list[1][colbert_key]:
                                colbert_ok = False
                        except Exception as e:
                            print(f"Błąd porównania {colbert_key}: {e}")
                            colbert_ok = False
                    else:
                        colbert_ok = False

                results[EmbeddingType.COLBERT] = colbert_ok

            # Po porównaniu wypiszmy, które embeddingi są obsługiwane
            print("\n--- Podsumowanie obsługiwanych embeddingów ---")
            for emb_type in EmbeddingType:
                if results[emb_type]:
                    print(f"✅ Model obsługuje: {emb_type.value}")
                    basic_dict[TransformerLibrary.FlagEmbedding].append(emb_type)
                else:
                    print(f"❌ Model nie obsługuje: {emb_type.value}")

        except Exception as e:
            print(f"Błąd podczas testów: {e}")
            return {}

        return basic_dict

    def generate_embeddings(self, text_chunks_or_queries: List[str], vector_database_instance: "VectorDatabaseInfo") -> Tuple[Optional[List[np.ndarray]], Optional[List[List[dict[str, float]]]], Optional[List[List[np.ndarray]]]]:
        """
        Generate embeddings based on the enum type and requested embedding type.
        transformer_library: TransformerLibrary = vector_database_instance.transformer_library
        """
        print(f'TRANSFORMER LIB FUNCTION: Generate Embeddings')
        list_of_embeddings_to_create = vector_database_instance.embedding_types

        dense_embeddings: Optional[np.ndarray] = None
        sparse_embeddings: Optional[List[dict[str, float]]] = None
        colbert_embeddings: Optional[List[np.ndarray]] = None

        embedding_model = _get_cached_model(
            transformer_library=self,
            model_name=vector_database_instance.embedding_model_name,
            float_precision=vector_database_instance.float_precision
        )

        if self == TransformerLibrary.SentenceTransformers:
            print(f'SentenceTransformers: Generate Embeddings')
            dense_embeddings = embedding_model.encode(text_chunks_or_queries, show_progress_bar=True, convert_to_numpy=True)

            print(f'dense_embeddings: {dense_embeddings.shape}')

        elif self == TransformerLibrary.FlagEmbedding:
            print(f'FlagEmbedding: Generate Embeddings')

            # WEKTORY ZNORMALIZOWANE WYRZUCA (DENSE)
            generated_embeddings = embedding_model.encode(
                sentences=text_chunks_or_queries,
                return_dense=EmbeddingType.DENSE in list_of_embeddings_to_create,
                return_sparse=EmbeddingType.SPARSE in list_of_embeddings_to_create,
                return_colbert_vecs=EmbeddingType.COLBERT in list_of_embeddings_to_create,
            )

            dense_embeddings = generated_embeddings.get('dense_vecs', None)
            print(f'dense_embeddings: {dense_embeddings.shape}')
            sparse_embeddings = generated_embeddings.get('lexical_weights', None)
            print(f'sparse_embeddings: {len(sparse_embeddings)}')
            colbert_embeddings = generated_embeddings.get('colbert_vecs', None)
            print(f'colbert_embeddings: {len(colbert_embeddings)}')


        return dense_embeddings, sparse_embeddings, colbert_embeddings

    def perform_search(
            self, *,
            text_chunks: List[str],
            chunks_metadata: List[ChunkMetadataModel],
            embeddings: tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]],
            query_list: List[str],
            vector_database_instance: "VectorDatabaseInfo",
            top_k: int,
            vector_choices: List[str]
    ) -> Tuple[List[str], List[ChunkMetadataModel], List[float]]:
        result_text: List[str] = []
        result_chunks_metadata: List[ChunkMetadataModel] = []
        result_scores: List[float] = []

        query_output: Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[
            List[np.ndarray]]] = self.generate_embeddings(query_list,
                                                          vector_database_instance)  # Generujemy embeddingi dla zapytania

        if self == TransformerLibrary.SentenceTransformers:
            print('SentenceTransformers Search')
            dense_embeddings = embeddings[0] # ONLY DENSE
            query_dense = query_output[0]

            dense_embeddings_tensor = torch.tensor(dense_embeddings, dtype=vector_database_instance.float_precision.torch_dtype)
            query_embeddings_tensor = torch.tensor(query_dense, dtype=vector_database_instance.float_precision.torch_dtype)

            results = util.semantic_search(query_embeddings_tensor, dense_embeddings_tensor, top_k=top_k)
            for result in results:
                many_results_text = []
                for result_from_dict in result:
                    corpus_id: int = result_from_dict['corpus_id']
                    result_text.append(text_chunks[corpus_id])
                    result_chunks_metadata.append(chunks_metadata[corpus_id])
                    result_scores.append(result_from_dict['score'])

            return result_text, result_chunks_metadata, result_scores


        elif self == TransformerLibrary.FlagEmbedding:
            # TODO: Implement FlagEmbedding search
            print('FlagEmbedding: Search')
            dense_embeddings, sparse_embeddings, colbert_embeddings = embeddings  # Rozpakowujemy krotkę embeddingów

            query_dense = query_output[0]  # Dense embedding dla zapytania
            query_sparse = query_output[1]  # Sparse embedding dla zapytania
            query_colbert = query_output[2]  # ColBERT embedding dla zapytania

            torch_d_type = vector_database_instance.float_precision.torch_dtype
            numpy_d_type = vector_database_instance.float_precision.numpy_dtype

            selected_model_path = self.get_embedding_model_path(vector_database_instance.embedding_model_name)
            model = BGEM3FlagModel(selected_model_path, use_fp16=(torch_d_type == torch.float16))

            for vector_choice in vector_choices:
                # DENSE
                if vector_choice == EmbeddingType.DENSE.value and dense_embeddings is not None:
                    print('FlagEmbedding: DENSE Search')
                    dense_embeddings_tensor = torch.tensor(dense_embeddings, dtype=vector_database_instance.float_precision.torch_dtype)
                    query_dense_tensor = torch.tensor(query_dense, dtype=vector_database_instance.float_precision.torch_dtype)
                    dense_score = model.model.compute_dense_score(query_dense_tensor, dense_embeddings_tensor)
                    #print(f'dense_score: {dense_score}')
                    dense_score = dense_score.squeeze(0) # BO TYLKO JEDEN TENSOR ZAPYTANIA

                    # Znalezienie top-k indeksów i wyników (np. top_k = 5)
                    top_k_values, top_k_indices = torch.topk(dense_score, k=top_k, largest=True)

                    # Pobranie tekstów, metadanych i wyników na podstawie indeksów
                    for i in range(top_k):
                        corpus_id = top_k_indices[i].item()  # Get the corpus index
                        result_text.append(text_chunks[corpus_id])  # Text of the chunk
                        result_chunks_metadata.append(chunks_metadata[corpus_id])  # Metadata of the chunk
                        result_scores.append(top_k_values[i].item())  # Corresponding similarity score

                # SPARSE
                elif vector_choice == EmbeddingType.SPARSE.value and sparse_embeddings is not None:
                    print('FlagEmbedding: SPARSE Search')
                    sparse_scores = model.compute_lexical_matching_score(query_sparse, sparse_embeddings)
                    sparse_scores = sparse_scores.flatten()
                    sparse_scores_tensor = torch.from_numpy(sparse_scores)
                    top_k_values, top_k_indices = torch.topk(sparse_scores_tensor, k=top_k, largest=True)
                    #print(f'sparse_scores ({sparse_scores_tensor.dtype}): {sparse_scores_tensor}')

                    for i in range(top_k):
                        corpus_id = top_k_indices[i].item()  # Indeks fragmentu w sparse_embeddings
                        result_text.append(text_chunks[corpus_id])  # Tekst fragmentu
                        result_chunks_metadata.append(chunks_metadata[corpus_id])  # Metadane fragmentu
                        result_scores.append(top_k_values[i].item())  # Wynik podobieństwa

                # COLBERT
                elif vector_choice == EmbeddingType.COLBERT.value and colbert_embeddings is not None:
                    # Obliczamy wyniki dla ColBERT
                    print('FlagEmbedding: COLBERT Search')
                    colbert_scores = []
                    query_array = query_colbert[0]
                    for chunk_embedding in colbert_embeddings:
                        colbert_scores.append(model.colbert_score(query_array, chunk_embedding))

                    # Konwersja listy wyników na tensor PyTorch z określonym typem danych
                    colbert_scores_tensor = torch.tensor(colbert_scores)
                    # Znalezienie top-k indeksów i wyników
                    top_k_values, top_k_indices = torch.topk(colbert_scores_tensor, k=top_k, largest=True)
                    # Przygotowanie list wynikowych
                    result_text = []
                    result_chunks_metadata = []
                    result_scores = []
                    # Pobranie tekstów, metadanych i wyników na podstawie indeksów
                    for i in range(top_k):
                        corpus_id = top_k_indices[i].item()  # Indeks fragmentu
                        result_text.append(text_chunks[corpus_id])  # Tekst fragmentu
                        result_chunks_metadata.append(chunks_metadata[corpus_id])  # Metadane fragmentu
                        result_scores.append(top_k_values[i].item())  # Wynik podobieństwa


            return result_text, result_chunks_metadata, result_scores
        else:
            return [], [], []






