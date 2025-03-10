import json
import os
import shutil
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
    def load_embedding_model(model_name: str) -> str:
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

    def generate_embeddings(self, text_chunks: List[str], vector_database_instance: "VectorDatabaseInfo") -> Tuple[Optional[np.ndarray], Optional[List[dict[str, float]]], Optional[List[np.ndarray]]]:
        """
        Generate embeddings based on the enum type and requested embedding type.
        transformer_library: TransformerLibrary = vector_database_instance.transformer_library
        """
        list_of_embeddings_to_create = vector_database_instance.embedding_types

        dense_embeddings: Optional[np.ndarray] = None
        sparse_embeddings: Optional[List[dict[str, float]]] = None
        colbert_embeddings: Optional[List[np.ndarray]] = None

        selected_model_path = self.load_embedding_model(vector_database_instance.embedding_model_name)
        if self == TransformerLibrary.SentenceTransformers:
            if vector_database_instance.float_precision == FloatPrecisionPointEnum.FP32:
                embedding_model = SentenceTransformer(
                    selected_model_path
                )
            elif vector_database_instance.float_precision == FloatPrecisionPointEnum.FP16:
                embedding_model = SentenceTransformer(
                    selected_model_path
                ).half()

            dense_embeddings = embedding_model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)

        elif self == TransformerLibrary.FlagEmbedding:
            embedding_model = BGEM3FlagModel(selected_model_path, use_fp16=(vector_database_instance.float_precision == FloatPrecisionPointEnum.FP16))

            generated_embeddings = embedding_model.encode(
                sentences=text_chunks,
                return_dense=EmbeddingType.DENSE in list_of_embeddings_to_create,
                return_sparse=EmbeddingType.SPARSE in list_of_embeddings_to_create,
                return_colbert_vecs=EmbeddingType.COLBERT in list_of_embeddings_to_create
            )

            dense_embeddings = generated_embeddings.get('dense_vecs', None)
            sparse_embeddings = generated_embeddings.get('lexical_weights', None)
            colbert_embeddings = generated_embeddings.get('colbert_vecs', None)

            print(f'dense1: {type(dense_embeddings)}')
            print(f'sparse1: {type(sparse_embeddings)}')
            print(f'colbert1: {type(colbert_embeddings)}')

        torch.cuda.empty_cache()
        return dense_embeddings, sparse_embeddings, colbert_embeddings

    def perform_search(
            self, *,
            text_chunks: List[str],
            chunks_metadata: List[ChunkMetadataModel],
            embeddings: tuple[List, List, List],
            query: str,
            vector_database_instance: "VectorDatabaseInfo", top_k: int
    ) -> Tuple[List[str], List[ChunkMetadataModel], List[float]]:
        if self == TransformerLibrary.SentenceTransformers:
            print('SentenceTransformers Search')
            dense_embeddings = embeddings[0] # ONLY DENSE
            query_embeddings = self.generate_embeddings([query], vector_database_instance)[0] # ONLY DENSE

            floating_precision = vector_database_instance.float_precision
            d_type = torch.float32 if floating_precision == FloatPrecisionPointEnum.FP32 else torch.float16

            dense_embeddings_tensor = torch.tensor(dense_embeddings, dtype=d_type)
            query_embeddings_tensor = torch.tensor(query_embeddings, dtype=d_type)

            result = util.semantic_search(query_embeddings_tensor, dense_embeddings_tensor, top_k=top_k)

            result_text: List[str] = []
            result_chunks_metadata: List[ChunkMetadataModel] = []
            result_scores: List[float] = []
            for result_from_dict in result[0]:
                corpus_id: int = result_from_dict['corpus_id']
                result_text.append(text_chunks[corpus_id])
                result_chunks_metadata.append(chunks_metadata[corpus_id])
                result_scores.append(result_from_dict['score'])


            torch.cuda.empty_cache()
            return result_text, result_chunks_metadata, result_scores


        elif self == TransformerLibrary.FlagEmbedding:
            print('FlagEmbedding Search')
            return [], [], []
        else:
            return [], [], []



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




