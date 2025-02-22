import os
import shutil
from enum import Enum
from pathlib import Path
from typing import List
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from src.config import MODEL_FOLDER
from src.enums.embedding_type_enum import EmbeddingType
from FlagEmbedding import FlagModel, BGEM3FlagModel
from huggingface_hub import snapshot_download


class TransformerLibrary(Enum):
    FlagEmbedding = ("FlagEmbedding", [EmbeddingType.DENSE, EmbeddingType.SPARSE, EmbeddingType.COLBERT])
    SentenceTransformers = ("SentenceTransformers", [EmbeddingType.DENSE])

    def __init__(self, display_name, supported_embeddings: List[EmbeddingType]):
        self.display_name = display_name
        self.supported_embeddings = supported_embeddings

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
    def is_sentence_transformer_model(target_dir: str, model_name: str) -> dict:
        print('⚙️ SentenceTransformers library Test')
        basic_dict = {
            TransformerLibrary.SentenceTransformers.display_name: []
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
                basic_dict[TransformerLibrary.SentenceTransformers.display_name].append(EmbeddingType.DENSE.value)
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
            TransformerLibrary.FlagEmbedding.display_name: []
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
                    basic_dict[TransformerLibrary.FlagEmbedding.display_name].append(emb_type.value)
                else:
                    print(f"❌ Model nie obsługuje: {emb_type.value}")

        except Exception as e:
            print(f"Błąd podczas testów: {e}")
            return {}

        return basic_dict




