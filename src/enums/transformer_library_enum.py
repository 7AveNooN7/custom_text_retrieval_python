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

    def __init__(self, value, supported_embeddings: List[EmbeddingType]):
        self._value_ = value
        self.supported_embeddings = supported_embeddings

    @staticmethod
    def list():
        """Zwraca listę wszystkich wartości enuma"""
        return [e.value for e in TransformerLibrary]

    @staticmethod
    def is_sentence_transformer_model(target_dir: str, model_name: str) -> dict:
        print('SentenceTransformers Check')
        # Przykładowe zdania do testu
        question = ["Where is number one"]
        text = [
            "one, two, three",
            "four, five, six"
        ]

        try:
            results = []

            for i in range(2):
                print(f"\n--- Iteracja {i + 1} ---")
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

                print(f"similarities: {similarities}")

            # Proste, bez tolerancji
            if torch.equal(results[0], results[1]):
                print("\n✅ Zmienna similarities jest identyczna w obu iteracjach.")
            else:
                print("\n❌ Zmienna similarities różni się między iteracjami.")


        except Exception as e:
            print(f"❌ Błąd: {e}")
            return {"status": "error", "reason": str(e)}

    # @staticmethod
    # def is_flag_embedding_model(target_dir: str, model_name: str):
    #     """Sprawdza, jakie typy osadzeń obsługuje model oraz przeprowadza testowe wyszukiwanie."""
    #     results = {"dense": False, "sparse": False, "colbert": False}
    #     question = ["Where is number one"]
    #     text = [
    #         "one, two, three",
    #         "four, five, six"
    #     ]
    #
    #     def run_embedding_tests(iteration: int):
    #         try:
    #             if iteration > 0:
    #                 safe_model_dir: str = str(model_name.replace("/", "_"))
    #                 target_temp_dir = str(os.path.join(MODEL_FOLDER, 'temp', safe_model_dir))
    #                 if os.path.exists(target_temp_dir):
    #                     shutil.rmtree(target_temp_dir)
    #                 snapshot_download(
    #                     repo_id=model_name,
    #                     local_dir=target_temp_dir
    #                 )
    #                 model = BGEM3FlagModel(target_temp_dir)
    #             else:
    #                 model = BGEM3FlagModel(target_dir)
    #
    #             output = {}
    #             # Dense embeddings
    #             try:
    #                 print('Dense embeddings')
    #
    #                 embeddings_1 = model.encode(question, batch_size=12, max_length=8192, )['dense_vecs']
    #                 embeddings_2 = model.encode(text)['dense_vecs']
    #                 dense_similarity = embeddings_1 @ embeddings_2.T
    #                 output['dense_similarity'] = embeddings_1 @ embeddings_2.T
    #                 print(f'dense_similarity: {dense_similarity}')
    #             except Exception as e:
    #                 print(f"❌ Dense: Błąd - {e}")
    #                 results["dense"] = False
    #
    #
    #             # Sparse embeddings
    #             try:
    #                 print('Sparse embeddings')
    #                 question_encoded = model.encode(question, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    #                 text_encoded = model.encode(text, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    #                 tokens_from_question = model.convert_id_to_token(question_encoded['lexical_weights'])
    #                 tokens_from_text = model.convert_id_to_token(text_encoded['lexical_weights'])
    #                 #print(f'tokens_from_question: {tokens_from_question}')
    #                 # compute the scores via lexical mathcing
    #                 lexical_scores_1 = model.compute_lexical_matching_score(question_encoded['lexical_weights'][0],
    #                                                                         text_encoded['lexical_weights'][0])
    #                 print(f'lexical_scores_1: {lexical_scores_1}')
    #                 lexical_scores_2 = model.compute_lexical_matching_score(question_encoded['lexical_weights'][0],
    #                                                                         text_encoded['lexical_weights'][1])
    #                 print(f'lexical_scores_2: {lexical_scores_2}')
    #
    #                 output['lexical_scores_1'] = lexical_scores_1
    #                 output['lexical_scores_2'] = lexical_scores_2
    #             except Exception as e:
    #                 print(f"❌ Sparse: Błąd - {e}")
    #                 results["sparse"] = False
    #
    #             # Sprawdź generowanie ColBERT embeddings
    #             try:
    #                 print('ColBERT embeddings')
    #                 question_encoded = model.encode(question, return_dense=False, return_sparse=False,
    #                                                 return_colbert_vecs=True)
    #                 text_encoded = model.encode(text, return_dense=False, return_sparse=False,
    #                                             return_colbert_vecs=True)
    #                 colbert_score_1 = model.colbert_score(question_encoded['colbert_vecs'][0],
    #                                                       text_encoded['colbert_vecs'][0])
    #                 colbert_score_2 = model.colbert_score(question_encoded['colbert_vecs'][0],
    #                                                       text_encoded['colbert_vecs'][1])
    #                 print(f'colbert_score_1: {colbert_score_1}')
    #                 print(f'colbert_score_2: {colbert_score_2}')
    #                 output['colbert_score_1'] = colbert_score_1
    #                 output['colbert_score_2'] = colbert_score_2
    #                 output['question_colbert_vec'] = question_encoded['colbert_vecs']
    #                 output['text_colbert_vec'] = text_encoded['colbert_vecs']
    #             except Exception as e:
    #                 print(f"❌ ColBERT: Błąd - {e}")
    #                 results["colbert"] = False
    #
    #
    #         except Exception as e:
    #             print(f"Błąd podczas testów osadzeń: {e}")
    #             return None
    #         return output
    #
    #     try:
    #         results_list = []
    #         for i in range(2):
    #             print(f"--- Iteracja {i + 1} ---")
    #             results_list.append(run_embedding_tests(i))
    #             print(results_list[-1])
    #
    #         # if all(results_list):
    #         #     print("--- Porównanie wyników ---")
    #         #     for key in results_list[0]:
    #         #         if all(results[key] == results_list[0][key] for results in results_list):
    #         #             print(f"✅ {key}: Wyniki są identyczne.")
    #         #         else:
    #         #             print(f"❌ {key}: Różnice w wynikach!")
    #         if all(results_list):
    #             print("--- Porównanie wyników ---")
    #             for key in results_list[0]:
    #                 try:
    #                     first_result = results_list[0][key]
    #                     if isinstance(first_result, np.ndarray):
    #                         # For NumPy arrays
    #                         if all(np.array_equal(first_result, results[key]) for results in results_list):
    #                             print(f"✅ {key}: Wyniki są identyczne.")
    #                         else:
    #                             print(f"❌ {key}: Różnice w wynikach!")
    #                     elif torch.is_tensor(first_result):
    #                         # For PyTorch tensors
    #                         if all(torch.equal(first_result, results[key]) for results in results_list):
    #                             print(f"✅ {key}: Wyniki są identyczne.")
    #                         else:
    #                             print(f"❌ {key}: Różnice w wynikach!")
    #                     else:
    #                         # For regular floats or other values
    #                         if all(first_result == results[key] for results in results_list):
    #                             print(f"✅ {key}: Wyniki są identyczne.")
    #                         else:
    #                             print(f"❌ {key}: Różnice w wynikach!")
    #                 except Exception as e:
    #                     print(f"Błąd podczas porównania {key}: {e}")
    #     except Exception as e:
    #         print(f"Błąd podczas testów: {e}")
    #         return None
    #
    #     return results

    @staticmethod
    def is_flag_embedding_model(target_dir: str, model_name: str):
        """Sprawdza, jakie typy osadzeń obsługuje model oraz przeprowadza testowe wyszukiwanie."""
        # Domyślnie ustawiamy na False – będziemy zmieniać na True, jeśli wyniki zgadzają się w obu iteracjach
        results = {"dense": False, "sparse": False, "colbert": False}

        question = ["Where is number one"]
        text = [
            "one, two, three",
            "four, five, six"
        ]

        def run_embedding_tests(iteration: int):
            try:
                if iteration > 0:
                    safe_model_dir: str = str(model_name.replace("/", "_"))
                    target_temp_dir = str(os.path.join(MODEL_FOLDER, 'temp', safe_model_dir))
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
                    print('Dense embeddings')
                    embeddings_1 = model.encode(question, batch_size=12, max_length=8192)['dense_vecs']
                    embeddings_2 = model.encode(text)['dense_vecs']
                    dense_similarity = embeddings_1 @ embeddings_2.T
                    output['dense_similarity'] = dense_similarity
                    print(f'Dense_similarity: {dense_similarity}')
                except Exception as e:
                    print(f"❌ Dense: Błąd - {e}")

                # Sparse embeddings
                try:
                    print('Sparse embeddings')
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
                    output['tokens_from_text'] = tokens_from_text

                    # compute the scores via lexical matching
                    lexical_scores_1 = model.compute_lexical_matching_score(
                        question_encoded['lexical_weights'][0],
                        text_encoded['lexical_weights'][0]
                    )
                    lexical_scores_2 = model.compute_lexical_matching_score(
                        question_encoded['lexical_weights'][0],
                        text_encoded['lexical_weights'][1]
                    )
                    print(f'lexical_scores_1: {lexical_scores_1}')
                    print(f'lexical_scores_2: {lexical_scores_2}')
                    output['lexical_scores_1'] = lexical_scores_1
                    output['lexical_scores_2'] = lexical_scores_2
                except Exception as e:
                    print(f"❌ Sparse: Błąd - {e}")

                # ColBERT embeddings
                try:
                    print('ColBERT embeddings')
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
                    print(f'colbert_score_1: {colbert_score_1}')
                    print(f'colbert_score_2: {colbert_score_2}')
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
                print(results_list[-1])

            # Jeżeli w obu iteracjach (results_list[0], results_list[1]) mamy dane, to je porównujemy
            if results_list[0] and results_list[1]:

                # 1. DENSE
                #   - sprawdzamy dense_similarity
                if (
                        "dense_similarity" in results_list[0]
                        and "dense_similarity" in results_list[1]
                ):
                    # Sprawdzamy czy wartości są takie same (np. np.array_equal)
                    try:
                        if np.array_equal(
                                results_list[0]["dense_similarity"],
                                results_list[1]["dense_similarity"]
                        ):
                            results["dense"] = True
                        else:
                            results["dense"] = False
                    except Exception as e:
                        print(f"Błąd porównania dense_similarity: {e}")
                        results["dense"] = False

                # 2. SPARSE
                #   - sprawdzamy tokens_from_question, tokens_from_text, lexical_scores_1, lexical_scores_2
                #     (wystarczy, że jedna z tych rzeczy się nie zgadza, to uznajemy że model nie wspiera w pełni Sparse)
                sparse_ok = True
                for sparse_key in ["tokens_from_question", "tokens_from_text",
                                   "lexical_scores_1", "lexical_scores_2"]:
                    if sparse_key in results_list[0] and sparse_key in results_list[1]:
                        try:
                            # tokens_from_question / tokens_from_text to listy stringów,
                            # lexical_scores_x może być float albo int
                            # Zakładamy, że wystarczy == lub np.array_equal w zależności od typu
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
                                # Porównanie list / stringów
                                if results_list[0][sparse_key] != results_list[1][sparse_key]:
                                    sparse_ok = False
                        except Exception as e:
                            print(f"Błąd porównania {sparse_key}: {e}")
                            sparse_ok = False
                    else:
                        sparse_ok = False
                results["sparse"] = sparse_ok

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
                results["colbert"] = colbert_ok

            # Po porównaniu wypiszmy, które embeddingi są obsługiwane
            print("\n--- Podsumowanie obsługiwanych embeddingów ---")
            for emb_type in ["dense", "sparse", "colbert"]:
                if results[emb_type]:
                    print(f"✅ Model obsługuje: {emb_type}")
                else:
                    print(f"❌ Model nie obsługuje: {emb_type}")

        except Exception as e:
            print(f"Błąd podczas testów: {e}")
            return None

        return results

