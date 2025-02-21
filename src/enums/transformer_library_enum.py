from enum import Enum
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.enums.embedding_type_enum import EmbeddingType
from FlagEmbedding import FlagModel, BGEM3FlagModel


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
    def is_sentence_transformer_model(model_dir: str) -> dict:
        print('SentenceTransformers')
        """
        Sprawdza, czy model jest kompatybilny z Sentence-Transformers,
        wykonując test kodowania i wyszukiwania na podstawie podobieństwa kosinusowego.
        """
        try:
            # Ładowanie modelu
            model = SentenceTransformer(model_dir)

            # Przykładowe zdania do testu
            documents = [
                "To jest przykładowe zdanie numer jeden.",
                "Drugie zdanie jest inne, ale związane tematycznie.",
                "To zdanie nie ma nic wspólnego z poprzednimi.",
                "Jabłka są czerwone, a banany żółte."
            ]

            # Zdanie do wyszukania
            query = "Szukam czegoś związanego z pierwszym przykładem."

            # Kodowanie dokumentów i zapytania
            embeddings = model.encode([query] + documents, convert_to_tensor=True)

            # Obliczanie podobieństwa kosinusowego
            similarities = SentenceTransformer.util.cos_sim(embeddings[0], embeddings[1:])[0]

            print(f'similarities: {similarities}')

        except Exception as e:
            print(f"❌ Błąd: {e}")
            return {"status": "error", "reason": str(e)}

    @staticmethod
    def is_flag_embedding_model(model_name_or_path: str):
        """Sprawdza, jakie typy osadzeń obsługuje model oraz przeprowadza testowe wyszukiwanie."""
        results = {"dense": False, "sparse": False, "colbert": False}
        question = ["Where is number one"]
        text = [
            "one, two, three",
            "four, five, six"
        ]

        try:
            # Załaduj model przy użyciu klasy BGEM3FlagModel
            model = BGEM3FlagModel(model_name_or_path)

            # Sprawdź generowanie dense embeddings
            try:
                print('Dense embeddings')

                embeddings_1 = model.encode(question,
                                            batch_size=12,
                                            max_length=8192,
                                            # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                            )['dense_vecs']
                embeddings_2 = model.encode(text)['dense_vecs']
                similarity = embeddings_1 @ embeddings_2.T
                print(f'similarity: {similarity}')
            except Exception as e:
                print(f"❌ Dense: Błąd - {e}")
                results["dense"] = False

            # Sprawdź generowanie sparse embeddings
            try:
                print('Sparse embeddings')

                question_encoded = model.encode(question, return_dense=False, return_sparse=True, return_colbert_vecs=False)
                text_encoded = model.encode(text, return_dense=False, return_sparse=True, return_colbert_vecs=False)

                tokens = model.convert_id_to_token(question_encoded['lexical_weights'])
                print(f'tokens: {tokens}')

                # compute the scores via lexical mathcing
                lexical_scores_1 = model.compute_lexical_matching_score(question_encoded['lexical_weights'][0], text_encoded['lexical_weights'][0])
                print(f'lexical_scores_1: {lexical_scores_1}')
                lexical_scores_2 = model.compute_lexical_matching_score(question_encoded['lexical_weights'][0], text_encoded['lexical_weights'][1])
                print(f'lexical_scores_2: {lexical_scores_2}')
            except Exception as e:
                print(f"❌ Sparse: Błąd - {e}")
                results["sparse"] = False

            # Sprawdź generowanie ColBERT embeddings
            try:
                print('ColBERT embeddings')

                question_encoded = model.encode(question, return_dense=False, return_sparse=False, return_colbert_vecs=True)
                text_encoded = model.encode(text, return_dense=False, return_sparse=False, return_colbert_vecs=True)

                colbert_score_1 = model.colbert_score(question_encoded['colbert_vecs'][0], text_encoded['colbert_vecs'][0])
                colbert_score_2 = model.colbert_score(question_encoded['colbert_vecs'][0], text_encoded['colbert_vecs'][1])

                print(f'colbert_score_1: {colbert_score_1}')
                print(f'colbert_score_2: {colbert_score_2}')
            except Exception as e:
                print(f"❌ ColBERT: Błąd - {e}")
                results["colbert"] = False

        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            return None

        return results


