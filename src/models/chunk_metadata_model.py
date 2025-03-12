import os


class ChunkMetadataModel:
    def __init__(self, *, source: str, hash_id: str, fragment_id: int, characters_count: int, tiktoken_tokens_count: int, model_tokenizer_token_count: int):
        self.source: str = os.path.basename(source)
        self.hash_id = hash_id
        self.fragment_id: int = fragment_id
        self.characters_count: int = characters_count
        self.tiktoken_tokens_count: int = tiktoken_tokens_count
        self.model_tokenizer_token_count: int = model_tokenizer_token_count

    def to_dict(self):
        return {
            "source": self.source,
            "hash_id": self.hash_id,
            "fragment_id": self.fragment_id,
            "characters_count": self.characters_count,
            "tiktoken_tokens_count": self.tiktoken_tokens_count,
            "model_tokenizer_token_count": self.model_tokenizer_token_count
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            source=data.get("source", ""),  # Placeholder, requires actual path handling
            hash_id=data.get("hash_id", ""),
            fragment_id=data.get("fragment_id", 0),
            characters_count=data.get("characters_count", 0),
            tiktoken_tokens_count=data.get("tiktoken_tokens_count", 0),
            model_tokenizer_token_count=data.get("model_tokenizer_token_count", 0)
        )
