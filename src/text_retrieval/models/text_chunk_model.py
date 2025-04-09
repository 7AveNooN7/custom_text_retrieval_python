import hashlib



class TextChunkModel:
    def __init__(self, text_chunk: str, source_file_name: str, fragment_index: int):
        """
        Inicjalizuje obiekt klasy TextChunker z określonym rozmiarem fragmentu i nakładką.
        """

        self.text_chunk = text_chunk
        self.source_file_name = source_file_name
        self.fragment_index = fragment_index


    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int):
        step = chunk_size - chunk_overlap
        if step <= 0:
            raise ValueError("Parametr 'chunk_overlap' musi być mniejszy niż 'chunk_size'!")

        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i: i + chunk_size]
            chunks.append(chunk)
        return chunks

    def generate_id(text: str, filename: str, index: int) -> str:
        return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()