class DownloadedModelInfo:
    """Reprezentuje pobrany model z katalogu."""

    def __init__(self, *, model_name: str, folder_name: str):
        self.name = model_name  # Nazwa modelu
        self.folder_name = folder_name  # Nazwa katalogu, a nie pełna ścieżka

    @classmethod
    def from_json(cls, *, json_data: dict, folder_name: str):
        """Tworzy obiekt DownloadedModelInfo na podstawie danych JSON."""
        model_name = json_data.get("model_name", folder_name)  # Jeśli brak "model_name", używa nazwy katalogu
        return cls(model_name=model_name, folder_name=folder_name)

    def to_json(self) -> dict:
        """Zwraca obiekt w formacie JSON."""
        return {"name": self.name, "folder_name": self.folder_name}

    def __repr__(self):
        return f"DownloadedModelInfo(name='{self.name}', folder_name='{self.folder_name}')"
