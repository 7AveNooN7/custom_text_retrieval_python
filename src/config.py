import os

# Pobranie ścieżki głównego katalogu projektu (gdzie znajduje się config.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 📌 Ścieżka do katalogu na bazy wektorowe w folderze `repository/chroma_db`
CHROMA_DB_FOLDER = os.path.join(BASE_DIR, "..", "repository", "chroma_db")
os.makedirs(CHROMA_DB_FOLDER, exist_ok=True)

# 📌 Ścieżka do katalogu na bazy wektorowe w folderze `repository/chroma_db`
WEAVIATE_DB_FOLDER = os.path.join(BASE_DIR, "..", "repository", "weaviate_db")
os.makedirs(CHROMA_DB_FOLDER, exist_ok=True)

# 📌 Folder cache do przechowywania/odczytu modeli w folderze `repository/downloaded_models`
MODEL_FOLDER = os.path.join(BASE_DIR, "..", "repository", "downloaded_models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Inne ustawienia konfiguracyjne:
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 400