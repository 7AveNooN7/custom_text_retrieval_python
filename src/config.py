import os

# Pobranie ścieżki głównego katalogu projektu (gdzie znajduje się config.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 📌 Ścieżka do katalogu na bazy wektorowe w folderze `repository/chroma_db`
CHROMA_DB_FOLDER = os.path.join(BASE_DIR, "..", "repository", "chroma_db")
os.makedirs(CHROMA_DB_FOLDER, exist_ok=True)

# 📌 Ścieżka do katalogu na bazy wektorowe w folderze `repository/chroma_db`
WEAVIATE_DB_FOLDER = os.path.join(BASE_DIR, "..", "repository", "weaviate_db")
os.makedirs(WEAVIATE_DB_FOLDER, exist_ok=True)

# 📌 Ścieżka do katalogu na LanceDB
LANCE_DB_FOLDER = os.path.join(BASE_DIR, "..", "repository", "lance_db")
os.makedirs(LANCE_DB_FOLDER, exist_ok=True)

# 📌 Ścieżka do katalogu na LanceDB
SQLITE_FOLDER = os.path.join(BASE_DIR, "..", "repository", "sqlite_db")
os.makedirs(SQLITE_FOLDER, exist_ok=True)

# 📌 Folder cache do przechowywania/odczytu modeli w folderze `repository/downloaded_models`
MODEL_FOLDER = os.path.join(BASE_DIR, "..", "repository", "downloaded_models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 📌 Folder do przechowywania utworzonych .txt
TXT_FOLDER = os.path.join(BASE_DIR, "..", "repository", "text_files")
os.makedirs(TXT_FOLDER, exist_ok=True)

# Inne ustawienia konfiguracyjne:
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 400


# PDF CHAPTERS TO AVOID
EXCLUDED_TITLES = ["Appendix", "Legacy", "References", "EULA", "Contents", "Preface",
                    "Acknowledgements", "Contributor", "Companion Website", "Website", "Abbreviations", "Cover", "Title Page",
                    "Copyright", "Copyright Page", "Front Cover", "Front Matter", "Disclaimer", "Blank Page", "About the Author",
                    "Author", "Credits", "Imprint", "Publisher's Note", "Editorial Note", "Introduction", "Dedication", "Prologue",
                    "Foreword", "Table of Contents", "Contents", "Index", "Glossary"
                   ]

EXCLUDED_STRUCTURAL_ELEMENTS = ["Figure", "Table"]


# in grobid/grobid-home/config/grobid.yaml set the parameter concurrency to your number of available threads
# at server side divided by 2 (8 threads available, set concurrency to 4)

# keep the concurrency at the client (number of simultaneous calls) at the same level as the concurrency parameter at server side,
# for instance if the server has 16 threads, use a concurrency of 8 and the client concurrency at 8 (it's the option n in the clients)
GROBID_MAX_WORKERS = 8

GROBID_URL = "http://127.0.0.1:8070"

