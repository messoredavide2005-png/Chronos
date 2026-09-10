# config.py
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

DATA_DIR = "data"
DB_DIR = "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# Velocizza le risposte (meno token = meno tempo)
MAX_TOKENS = 500  # Limita la risposta a ~500 parole