import functools
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")


@functools.lru_cache(maxsize=64)
def build_embeddings(s: str):
    return model.encode(s, normalize_embeddings=True)
