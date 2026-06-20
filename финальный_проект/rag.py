"""
rag.py
------
RAG сохраняет текущие ответы продавцов на вопросы отзывы пользователей,
чтобы агент мог выступать как специалист поддержки, на основании уже выданных ответов.

Хранение данных будет осуществлять в ChromaDB
В качестве эмбеддинг модели используетсяMiniLM-L12-v2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

REVIEWS_PATH  = Path("input/reviews.jsonl")
CHROMA_DIR    = Path("chroma_db")
COLLECTION    = "seller_replies"
MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_K     = 3


# ---------------------------------------------------------------------------
# Результат retrieval
# ---------------------------------------------------------------------------

class RAGExample(NamedTuple):
    review_text: str
    answer_text: str
    rating:      int
    distance:    float


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class RetrieverRAG:
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION,
        model_name: str = MODEL_NAME,
    ) -> None:
        self._model = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_collection(collection_name)

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RAGExample]:
        """
        Найти top-k похожих пар (отзыв → ответ продавца) по тексту запроса.
        """
        embedding = self._model.encode(query, normalize_embeddings=True).tolist()

        results = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )

        examples: list[RAGExample] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            examples.append(RAGExample(
                review_text=meta.get("review_text", ""),
                answer_text=doc,
                rating=int(meta.get("rating", 0)),
                distance=float(dist),
            ))

        return examples


# ---------------------------------------------------------------------------
# Построение индекса
# ---------------------------------------------------------------------------

def build_index(
    reviews_path: Path = REVIEWS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> None:
    """
    Производим эмбеддинг поля answer, сохраняем в Chroma.

    Индексируем answer, а не текст отзыва, потому что на этапе retrieval
    мы ищем какой ответ продавца подходит к похожей жалобе.
    В метаданных сохраняем оригинальный текст отзыва — он нужен Writer
    как контекст.
    """
    if not reviews_path.exists():
        raise FileNotFoundError(
            f"Файл {reviews_path} не найден. "
        )

    model = SentenceTransformer(model_name)

    records: list[dict] = []
    with open(reviews_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas, embeddings = [], [], [], []

    answers = [r["answer"] for r in records]

    for batch_start in range(0, len(answers), batch_size):
        batch = answers[batch_start: batch_start + batch_size]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)

        for i, (vec, record) in enumerate(zip(vecs, records[batch_start: batch_start + batch_size])):
            idx = batch_start + i
            ids.append(str(idx))
            docs.append(record["answer"])
            metas.append({
                "review_text": record["text"][:500],
                "rating": record["rating"],
                "nm_id": record.get("nm_id", 0),
            })
            embeddings.append(vec.tolist())

        progress = min(batch_start + batch_size, len(answers))
        print(f"  {progress}/{len(answers)}")

    col.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings,
    )
