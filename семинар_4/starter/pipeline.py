"""
Наивный RAG: ChromaDB + OpenAI, fixed-size chunking, только dense-поиск.

Команды:
    python pipeline.py ingest
    python pipeline.py ask "Кто жаловался на push-уведомления?"

TODO для семинара:
    Блок 3, Фикс 1 — заменить фиксированные чанки на рекурсивные по абзацам
    Блок 3, Фикс 2 — обернуть ответ в Pydantic RAGAnswer
    Блок 3, Фикс 3 — добавить BM25-гибрид через rank-bm25 и RRF
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_client import get_model, make_client, make_raw_client
from rank_bm25 import BM25Okapi
from schema import RAGAnswer

# Блок 1 — наивный RAG: ответ модели идёт обычным текстом
client = make_client()
MODEL = get_model()
chroma = chromadb.PersistentClient(path="./chroma_db")

TOP_K = 5
FIXED_SIZE = 2000

DATA_DIR = Path(__file__).parent / "wiki_data"
BM25_CACHE_FIXED = Path(__file__).parent / "bm25_cache_fixed.json"
BM25_CACHE_REC   = Path(__file__).parent / "bm25_cache_rec.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)

# ─────────────────────────────────────────────────
# Chroma + эмбеддер
# ─────────────────────────────────────────────────

print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Эмбеддер готов за {time.time() - _t_embed:.1f}с", flush=True)

collection_fixed = chroma.get_or_create_collection(
    name="history_fixed",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)

collection_rec = chroma.get_or_create_collection(
    name="history_recursive",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)

# ─────────────────────────────────────────────────
# Токенизация
# ─────────────────────────────────────────────────

def tokenize_ru(text: str):
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text_fixed(text: str, size: int = FIXED_SIZE) -> list[str]:
    return [text[i: i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def chunk_text_recursive(text: str) -> list[str]:
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def _clear_collection(col):
    existing = col.get()
    if existing["ids"]:
        col.delete(ids=existing["ids"])


def ingest():
    _clear_collection(collection_fixed)
    _clear_collection(collection_rec)

    all_chunks_rec = []
    all_ids_rec = []
    all_chunks_fixed = []
    all_ids_fixed = []


    # Стратегия 1 - разбиение на фиксированные чанки
    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text_fixed(text)
        ids    = [f"{f.stem}__fixed_{i}" for i in range(len(chunks))]
        metas  = [{"source": f.stem, "strategy": "fixed"} for _ in chunks]
        collection_fixed.add(documents=chunks, ids=ids, metadatas=metas)
        all_chunks_fixed.extend(chunks)
        all_ids_fixed.extend(ids)
        print(f"  {f.stem}: {len(chunks)} чанков")

    # Стратегия 2 - рекурсивное разбиение
    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text_recursive(text)
        ids = [f"{f.stem}__rec_{i}" for i in range(len(chunks))]
        metas = [{"source": f.stem, "strategy": "recursive"} for _ in chunks]
        collection_rec.add(documents=chunks, ids=ids, metadatas=metas)
        all_chunks_rec.extend(chunks)
        all_ids_rec.extend(ids)
        print(f"  {f.stem}: {len(chunks)} чанков")

    for cache_path, ids_list, chunks_list in [
        (BM25_CACHE_FIXED, all_ids_fixed, all_chunks_fixed),
        (BM25_CACHE_REC, all_ids_rec, all_chunks_rec),
    ]:
        bm25_data = {"ids": ids_list, "tokens": [tokenize_ru(c) for c in chunks_list], "texts": chunks_list}
        cache_path.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")

def _load_bm25(cache_path: Path):
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]

def dense_retrieve(query: str, collection, k: int = TOP_K) -> dict:
    return collection.query(query_texts=[query], n_results=k)


def hybrid_retrieve(query: str, collection, k: int = 5, top: int = 15, c: int = 60) -> dict:
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    cache_path = BM25_CACHE_FIXED if collection.name == "history_fixed" else BM25_CACHE_REC
    bm25, bm25_ids, bm25_texts = _load_bm25(cache_path)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)

    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[
        :top
    ]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    # top-k списка
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    # достаем тексты по id
    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id[i] for i in top_ids]]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по истории России. "
        "Опирайся ТОЛЬКО на контекст ниже. Если в контексте нет ответа — "
        "скажи об этом прямо.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )


def ask(query: str):
    print("Гибридный поиск...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, collection_rec, k=5)
    ids = hits["ids"][0]
    print(f"  нашёл {len(ids)} чанков за {time.time() - t0:.1f}с: {', '.join(ids)}", flush=True)

    prompt = build_prompt(query, hits)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print("Найденные фрагменты:")
    for cid, doc in zip(hits["ids"][0], hits["documents"][0]):
        print(f"\n  [{cid}]\n  {doc[:200]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py {ingest|ask} [вопрос]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
