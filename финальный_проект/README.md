# Агент анализ жалоб покупателей Wildberries

## Быстрый старт

```bash
pip install -r requirements.txt
python prepare_input.py     
python eval.py              
```

## Структура проекта

```
├── prepare_input.py   — стриминг и фильтрация данных из HuggingFace
├── schema.py          — Pydantic-модели и field_validator-ы
├── rag.py             — индекс Chroma + retrieval через MiniLM
├── agents.py          — три агента: Analyst, Writer, Judge
├── hallucination.py   — проверка ghost-цитат и выдуманных фактов
├── pipeline.py        — Map-Reduce оркестратор
├── eval.py            — тест-сет (30 отзывов), метрики правильности и пути
├── llm_client.py      — OpenAI-совместимый клиент с JSON-режимом
├── input/
│   ├── reviews.jsonl       — основной корпус
│   ├── reviews_eval.jsonl  — отложенный тест-сет
│   └── chroma_db/          — векторный индекс (создаётся автоматически)
└── output/
    ├── eval_results.jsonl  — полные результаты eval
    └── eval_table.csv      — сводная таблица метрик
```

## Техники курса

| Техника | Файл |
|---|---|
| IE + аспектный анализ + Map-Reduce | `agents.py` (Analyst), `pipeline.py` |
| RAG | `rag.py`, `agents.py` (Writer) |
| Мультиагент | `agents.py`, `pipeline.py` |
| LLM-as-judge | `agents.py` |
| Структурированный вывод + field_validator | `schema.py` |
| Проверка галлюцинаций | `hallucination.py` |
