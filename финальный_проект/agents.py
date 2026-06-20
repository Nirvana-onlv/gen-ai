"""
agents.py
---------
Три агента мультиагентного пайплайна:
  Analyst  — извлекает аспекты и тональность из отзыва (IE + аспектный анализ)
  Writer   — генерирует ответ продавца на основе аспектов + RAG-контекста
  Judge    — оценивает качество ответа (LLM-as-judge)
"""
from __future__ import annotations

from llm_client import make_client, get_model
from rag import RetrieverRAG, RAGExample
from schema import (
    ReviewInput,
    AspectAnalysis,
    SellerReply,
    JudgeVerdict,
)

_CLIENT = make_client()


# ---------------------------------------------------------------------------
# Agent Analyst
# ---------------------------------------------------------------------------

class Analyst:
    """
    Извлекает аспекты, тональность и резюме из отзыва.
    """

    SYSTEM = """Ты аналитик отзывов интернет-магазина Wildberries.
Твоя задача — внимательно прочитать отзыв покупателя и структурированно извлечь:
- все упомянутые аспекты (качество товара, доставка, упаковка, соответствие описанию, цена, сервис)
- конкретную проблему по каждому аспекту
- дословную цитату из отзыва, подтверждающую проблему
- общую тональность
- есть ли расхождение между числовой оценкой и текстом

Будь точным: не придумывай проблемы которых нет в тексте, цитируй дословно."""

    def run(self, review: ReviewInput, max_retries: int = 2) -> AspectAnalysis:
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": (
                f"Оценка покупателя: {review.rating} из 5\n\n"
                f"Текст отзыва:\n{review.text}"
            )},
        ]

        return _CLIENT.chat.completions.create(
            model=get_model(),
            messages=messages,
            response_model=AspectAnalysis,
            max_retries=max_retries,
        )


# ---------------------------------------------------------------------------
# Agent Writer
# ---------------------------------------------------------------------------

class Writer:
    """
    Генерирует улучшенный ответ продавца.
    """

    SYSTEM = """Ты опытный специалист службы поддержки интернет-магазина Wildberries.
Твоя задача — написать вежливый, конкретный и полезный ответ на отзыв покупателя.

Правила:
- Обращайся к каждой проблеме из анализа
- Не обещай того чего не было в исходном отзыве или примерах
- Используй тон из примеров похожих ответов если они есть
- Ответ 3–7 предложений
- Пиши от лица магазина, не от лица конкретного человека"""

    def __init__(self, rag: RetrieverRAG) -> None:
        self._rag = rag

    def _format_examples(self, examples: list[RAGExample]) -> str:
        if not examples:
            return ""
        lines = ["Примеры похожих ответов продавцов:\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"[Пример {i}] (оценка {ex.rating}, схожесть {1 - ex.distance:.2f})")
            lines.append(f"Жалоба: {ex.review_text[:200]}")
            lines.append(f"Ответ:  {ex.answer_text[:300]}")
            lines.append("")
        return "\n".join(lines)

    def run(
            self,
            review: ReviewInput,
            analysis: AspectAnalysis,
            k: int = 3,
            distance_threshold: float = 0.8,
            max_retries: int = 2,
    ) -> SellerReply:
        raw_examples = self._rag.retrieve(review.text, k=k)
        examples = [ex for ex in raw_examples if ex.distance < distance_threshold]

        aspects_text = "\n".join(
            f"- {e.aspect.value}: {e.problem}" for e in analysis.aspects
        )

        user_content = (
                f"Отзыв покупателя (оценка {review.rating}):\n{review.text}\n\n"
                f"Выявленные проблемы:\n{aspects_text}\n\n"
                f"Общее резюме: {analysis.summary}\n\n"
                + self._format_examples(examples)
        )

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": user_content},
        ]

        result = _CLIENT.chat.completions.create(
            model=get_model(),
            messages=messages,
            response_model=SellerReply,
            max_retries=max_retries,
        )

        object.__setattr__(result, "rag_examples_used", len(examples))
        return result


# ---------------------------------------------------------------------------
# Agent Judge
# ---------------------------------------------------------------------------

class Judge:
    """
    Оценивает качество сгенерированного ответа продавца.
    """

    SYSTEM = """Ты эксперт по качеству клиентского сервиса.
Оцени ответ продавца на отзыв покупателя по трём осям (каждая 1–5):

1. Вежливость (politeness_score) — тон, уважение, отсутствие формализма
2. Конкретность (specificity_score) — адресует ли ответ конкретные проблемы из отзыва
3. Решение проблемы (resolution_score) — предлагает ли реальный выход или просто извиняется

Итоговая оценка (overall_score) — взвешенное суждение по всем трём осям.
Также сравни сгенерированный ответ с оригинальным ответом продавца из датасета.

Будь строгим: оценка 5 — только если ответ действительно образцовый."""

    def run(
            self,
            review: ReviewInput,
            analysis: AspectAnalysis,
            reply: SellerReply,
            max_retries: int = 2,
    ) -> JudgeVerdict:
        aspects_text = "\n".join(
            f"- {e.aspect.value}: {e.problem}" for e in analysis.aspects
        )

        user_content = (
            f"Отзыв покупателя (оценка {review.rating}):\n{review.text}\n\n"
            f"Проблемы из анализа:\n{aspects_text}\n\n"
            f"Оригинальный ответ продавца (из датасета):\n{review.answer}\n\n"
            f"Сгенерированный ответ:\n{reply.reply_text}"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": user_content},
        ]

        return _CLIENT.chat.completions.create(
            model=get_model(),
            messages=messages,
            response_model=JudgeVerdict,
            max_retries=max_retries,
        )