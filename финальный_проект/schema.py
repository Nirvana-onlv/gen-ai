from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

class Aspect(str, Enum):
    QUALITY      = "качество_товара"
    DELIVERY     = "доставка"
    PACKAGING    = "упаковка"
    COMPLIANCE   = "соответствие_описанию"
    PRICE        = "цена"
    SERVICE      = "сервис_продавца"
    OTHER        = "другое"


class Sentiment(str, Enum):
    POSITIVE = "позитивный"
    NEGATIVE = "негативный"
    MIXED    = "смешанный"


class ReplyTone(str, Enum):
    APOLOGETIC   = "извинительный"
    INFORMATIVE  = "информационный"
    EMPATHETIC   = "сочувственный"
    SOLUTION     = "решение_проблемы"


# ---------------------------------------------------------------------------
# Входные данные
# ---------------------------------------------------------------------------

class ReviewInput(BaseModel):
    rating: int    = Field(description="Оценка покупателя от 1 до 5")
    text:   str    = Field(description="Текст отзыва")
    answer: str    = Field(description="Ответ продавца")
    color:  str    = Field(default="", description="Цвет/вариант товара")

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: int) -> int:
        """Оценка должна быть в диапазоне 1–5."""
        if not (1 <= v <= 5):
            raise ValueError(f"rating должен быть 1–5, получено {v}")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Отзыв пользователя не должен быть пустым"""
        if not v.strip():
            raise ValueError("text не может быть пустым")
        return v.strip()

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        """Ответ продавца не может быть пустым"""
        if not v.strip():
            raise ValueError("Ответ продавца не может быть пустым")
        return v.strip()


# ---------------------------------------------------------------------------
# Agent Analyst → IE + аспектный анализ
# ---------------------------------------------------------------------------

class AspectEntry(BaseModel):
    aspect:  Aspect = Field(description="Категория аспекта")
    problem: str    = Field(description="Конкретная проблема, упомянутая в отзыве (1–2 предложения)")
    quote:   str    = Field(description="Дословная цитата из отзыва, подтверждающая проблему")

    @field_validator("problem", "quote")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Поле не может быть пустым")
        return v.strip()


class AspectAnalysis(BaseModel):
    """
    Извлекает аспекты, определяет тональность и проверяет
    согласованность оценки с текстом.
    """
    sentiment:          Sentiment         = Field(description="Общая тональность отзыва")
    aspects:            list[AspectEntry] = Field(description="Список выявленных аспектов")
    summary:            str               = Field(description="Краткое резюме проблем")
    rating_mismatch:    bool              = Field(
        description="True если тональность текста не соответствует числовой оценке "
                    "(например, позитивный текст при оценке 1)"
    )
    suggested_rating:   Optional[int]     = Field(
        default=None,
        description="Предлагаемый агентом рейтинг на основе текста"
    )

    @field_validator("aspects")
    @classmethod
    def at_least_one_aspect(cls, v: list[AspectEntry]) -> list[AspectEntry]:
        if not v:
            raise ValueError("Должен быть выявлен хотя бы один аспект")
        return v

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary не может быть пустым")
        return v.strip()

    @field_validator("suggested_rating")
    @classmethod
    def suggested_rating_in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 5):
            raise ValueError(f"suggested_rating должен быть 1–5, получено {v}")
        return v

    @model_validator(mode="after")
    def mismatch_requires_suggestion(self) -> "AspectAnalysis":
        """Если есть расхождение — должна быть предложена альтернативная оценка."""
        if self.rating_mismatch and self.suggested_rating is None:
            raise ValueError(
                "Если rating_mismatch=True, поле suggested_rating обязательно"
            )
        return self


# ---------------------------------------------------------------------------
# Agent Writer → ответ продавца
# ---------------------------------------------------------------------------

class SellerReply(BaseModel):
    """
    Результат работы Agent Writer.
    Генерирует улучшенный ответ продавца на основе аспектов и RAG-контекста.
    """
    reply_text:      str            = Field(description="Текст ответа продавца (3–7 предложений)")
    tone:            ReplyTone      = Field(description="Тональность ответа")
    addressed_aspects: list[Aspect] = Field(description="Какие аспекты из анализа учтены в ответе")
    rag_examples_used: int          = Field(description="Сколько RAG-примеров использовано (0–3)")

    @field_validator("reply_text")
    @classmethod
    def reply_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reply_text не может быть пустым")
        return v.strip()

    @field_validator("rag_examples_used")
    @classmethod
    def rag_examples_in_range(cls, v: int) -> int:
        if not (0 <= v <= 3):
            raise ValueError(f"rag_examples_used должен быть 0–3, получено {v}")
        return v

    @field_validator("addressed_aspects")
    @classmethod
    def at_least_one_addressed(cls, v: list[Aspect]) -> list[Aspect]:
        if not v:
            raise ValueError("Ответ должен затрагивать хотя бы один аспект")
        return v


# ---------------------------------------------------------------------------
# Проверка галлюцинаций
# ---------------------------------------------------------------------------

class GhostItem(BaseModel):
    """Одна найденная галлюцинация в ответе продавца."""
    fragment:    str = Field(description="Фрагмент из ответа, который является галлюцинацией")
    explanation: str = Field(description="Почему это галлюцинация — чего не было в оригинальном отзыве")


class HallucinationReport(BaseModel):
    """Результат проверки ghost-цитат и выдуманных фактов в ответе продавца."""
    has_hallucinations: bool            = Field(description="Есть ли галлюцинации")
    ghost_items:        list[GhostItem] = Field(default_factory=list, description="Список найденных галлюцинаций")
    hallucination_count: int            = Field(description="Количество найденных галлюцинаций")

    @model_validator(mode="after")
    def count_matches_items(self) -> "HallucinationReport":
        """Счётчик должен совпадать с длиной списка."""
        if self.hallucination_count != len(self.ghost_items):
            raise ValueError(
                f"hallucination_count={self.hallucination_count} не совпадает "
                f"с len(ghost_items)={len(self.ghost_items)}"
            )
        if self.has_hallucinations and not self.ghost_items:
            raise ValueError("has_hallucinations=True, но ghost_items пуст")
        return self


# ---------------------------------------------------------------------------
# Agent Judge → оценка качества ответа
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """
    Результат работы Agent Judge (LLM-as-judge).
    Оценивает сгенерированный ответ продавца по трём осям.
    """
    politeness_score:    int = Field(description="Вежливость ответа: 1–5")
    specificity_score:   int = Field(description="Конкретность — насколько ответ адресует проблему: 1–5")
    resolution_score:    int = Field(description="Решает ли ответ проблему покупателя: 1–5")
    overall_score:       int = Field(description="Итоговая оценка качества ответа: 1–5")
    reasoning:           str = Field(description="Обоснование оценки (2–4 предложения)")
    better_than_original: bool = Field(
        description="True если сгенерированный ответ лучше оригинального ответа продавца из датасета"
    )

    @field_validator("politeness_score", "specificity_score", "resolution_score", "overall_score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"Оценка должна быть 1–5, получено {v}")
        return v

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning не может быть пустым")
        return v.strip()

    @model_validator(mode="after")
    def overall_consistent(self) -> "JudgeVerdict":
        """
        Итоговая оценка не должна сильно расходиться со средним по трём осям.
        Допустимое отклонение — ±1.
        """
        avg = (self.politeness_score + self.specificity_score + self.resolution_score) / 3
        if abs(self.overall_score - avg) > 1.5:
            raise ValueError(
                f"overall_score={self.overall_score} слишком далёк от среднего "
                f"по осям ({avg:.1f}). Пересмотри оценку."
            )
        return self


# ---------------------------------------------------------------------------
# Итоговая запись
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """
    Полная итоговая запись для output/results.json.
    Агрегирует все этапы пайплайна.
    """
    input:                ReviewInput         = Field(description="Исходный отзыв")
    analysis:             AspectAnalysis      = Field(description="Результат Analyst")
    reply:                SellerReply         = Field(description="Результат Writer")
    hallucination_report: HallucinationReport = Field(description="Результат проверки галлюцинаций")
    verdict:              JudgeVerdict        = Field(description="Результат Judge")


# ---------------------------------------------------------------------------
# Map-Reduce: резюме по товару (несколько отзывов → один отчёт)
# ---------------------------------------------------------------------------

class AspectSummaryEntry(BaseModel):
    """Сводка по одному аспекту на основе нескольких отзывов."""
    aspect:          Aspect    = Field(description="Аспект")
    mention_count:   int       = Field(description="Сколько отзывов упомянули этот аспект")
    common_problems: list[str] = Field(description="Топ-3 повторяющиеся проблемы")

    @field_validator("mention_count")
    @classmethod
    def positive_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("mention_count должен быть ≥1")
        return v


class ProductSummary(BaseModel):
    """
    Результат Reduce-шага: сводный отчёт по товару (nm_id)
    на основе N отзывов — multi-document summary.
    """
    nm_id:            int                    = Field(description="Артикул товара")
    review_count:     int                    = Field(description="Сколько отзывов вошло в сводку")
    avg_rating:       float                  = Field(description="Средняя оценка по этим отзывам")
    dominant_sentiment: Sentiment            = Field(description="Преобладающая тональность")
    aspect_summaries: list[AspectSummaryEntry] = Field(description="Сводка по аспектам")
    executive_summary: str                   = Field(description="Итоговое резюме для продавца (3–5 предложений)")

    @field_validator("review_count")
    @classmethod
    def positive_review_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("review_count должен быть ≥1")
        return v

    @field_validator("avg_rating")
    @classmethod
    def avg_rating_in_range(cls, v: float) -> float:
        if not (1.0 <= v <= 5.0):
            raise ValueError(f"avg_rating должен быть 1.0–5.0, получено {v}")
        return v