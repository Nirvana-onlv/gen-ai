"""
schema.py — общие Pydantic-схемы пайплайна
===========================================
Заполняется постепенно, по мере прохождения раундов. На старте — пусто.

Карта моделей по раундам:
  Раунд 1   — Concern, Participant
  Раунд 2   — AspectSentiment, ParticipantSentiment
  Раунд 2.5 — DiscoveredAspects (для autodiscovery)
  Раунд 3   — ChunkSummary, DiscussionSummary
  Раунд 3.5 — GroupSummary (для иерархического Map-Reduce)
  Раунд 5   — ActionVerdict, JudgeReport
  Раунд 7   — MultiDocSummary
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════
class Issue(BaseModel):
    category: Literal["battery", "camera", "performance", "design", "price", "software"]
    severity: int = Field(ge=1, le=5)
    quote: str


class Review(BaseModel):
    review_id: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    detalization: Literal["cursory", "detailed", "unclear"]
    issues: list[Issue]
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("detalization")
    @classmethod
    def validate_sentiment(cls, value):
        if value not in ["cursory", "detailed", "unclear"]:
            raise ValueError("sentiment should be one of 'cursory', 'detailed', 'unclear'")
        return value


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════
class AspectSentiment(BaseModel):
    aspect: Literal["battery", "camera", "performance", "design", "price", "software"]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    review_id: str
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════
class DiscoveredAspect(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicReview(BaseModel):
    review_id: str
    aspects: list[DynamicAspect]


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce-резюме
# ══════════════════════════════════════════════════════════
class ChunkSummary(BaseModel):
    review_ids: list[str]
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════
class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str

