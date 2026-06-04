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

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════
class Concern(BaseModel):
    category: Literal["price", "speed", "ux", "support", "feature"]
    severity: int = Field(ge=1, le=5)
    quote: str


class Participant(BaseModel):
    name: str
    age: Optional[int] = None
    city: str
    occupation: str
    concerns: list[Concern]
    competitor_mentions: list[str] = Field(default_factory=list)


class MatchVerdict(BaseModel):
    matched: bool
    matched_index: int = Field(default=-1, description="номер жалобы или -1")
    reason: str = ""


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════
class AspectSentiment(BaseModel):
    aspect: Literal["price", "speed", "ux", "support", "feature"]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ParticipantSentiment(BaseModel):
    name: str
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════
class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicParticipant(BaseModel):
    name: str
    aspects: list[DynamicAspect]


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce-резюме
# ══════════════════════════════════════════════════════════
class ChunkSummary(BaseModel):
    speaker: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class DiscussionSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


# ══════════════════════════════════════════════════════════
# Раунд 3.5 — Иерархический Map-Reduce
# ══════════════════════════════════════════════════════════
class GroupSummary(BaseModel):
    speakers: list[str]
    themes: list[str] = Field(min_length=1, max_length=6)
    overall_sentiment: Literal["positive", "negative", "mixed"]


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


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка
# ══════════════════════════════════════════════════════════
class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_bank: dict[str, list[str]]
    overall_headline: str
