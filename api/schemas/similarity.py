"""Pydantic schemas for similarity endpoints."""

from pydantic import BaseModel, Field


class SimilarityRequest(BaseModel):
    metric: str = Field(default="pearson", pattern="^(pearson|euclidean|dtw)$")
    auto_k: bool = True
    manual_k: int = Field(default=4, ge=2, le=8)
