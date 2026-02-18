"""Pydantic schemas for comparison endpoints."""

from pydantic import BaseModel, Field


class ComparisonRequest(BaseModel):
    models: list[str] = Field(default=["XGBoost", "LightGBM", "AutoARIMA"])
    sku_ids: list[str] = Field(default=[])
    horizon: int = Field(default=30, ge=7, le=90)
    test_days: int = Field(default=60, ge=30, le=120)
