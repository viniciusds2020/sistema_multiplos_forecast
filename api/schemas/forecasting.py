"""Pydantic schemas for forecasting endpoints."""

from pydantic import BaseModel, Field


class IndividualForecastRequest(BaseModel):
    sku_id: str
    models: list[str] = Field(default=["XGBoost", "LightGBM", "AutoARIMA"])
    horizon: int = Field(default=30, ge=7, le=90)
    test_days: int = Field(default=60, ge=30, le=120)


class AggregatedForecastRequest(BaseModel):
    models: list[str] = Field(default=["XGBoost", "LightGBM", "AutoARIMA"])
    horizon: int = Field(default=30, ge=7, le=90)
    test_days: int = Field(default=60, ge=30, le=120)
    metric: str = Field(default="pearson", pattern="^(pearson|euclidean|dtw)$")
    weight_method: str = Field(default="rolling", pattern="^(rolling|static)$")
