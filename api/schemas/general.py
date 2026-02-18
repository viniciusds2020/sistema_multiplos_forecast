"""Pydantic schemas for general endpoints."""

from pydantic import BaseModel


class CacheClearResponse(BaseModel):
    status: str
