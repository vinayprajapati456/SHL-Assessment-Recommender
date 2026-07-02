"""Pydantic models for the SHL Assessment Recommender API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    messages: list[Message] = Field(
        ..., description="Full conversation history, alternating user/assistant"
    )


class Recommendation(BaseModel):
    """A single assessment recommendation."""

    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(
        ..., description="Abbreviated test type codes (e.g. 'K', 'P', 'A,S')"
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    reply: str = Field(..., description="The agent's natural language response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when gathering context; 1-10 items when recommending",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"


class CatalogItem(BaseModel):
    """Processed catalog assessment entry."""

    entity_id: str
    name: str
    link: str
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = "yes"
    adaptive: str = "no"
    description: str = ""
    keys: list[str] = Field(default_factory=list)
    test_type: str = ""
    search_text: str = ""  # precomputed text for retrieval
