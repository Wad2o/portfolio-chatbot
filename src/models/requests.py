from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    sessionId: str | None = Field(default=None, max_length=128)
    # UI language hint from the page; the message language takes priority.
    locale: str | None = Field(default=None, max_length=10)
