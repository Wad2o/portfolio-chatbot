from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .state import QualificationState

ActionType = Literal["navigate", "configure", "contact"]


class Suggestion(BaseModel):
    label: str
    value: str


class Action(BaseModel):
    type: ActionType
    label: str
    url: str | None = None
    data: dict[str, str] | None = None


class Metadata(BaseModel):
    intent: str
    recommendedOffer: str | None = None
    qualificationStage: str = "none"
    language: str = "en"


class ChatResponse(BaseModel):
    sessionId: str
    messageId: str
    message: str
    suggestions: list[Suggestion] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    metadata: Metadata
    qualification: QualificationState
