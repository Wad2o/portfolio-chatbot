from __future__ import annotations

from pydantic import BaseModel, Field

from ..security.validation import clean_text

MAX_LIST_ITEMS = 15


class QualificationState(BaseModel):
    """Structured project info, kept apart from chat text so it can later
    pre-fill the pricing configurator."""

    projectType: str | None = None
    developmentType: str | None = None
    numberOfPages: int | None = None
    features: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    additionalNotes: str | None = None

    def is_empty(self) -> bool:
        return not (
            self.projectType
            or self.developmentType
            or self.numberOfPages
            or self.features
            or self.integrations
            or self.additionalNotes
        )

    def merge(
        self,
        update: object,
        allowed_project_types: set[str],
        allowed_development_types: set[str],
    ) -> "QualificationState":
        """Return a new state with the model's proposed update applied.

        Every field is validated server-side; anything unknown is ignored so a
        confused (or manipulated) model cannot corrupt the state."""
        if not isinstance(update, dict):
            return self
        data = self.model_dump()

        project_type = update.get("projectType")
        if isinstance(project_type, str) and project_type in allowed_project_types:
            data["projectType"] = project_type

        development = update.get("developmentType")
        if isinstance(development, str) and development in allowed_development_types:
            data["developmentType"] = development

        pages = update.get("numberOfPages")
        if isinstance(pages, int) and not isinstance(pages, bool) and 1 <= pages <= 200:
            data["numberOfPages"] = pages

        for key in ("features", "integrations"):
            values = update.get(key)
            if not isinstance(values, list):
                continue
            merged = list(data[key])
            seen = {v.lower() for v in merged}
            for value in values:
                if not isinstance(value, str):
                    continue
                cleaned = clean_text(value, 60)
                if cleaned and cleaned.lower() not in seen and len(merged) < MAX_LIST_ITEMS:
                    merged.append(cleaned)
                    seen.add(cleaned.lower())
            data[key] = merged

        notes = update.get("additionalNotes")
        if isinstance(notes, str):
            cleaned = clean_text(notes, 500)
            if cleaned:
                data["additionalNotes"] = cleaned

        return QualificationState(**data)
