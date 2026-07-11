from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    chat: bool = True
    vision: bool = False
    embedding: bool = False
    heavy: bool = False


def capabilities_for(model: str) -> ModelCapabilities:
    name = model.lower()
    return ModelCapabilities(
        chat="embedding" not in name,
        vision=any(token in name for token in ("-vl", "vision", "llava")),
        embedding="embedding" in name or "embed" in name,
        heavy=any(token in name for token in ("26b", "70b", "72b", "32b")),
    )
