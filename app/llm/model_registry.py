from __future__ import annotations

import re
from dataclasses import dataclass


_PARAMETER_PATTERN = re.compile(r"(?<![a-z0-9])(?P<size>\d+(?:\.\d+)?)b(?![a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class ModelCapabilities:
    chat: bool = True
    vision: bool = False
    embedding: bool = False
    heavy: bool = False
    parameter_billions: float | None = None


def capabilities_for(model: str) -> ModelCapabilities:
    """Infer conservative local-model capabilities from an Ollama model name.

    Ollama model names are not a formal capability registry, so this function is
    deliberately conservative. It is used for resource admission, not as proof
    that a model supports a capability.
    """

    name = model.strip().lower()
    embedding = "embedding" in name or "embed" in name
    vision = any(token in name for token in ("-vl", ":vl", "vision", "llava"))

    parameter_sizes = [
        float(match.group("size")) for match in _PARAMETER_PATTERN.finditer(name)
    ]
    parameter_billions = max(parameter_sizes) if parameter_sizes else None
    heavy = bool(
        (parameter_billions is not None and parameter_billions >= 20.0)
        or any(token in name for token in ("70b", "72b", "405b"))
    )

    return ModelCapabilities(
        chat=not embedding,
        vision=vision,
        embedding=embedding,
        heavy=heavy,
        parameter_billions=parameter_billions,
    )
