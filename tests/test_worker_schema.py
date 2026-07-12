from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.schemas.worker import WorkerResult


def test_worker_result_normalizes_numeric_evidence_ids() -> None:
    result = WorkerResult.model_validate_json(
        json.dumps(
            {
                "answer": "Current conditions require external evidence.",
                "claims": [
                    {
                        "text": "The worker emitted a positional-looking reference.",
                        "evidence_ids": [0],
                    }
                ],
                "assumptions": [],
                "missing_information": [],
                "confidence": 0.4,
            }
        )
    )

    assert result.claims[0].evidence_ids == ["0"]


def test_worker_result_normalizes_singleton_string_lists() -> None:
    result = WorkerResult.model_validate(
        {
            "answer": "An answer.",
            "claims": [
                {
                    "text": "A claim.",
                    "evidence_ids": "source-1",
                }
            ],
            "assumptions": "Location was not explicitly resolved.",
            "missing_information": None,
            "confidence": 0.5,
        }
    )

    assert result.claims[0].evidence_ids == ["source-1"]
    assert result.assumptions == ["Location was not explicitly resolved."]
    assert result.missing_information == []


def test_worker_result_preserves_canonical_output() -> None:
    payload = {
        "answer": "An evidence-backed answer.",
        "claims": [
            {
                "text": "A supported claim.",
                "evidence_ids": ["weather-source-1"],
            }
        ],
        "assumptions": ["The requested location is Indianapolis."],
        "missing_information": ["Exact neighborhood was not provided."],
        "confidence": 0.8,
    }

    result = WorkerResult.model_validate(payload)

    assert result.model_dump() == payload


@pytest.mark.parametrize(
    "invalid_evidence_ids",
    [
        {"position": 0},
        [["source-1"]],
        [True],
        [1.5],
    ],
)
def test_worker_result_rejects_ambiguous_evidence_id_shapes(
    invalid_evidence_ids: object,
) -> None:
    with pytest.raises(ValidationError):
        WorkerResult.model_validate(
            {
                "answer": "An answer.",
                "claims": [
                    {
                        "text": "A claim.",
                        "evidence_ids": invalid_evidence_ids,
                    }
                ],
            }
        )
