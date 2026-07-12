from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.schemas.planning import ExecutionPlan


def test_execution_plan_normalizes_observed_local_model_output() -> None:
    payload = {
        "goal": "Research and compare two APIs.",
        "tasks": [
            {
                "id": 1,
                "objective": "Research current official API capabilities.",
                "required_evidence": ["Official OpenAI and Anthropic documentation"],
                "completion_criteria": (
                    "Evidence must be sourced and used for factual claims."
                ),
                "depends_on": [],
            },
            {
                "id": 2,
                "objective": "Compare the APIs and recommend an integration.",
                "required_evidence": "Findings from task 1",
                "completion_criteria": (
                    "Output must include a concise matrix and justified recommendation."
                ),
                "depends_on": [1],
            },
        ],
    }

    plan = ExecutionPlan.model_validate_json(json.dumps(payload))

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert plan.tasks[0].completion_criteria == [
        "Evidence must be sourced and used for factual claims."
    ]
    assert plan.tasks[1].required_evidence == ["Findings from task 1"]
    assert plan.tasks[1].depends_on == ["1"]


def test_execution_plan_preserves_canonical_output() -> None:
    plan = ExecutionPlan.model_validate(
        {
            "goal": "Canonical plan",
            "tasks": [
                {
                    "id": "task-1",
                    "objective": "Do the work",
                    "required_evidence": ["source"],
                    "completion_criteria": ["criterion"],
                    "depends_on": [],
                }
            ],
        }
    )

    assert plan.tasks[0].model_dump() == {
        "id": "task-1",
        "objective": "Do the work",
        "required_evidence": ["source"],
        "completion_criteria": ["criterion"],
        "depends_on": [],
    }


def test_execution_plan_still_rejects_unrelated_malformed_values() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(
            {
                "goal": "Invalid plan",
                "tasks": [
                    {
                        "id": True,
                        "objective": "Do the work",
                        "completion_criteria": {"unexpected": "object"},
                    }
                ],
            }
        )
