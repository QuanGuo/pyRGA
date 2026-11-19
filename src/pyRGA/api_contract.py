"""Export the shared schema bundle without requiring HTTP or MCP extras."""

from __future__ import annotations

import json

from .experiment_models import (
    ExperimentExport, GenerateFieldRequest, GeneratedField, InverseRequest,
    InverseResult, PumpingTestRequest, PumpingTestResult,
)
from .jobs import JobStatus


def experiment_schemas() -> dict:
    models = (GenerateFieldRequest, GeneratedField, PumpingTestRequest,
              PumpingTestResult, InverseRequest, InverseResult, ExperimentExport,
              JobStatus)
    return {
        "schema_version": "1.0",
        "job_kinds": {
            "generate_field": {"request": "GenerateFieldRequest", "result": "GeneratedField"},
            "run_pumping_test": {"request": "PumpingTestRequest", "result": "PumpingTestResult"},
            "run_inverse_model": {"request": "InverseRequest", "result": "InverseResult"},
        },
        "schemas": {model.__name__: model.model_json_schema() for model in models},
    }


def main() -> None:
    print(json.dumps(experiment_schemas(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
