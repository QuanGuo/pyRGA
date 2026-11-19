"""MCP interface for agent-driven pyRGA validation and simulation."""

from __future__ import annotations

from typing import Any
from contextlib import asynccontextmanager

from pydantic import ValidationError

from .models import DarcyDataset, ForwardResult
from .rendering import write_forward_report
from .workflows import run_forward
from .jobs import JobKind, JobManager
from .experiment_models import (
    ExperimentExport, GenerateFieldRequest, GeneratedField, PumpingTestRequest,
    PumpingTestResult, InverseRequest, InverseResult,
)


def generate_field_payload(payload: GenerateFieldRequest) -> dict[str, Any]:
    """Generate a seeded conductivity field independently of the inverse basis."""
    from .experiments import generate_field
    return generate_field(GenerateFieldRequest.model_validate(payload)).model_dump(mode="json")


def run_pumping_test_payload(payload: PumpingTestRequest) -> dict[str, Any]:
    """Collect a single-pump test with explicit monitors and seeded noise."""
    from .experiments import run_pumping_test
    return run_pumping_test(PumpingTestRequest.model_validate(payload)).model_dump(mode="json")


def run_inverse_payload(payload: InverseRequest) -> dict[str, Any]:
    """Fit conductivity to selected test observations without truth-field inputs."""
    from .inverse import run_inverse
    return run_inverse(InverseRequest.model_validate(payload)).model_dump(mode="json")


def experiment_schema_payload() -> dict[str, Any]:
    """Return versioned JSON Schemas for the portal experiment contract."""
    from .api_contract import experiment_schemas
    return experiment_schemas()


def prepare_inverse_payload(experiment: ExperimentExport) -> dict[str, Any]:
    """Convert selected immutable test snapshots to a truth-free inverse request."""
    return ExperimentExport.model_validate(experiment).inverse_request().model_dump(mode="json")


def validate_dataset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a JSON-compatible dataset and return a compact assessment."""
    try:
        dataset = DarcyDataset.model_validate(payload)
    except ValidationError as error:
        # Model-validator errors include the original exception in their context,
        # which cannot be serialized into an MCP tool response.
        return {
            "valid": False,
            "errors": error.errors(include_url=False, include_context=False),
        }
    return {
        "valid": True,
        "dataset_name": dataset.name,
        "element_count": dataset.grid.element_count,
        "node_count": dataset.grid.node_count,
        "well_count": len(dataset.wells),
        "observation_count": len(dataset.observations),
    }


def run_forward_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a dataset and execute its steady-state pumping tests."""
    dataset = DarcyDataset.model_validate(payload)
    return run_forward(dataset).model_dump(mode="json")


def render_forward_payload(result: dict[str, Any], output_directory: str) -> dict[str, str]:
    """Render a forward result to an HTML/SVG artifact bundle."""
    destination = write_forward_report(
        ForwardResult.model_validate(result),
        output_directory,
    )
    return {"report": str(destination), "visualization": str(destination.parent / "hydraulic-head.svg")}


def create_server():
    """Create the MCP server lazily so normal pyRGA imports do not require MCP."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError(
            "MCP support is not installed; install pyrga-ht[mcp]"
        ) from error

    # No worker is started until a job is submitted. The stdio server owns its
    # manager and always cleans up workers when its MCP lifespan ends.
    jobs = JobManager()

    @asynccontextmanager
    async def lifespan(server):
        try:
            yield {"jobs": jobs}
        finally:
            jobs.close()

    server = FastMCP(
        "pyRGA",
        instructions=(
            "Validate groundwater datasets and run Darcy-flow forward models. "
            "Call validate_dataset for legacy datasets. For experiment workflows, "
            "inspect experiment_schema. Use submit_experiment_job and polling for "
            "long computations requiring progress or cancellation."
        ),
        lifespan=lifespan,
    )
    server.tool(name="validate_dataset")(validate_dataset_payload)
    server.tool(name="run_forward_model")(run_forward_payload)
    server.tool(name="render_forward_result")(render_forward_payload)
    server.tool(name="generate_field")(generate_field_payload)
    server.tool(name="run_pumping_test")(run_pumping_test_payload)
    server.tool(name="run_inverse_model")(run_inverse_payload)
    server.tool(name="experiment_schema")(experiment_schema_payload)
    server.tool(name="prepare_inverse_request")(prepare_inverse_payload)

    def submit_experiment_job(kind: JobKind, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue generation, a pumping test or inversion and return its job ID."""
        return jobs.submit(kind, payload).model_dump(mode="json")

    def get_experiment_job(job_id: str) -> dict[str, Any]:
        """Poll job status and progress without returning large result arrays."""
        return jobs.status(job_id).model_dump(mode="json")

    def get_experiment_job_result(job_id: str) -> dict[str, Any]:
        """Fetch a successful result; unfinished, failed or cancelled jobs error."""
        return jobs.result(job_id)

    def cancel_experiment_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; terminal states remain unchanged."""
        return jobs.cancel(job_id).model_dump(mode="json")

    server.tool(name="submit_experiment_job")(submit_experiment_job)
    server.tool(name="get_experiment_job")(get_experiment_job)
    server.tool(name="get_experiment_job_result")(get_experiment_job_result)
    server.tool(name="cancel_experiment_job")(cancel_experiment_job)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
