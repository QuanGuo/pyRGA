"""MCP interface for agent-driven pyRGA validation and simulation."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .models import DarcyDataset, ForwardResult
from .rendering import write_forward_report
from .workflows import run_forward


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

    server = FastMCP(
        "pyRGA",
        instructions=(
            "Validate groundwater datasets and run Darcy-flow forward models. "
            "Call validate_dataset before computational tools."
        ),
    )
    server.tool(name="validate_dataset")(validate_dataset_payload)
    server.tool(name="run_forward_model")(run_forward_payload)
    server.tool(name="render_forward_result")(render_forward_payload)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
