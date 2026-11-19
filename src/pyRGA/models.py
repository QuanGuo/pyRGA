"""Canonical, transport-safe data models for pyRGA workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GridSpec(BaseModel):
    """A regular two-dimensional finite-element grid."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    nx: int = Field(gt=0, description="Number of elements in the x direction")
    ny: int = Field(gt=0, description="Number of elements in the y direction")
    length_x: float = Field(gt=0, description="Domain length in the x direction")
    length_y: float = Field(gt=0, description="Domain length in the y direction")
    length_unit: Literal["m"] = "m"
    crs: str | None = None

    @property
    def element_count(self) -> int:
        return self.nx * self.ny

    @property
    def node_count(self) -> int:
        return (self.nx + 1) * (self.ny + 1)


class Well(BaseModel):
    """A pumping or observation well located on a model node."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1)
    node: int = Field(ge=0)
    pumping_rate: float = 0.0
    pumping_rate_unit: Literal["m^3/s"] = "m^3/s"
    x: float | None = None
    y: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HeadObservation(BaseModel):
    """A hydraulic-head measurement and its uncertainty."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    well_id: str
    head: float
    head_unit: Literal["m"] = "m"
    timestamp: datetime | None = None
    standard_deviation: float | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Origin and transformations used to create a dataset."""

    source: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_url: str | None = None
    license: str | None = None
    transformations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DarcyDataset(BaseModel):
    """Validated input shared by Python, MCP, and portal workflows."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    grid: GridSpec
    conductivity: list[float]
    conductivity_unit: Literal["m/s"] = "m/s"
    wells: list[Well] = Field(default_factory=list)
    observations: list[HeadObservation] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references_and_shapes(self) -> "DarcyDataset":
        if len(self.conductivity) != self.grid.element_count:
            raise ValueError(
                "conductivity must contain exactly nx * ny values "
                f"({self.grid.element_count} expected)"
            )
        if any(value <= 0 for value in self.conductivity):
            raise ValueError("conductivity values must be positive")
        well_ids = [well.id for well in self.wells]
        if len(well_ids) != len(set(well_ids)):
            raise ValueError("well ids must be unique")
        if any(well.node >= self.grid.node_count for well in self.wells):
            raise ValueError("well node lies outside the grid")
        boundary_pumps = [
            well.id for well in self.wells
            if well.pumping_rate != 0
            and well.node % (self.grid.nx + 1) in (0, self.grid.nx)
        ]
        if boundary_pumps:
            raise ValueError(
                "pumping wells cannot lie on left or right fixed-head boundaries: "
                f"{boundary_pumps}"
            )
        unknown = {item.well_id for item in self.observations} - set(well_ids)
        if unknown:
            raise ValueError(f"observations reference unknown wells: {sorted(unknown)}")
        return self


class SimulationSummary(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    minimum_head: float
    maximum_head: float
    mean_head: float
    pumping_well_count: int


class ForwardResult(BaseModel):
    """JSON-safe forward-model result suitable for artifact storage."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["steady_state"] = "steady_state"
    dataset_name: str
    grid: GridSpec
    well_ids: list[str]
    heads: list[list[float]]
    summary: SimulationSummary
    solver_version: str
