"""Versioned experiment contract shared by Python, HTTP, MCP and the portal.

Conductivity is element-centred; hydraulic heads and wells use grid nodes.
Arrays are row-major with x varying fastest and the origin at bottom left.
"""

from __future__ import annotations

from math import isclose
import hashlib
import json
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import GridSpec


class ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def validate_grid(grid: GridSpec) -> None:
    if not 2 <= grid.nx <= 64 or grid.nx != grid.ny:
        raise ValueError("experiment grids must have equal nx and ny between 2 and 64")
    if not isclose(grid.length_x, grid.length_y, rel_tol=1e-12, abs_tol=0):
        raise ValueError("experiment grids require square physical cells")


def validate_node(grid: GridSpec, node: int, *, pump: bool = False) -> None:
    if not 0 <= node < grid.node_count:
        raise ValueError("well node lies outside the grid")
    if pump and node % (grid.nx + 1) in (0, grid.nx):
        raise ValueError("pumping wells cannot lie on left or right fixed-head boundaries")


class StructureParameters(ExperimentModel):
    covariance_type: Literal["gaussian", "exponential", "matern"] = "gaussian"
    mean_log_conductivity: float = Field(default=-9.210340371976184, ge=-30, le=10)
    log_variance: float = Field(default=1, gt=0, le=10)
    correlation_length_x: float = Field(default=48, gt=0)
    correlation_length_y: float = Field(default=64, gt=0)
    matern_nu: float = Field(default=0.5, gt=0, le=10)


class GenerateFieldRequest(ExperimentModel):
    grid: GridSpec = Field(default_factory=lambda: GridSpec(
        nx=32, ny=32, length_x=320, length_y=320
    ))
    structure: StructureParameters = Field(default_factory=StructureParameters)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def check_grid(self):
        validate_grid(self.grid)
        return self


class GeneratedField(ExperimentModel):
    field_id: str = Field(min_length=1)
    request: GenerateFieldRequest
    conductivity: list[float]
    solver_version: str

    @staticmethod
    def content_id(request: GenerateFieldRequest, conductivity: list[float], solver_version: str) -> str:
        identity = json.dumps({"request": request.model_dump(mode="json"),
                               "conductivity": conductivity,
                               "solver_version": solver_version},
                              sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(identity.encode()).hexdigest()

    @model_validator(mode="after")
    def check_field(self):
        if len(self.conductivity) != self.request.grid.element_count:
            raise ValueError("conductivity must contain nx * ny values")
        if any(value <= 0 for value in self.conductivity):
            raise ValueError("conductivity values must be positive")
        if self.field_id != self.content_id(self.request, self.conductivity, self.solver_version):
            raise ValueError("field_id does not match the materialized field and generation settings")
        return self


class InstalledWell(ExperimentModel):
    id: str = Field(min_length=1, max_length=128)
    node: int = Field(ge=0)


class NoiseSettings(ExperimentModel):
    relative: float = Field(default=0.05, ge=0, le=1)
    absolute_floor: float = Field(default=0.001, ge=0)
    seed: int = Field(default=44, ge=0, le=2**32 - 1)


class PumpingTestRequest(ExperimentModel):
    field: GeneratedField
    wells: list[InstalledWell] = Field(min_length=2, max_length=25)
    test_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    pump_well_id: str
    pumping_rate: float = Field(default=-0.001)
    monitor_well_ids: list[str] | None = None
    noise: NoiseSettings = Field(default_factory=NoiseSettings)

    @model_validator(mode="after")
    def check_wells(self):
        ids = [well.id for well in self.wells]
        nodes = [well.node for well in self.wells]
        if len(set(ids)) != len(ids) or len(set(nodes)) != len(nodes):
            raise ValueError("installed well ids and nodes must be unique")
        grid = self.field.request.grid
        for well in self.wells:
            validate_node(grid, well.node, pump=well.id == self.pump_well_id)
        if self.pump_well_id not in ids or self.pumping_rate == 0:
            raise ValueError("select an installed pumping well and a nonzero pumping rate")
        monitors = self.monitor_well_ids
        if monitors is None:
            monitors = [well_id for well_id in ids if well_id != self.pump_well_id]
            self.monitor_well_ids = monitors
        if not monitors or len(set(monitors)) != len(monitors):
            raise ValueError("select at least one unique monitoring well")
        if self.pump_well_id in monitors or not set(monitors).issubset(ids):
            raise ValueError("monitor wells must be installed wells other than the active pump")
        return self


class Measurement(ExperimentModel):
    test_id: str
    well_id: str
    node: int = Field(ge=0)
    clean_head: float
    head: float
    standard_deviation: float = Field(ge=0)


class PumpingTestResult(ExperimentModel):
    test_id: str
    field_id: str
    grid: GridSpec
    wells: list[InstalledWell]
    pump_well_id: str
    pumping_rate: float
    noise: NoiseSettings
    heads: list[float]
    measurements: list[Measurement] = Field(min_length=1)
    solver_version: str

    @model_validator(mode="after")
    def check_result(self):
        validate_grid(self.grid)
        if len(self.heads) != self.grid.node_count:
            raise ValueError("heads must contain (nx + 1) * (ny + 1) values")
        wells = {well.id: well for well in self.wells}
        if len(wells) != len(self.wells) or len({w.node for w in self.wells}) != len(wells):
            raise ValueError("snapshot wells must have unique ids and nodes")
        if self.pump_well_id not in wells or self.pumping_rate == 0:
            raise ValueError("invalid pumping well in test snapshot")
        for well in self.wells:
            validate_node(self.grid, well.node, pump=well.id == self.pump_well_id)
        ids = []
        for measurement in self.measurements:
            if (measurement.test_id != self.test_id or measurement.well_id not in wells
                    or wells[measurement.well_id].node != measurement.node
                    or measurement.well_id == self.pump_well_id):
                raise ValueError("measurement does not match its test/well snapshot")
            if not isclose(measurement.clean_head, self.heads[measurement.node], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("clean measurement must match the saved head field")
            ids.append(measurement.well_id)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate monitoring measurement in test")
        return self


class ObservedHead(ExperimentModel):
    well_id: str
    node: int = Field(ge=0)
    head: float
    standard_deviation: float = Field(ge=0)


class InverseTest(ExperimentModel):
    test_id: str
    pump_node: int = Field(ge=0)
    pumping_rate: float
    observations: list[ObservedHead] = Field(min_length=1, max_length=24)


class InverseSettings(ExperimentModel):
    components: int = Field(default=20, ge=1, le=50)
    ensemble_size: int = Field(default=200, ge=2, le=400)
    seed: int = Field(default=43, ge=0, le=2**32 - 1)
    max_iterations: int = Field(default=10, ge=1, le=50)
    tolerance: float = Field(default=1e-5, gt=0, le=1)
    regularization: float = Field(default=1, gt=0)
    initial_damping: float = Field(default=0.001, gt=0)


class InverseRequest(ExperimentModel):
    grid: GridSpec
    prior: StructureParameters = Field(default_factory=StructureParameters)
    settings: InverseSettings = Field(default_factory=InverseSettings)
    tests: list[InverseTest] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_tests(self):
        validate_grid(self.grid)
        if self.settings.components > min(self.settings.ensemble_size - 1, self.grid.element_count):
            raise ValueError("components must not exceed ensemble_size - 1 or element count")
        ids = [test.test_id for test in self.tests]
        if len(ids) != len(set(ids)):
            raise ValueError("inverse test ids must be unique")
        for test in self.tests:
            validate_node(self.grid, test.pump_node, pump=True)
            if test.pumping_rate == 0:
                raise ValueError("inverse tests require a nonzero pumping rate")
            nodes, wells = [], []
            for observation in test.observations:
                validate_node(self.grid, observation.node)
                if observation.node == test.pump_node:
                    raise ValueError("the active pump cannot also be a monitor")
                nodes.append(observation.node)
                wells.append(observation.well_id)
            if len(set(nodes)) != len(nodes) or len(set(wells)) != len(wells):
                raise ValueError("inverse monitoring nodes and ids must be unique per test")
        if not any(observation.node % (self.grid.nx + 1) not in (0, self.grid.nx)
                   for test in self.tests for observation in test.observations):
            raise ValueError("inversion requires at least one monitor away from fixed-head boundaries")
        return self


class InverseIteration(ExperimentModel):
    iteration: int
    objective: float
    data_misfit: float
    prior_penalty: float
    step_norm: float
    damping: float
    accepted: bool


class PredictedMeasurement(ExperimentModel):
    test_id: str
    well_id: str
    observed_head: float
    predicted_head: float
    standard_deviation: float
    residual: float


class InverseResult(ExperimentModel):
    request: InverseRequest
    estimated_conductivity: list[float]
    coefficients: list[float]
    mean_log_conductivity: float
    initial_objective: float
    final_objective: float
    converged: bool
    termination_reason: str
    iterations: list[InverseIteration]
    predictions: list[PredictedMeasurement]
    solver_version: str

    @model_validator(mode="after")
    def check_result(self):
        if (len(self.estimated_conductivity) != self.request.grid.element_count
                or any(value <= 0 for value in self.estimated_conductivity)):
            raise ValueError("estimated conductivity must contain nx * ny positive values")
        if len(self.coefficients) != self.request.settings.components:
            raise ValueError("coefficient count must match inverse components")
        expected = [(test.test_id, item.well_id, item.head)
                    for test in self.request.tests for item in test.observations]
        actual = [(item.test_id, item.well_id, item.observed_head) for item in self.predictions]
        if expected != actual:
            raise ValueError("inverse predictions must match selected observations in order")
        if not self.iterations or self.iterations[0].objective != self.initial_objective or self.iterations[-1].objective != self.final_objective:
            raise ValueError("inverse history must match initial and final objective values")
        return self


class ExperimentExport(ExperimentModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    field: GeneratedField
    wells: list[InstalledWell] = Field(default_factory=list, max_length=25)
    tests: list[PumpingTestResult] = Field(default_factory=list, max_length=64)
    selected_test_ids: list[str] = Field(default_factory=list)
    inverse_prior: StructureParameters = Field(default_factory=StructureParameters)
    inverse_settings: InverseSettings = Field(default_factory=InverseSettings)
    inverse_result: InverseResult | None = None

    @model_validator(mode="before")
    @classmethod
    def default_prior_to_generation(cls, value):
        if isinstance(value, dict) and "inverse_prior" not in value:
            field = value.get("field")
            if isinstance(field, GeneratedField):
                prior = field.request.structure.model_dump()
            elif isinstance(field, dict):
                request = field.get("request")
                if isinstance(request, GenerateFieldRequest):
                    prior = request.structure.model_dump()
                elif isinstance(request, dict):
                    prior = request.get("structure", {})
                else:
                    return value
            else:
                return value
            value = {**value, "inverse_prior": prior}
        return value

    @model_validator(mode="after")
    def check_experiment(self):
        grid = self.field.request.grid
        if len({w.id for w in self.wells}) != len(self.wells) or len({w.node for w in self.wells}) != len(self.wells):
            raise ValueError("installed wells must have unique ids and nodes")
        for well in self.wells:
            validate_node(grid, well.node)
        ids = [test.test_id for test in self.tests]
        if len(ids) != len(set(ids)):
            raise ValueError("test ids must be unique within an experiment")
        if any(test.field_id != self.field.field_id or test.grid != grid for test in self.tests):
            raise ValueError("test snapshots must belong to the current true field and grid")
        if len(set(self.selected_test_ids)) != len(self.selected_test_ids) or not set(self.selected_test_ids).issubset(ids):
            raise ValueError("selected_test_ids must reference unique collected tests")
        if self.inverse_result is not None:
            expected = self.inverse_request()
            if self.inverse_result.request != expected:
                raise ValueError("inverse result is stale for the selected tests or prior/settings")
        return self

    def inverse_request(self) -> InverseRequest:
        """Construct a truth-free inverse input from selected immutable runs."""
        by_id = {test.test_id: test for test in self.tests}
        tests = []
        for test_id in self.selected_test_ids:
            test = by_id[test_id]
            pump = next(well for well in test.wells if well.id == test.pump_well_id)
            tests.append(InverseTest(
                test_id=test_id, pump_node=pump.node, pumping_rate=test.pumping_rate,
                observations=[ObservedHead(
                    well_id=item.well_id, node=item.node, head=item.head,
                    standard_deviation=item.standard_deviation,
                ) for item in test.measurements],
            ))
        return InverseRequest(grid=self.field.request.grid, prior=self.inverse_prior,
                              settings=self.inverse_settings, tests=tests)
