"""Reproducible synthetic experiments, independent of transport and storage."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from .experiment_models import (
    GenerateFieldRequest, GeneratedField, Measurement, PumpingTestRequest,
    PumpingTestResult, StructureParameters, validate_grid, validate_node,
)
from .gwsolver import assemble_matrix
from .models import GridSpec


class WorkflowCancelled(RuntimeError):
    """A caller requested cancellation at a workflow checkpoint."""


def checkpoint(cancelled=None):
    if cancelled is not None and cancelled():
        raise WorkflowCancelled("computation cancelled")


def report(progress, fraction, stage, message, **detail):
    if progress is not None:
        progress({"progress": fraction, "stage": stage, "message": message, **detail})


def sample_log_fields(grid: GridSpec, structure: StructureParameters, seed: int,
                      count: int = 1, *, stream: int = 0) -> np.ndarray:
    """Sample periodic stationary log-K fields using physical correlation lengths.

    Normalize the spectrum in log space; its sum, rather than any individual
    realization's empirical variance, determines the requested marginal variance.
    """
    validate_grid(grid)
    kx = 2 * np.pi * np.fft.fftfreq(grid.nx, d=grid.length_x / grid.nx)
    ky = 2 * np.pi * np.fft.fftfreq(grid.ny, d=grid.length_y / grid.ny)
    with np.errstate(over="ignore", invalid="ignore"):
        radius = (structure.correlation_length_x * kx[None, :]) ** 2 + (
            structure.correlation_length_y * ky[:, None]
        ) ** 2
        if structure.covariance_type == "gaussian":
            log_spectrum = -0.5 * radius
        else:
            nu = 0.5 if structure.covariance_type == "exponential" else structure.matern_nu
            log_spectrum = -(nu + 1) * np.log1p(radius / (2 * nu))
        spectrum = np.exp(log_spectrum)
        spectrum *= structure.log_variance * grid.element_count / spectrum.sum()
    if not np.isfinite(spectrum).all():
        raise ValueError("correlation lengths and grid sizes produce a non-finite spectrum")
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))
    white = rng.normal(size=(count, grid.ny, grid.nx))
    fields = np.fft.ifft2(np.fft.fft2(white) * np.sqrt(spectrum)).real
    return fields.reshape(count, grid.element_count) + structure.mean_log_conductivity


def generate_field(request: GenerateFieldRequest, *, progress=None, cancelled=None) -> GeneratedField:
    """Generate materialized truth from a local seed, independently of inverse PCA."""
    request = GenerateFieldRequest.model_validate(request.model_dump())
    checkpoint(cancelled)
    report(progress, 0.1, "generation", "Generating seeded log-conductivity field")
    log_field = sample_log_fields(request.grid, request.structure, request.seed)[0]
    with np.errstate(over="ignore", under="ignore"):
        conductivity = np.exp(log_field)
    if not np.isfinite(conductivity).all() or np.any(conductivity <= 0):
        raise ValueError("generated conductivity is not finite and positive; adjust field parameters")
    checkpoint(cancelled)
    from . import __version__
    field = GeneratedField(
        field_id=GeneratedField.content_id(request, conductivity.tolist(), __version__), request=request,
        conductivity=conductivity.tolist(), solver_version=__version__,
    )
    report(progress, 1, "complete", "Field generated")
    return field


def solve_heads(grid: GridSpec, conductivity, pump_nodes, pumping_rates, *, cancelled=None):
    """Solve independent nodal loads with one exact stiffness factorization.

    Zero heads on left/right, no-flux on top/bottom, unit aquifer thickness.
    Using one accurate factorization also keeps inverse finite differences smooth.
    """
    validate_grid(grid)
    checkpoint(cancelled)
    k = np.asarray(conductivity, dtype=float)
    if k.shape != (grid.element_count,) or not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("conductivity must contain nx * ny finite positive values")
    if len(pump_nodes) != len(pumping_rates) or not len(pump_nodes):
        raise ValueError("provide equally sized, nonempty pump node/rate lists")
    for node in pump_nodes:
        validate_node(grid, node, pump=True)
    if not np.isfinite(pumping_rates).all() or np.any(np.asarray(pumping_rates) == 0):
        raise ValueError("pumping rates must be finite and nonzero")
    stiffness = np.array([
        [4, -1, -1, -2], [-1, 4, -2, -1],
        [-1, -2, 4, -1], [-2, -1, -1, 4],
    ], dtype=float) / 6
    matrix = assemble_matrix(grid.nx + 1, grid.ny + 1, stiffness, k)
    mask = np.arange(grid.node_count) % (grid.nx + 1)
    free = (mask != 0) & (mask != grid.nx)
    reduced = matrix[free, :][:, free].tocsc()
    try:
        factor = splu(reduced)
        heads = np.zeros((len(pump_nodes), grid.node_count))
        for i, (node, rate) in enumerate(zip(pump_nodes, pumping_rates)):
            checkpoint(cancelled)
            force = np.zeros(grid.node_count)
            force[node] = rate
            rhs = force[free]
            heads[i, free] = factor.solve(rhs)
            residual = np.linalg.norm(reduced @ heads[i, free] - rhs)
            if not np.isfinite(heads[i]).all() or residual > 1e-7 * np.linalg.norm(rhs):
                raise ValueError("forward solve failed its residual check; check conductivity contrast")
    except WorkflowCancelled:
        raise
    except RuntimeError as error:
        raise ValueError("conductivity produced a singular or ill-conditioned system") from error
    checkpoint(cancelled)
    return heads


def run_pumping_test(request: PumpingTestRequest, *, progress=None, cancelled=None) -> PumpingTestResult:
    """Snapshot a single-pump experiment and seeded, ordered monitor measurements."""
    request = PumpingTestRequest.model_validate(request.model_dump())
    grid = request.field.request.grid
    wells = {well.id: well for well in request.wells}
    report(progress, 0.1, "forward", "Solving steady-state pumping test")
    head = solve_heads(grid, request.field.conductivity,
                       [wells[request.pump_well_id].node], [request.pumping_rate],
                       cancelled=cancelled)[0]
    monitors = request.monitor_well_ids
    clean = np.array([head[wells[well_id].node] for well_id in monitors])
    sigma = np.hypot(request.noise.relative * np.abs(clean), request.noise.absolute_floor)
    noisy = clean + np.random.default_rng(request.noise.seed).normal(size=len(clean)) * sigma
    measurements = [Measurement(
        test_id=request.test_id, well_id=well_id, node=wells[well_id].node,
        clean_head=float(clean[i]), head=float(noisy[i]), standard_deviation=float(sigma[i]),
    ) for i, well_id in enumerate(monitors)]
    checkpoint(cancelled)
    from . import __version__
    result = PumpingTestResult(
        test_id=request.test_id, field_id=request.field.field_id, grid=grid,
        wells=request.wells, pump_well_id=request.pump_well_id,
        pumping_rate=request.pumping_rate, noise=request.noise,
        heads=head.tolist(), measurements=measurements, solver_version=__version__,
    )
    report(progress, 1, "complete", "Pumping test and measurements collected")
    return result
