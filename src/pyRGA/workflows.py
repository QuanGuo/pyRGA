"""Validated, side-effect-free workflows built on the numerical core."""

from __future__ import annotations

from math import isclose

import numpy as np

from .gwsolver import groundwater_solver
from .models import DarcyDataset, ForwardResult, SimulationSummary


def run_forward(dataset: DarcyDataset) -> ForwardResult:
    """Run one steady-state solve for every pumping well in ``dataset``."""
    if dataset.grid.nx != dataset.grid.ny:
        raise ValueError("the current numerical core supports square grids only")
    dx = dataset.grid.length_x / dataset.grid.nx
    dy = dataset.grid.length_y / dataset.grid.ny
    if not isclose(dx, dy, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("the current numerical core supports square cells only (dx == dy)")

    pumping_wells = [well for well in dataset.wells if well.pumping_rate != 0]
    if not pumping_wells:
        raise ValueError("at least one well must have a non-zero pumping rate")

    conductivity = np.asarray(dataset.conductivity, dtype=float)
    head_fields = []
    for well in pumping_wells:
        # Apply the SI volumetric rate directly as the nodal load. The 2D
        # conductivity stiffness assumes a unit aquifer thickness of one metre.
        head_fields.append(
            groundwater_solver(conductivity, well.node, well.pumping_rate)
        )

    heads = np.asarray(head_fields)
    from . import __version__

    return ForwardResult(
        dataset_name=dataset.name,
        grid=dataset.grid,
        well_ids=[well.id for well in pumping_wells],
        heads=heads.tolist(),
        summary=SimulationSummary(
            minimum_head=float(heads.min()),
            maximum_head=float(heads.max()),
            mean_head=float(heads.mean()),
            pumping_well_count=len(pumping_wells),
        ),
        solver_version=__version__,
    )
