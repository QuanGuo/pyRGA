"""Provenance-aware readers and writers for canonical pyRGA datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import DarcyDataset, GridSpec, Provenance, Well


def load_dataset_json(path: str | Path) -> DarcyDataset:
    """Load and validate a canonical dataset from JSON."""
    source_path = Path(path)
    return DarcyDataset.model_validate_json(source_path.read_text(encoding="utf-8"))


def save_dataset_json(dataset: DarcyDataset, path: str | Path) -> Path:
    """Write a deterministic, human-readable canonical JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        dataset.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return destination


def dataset_from_csv(
    conductivity_path: str | Path,
    wells_path: str | Path,
    *,
    name: str,
    grid: GridSpec,
    conductivity_unit: str = "m/s",
) -> DarcyDataset:
    """Build a dataset from simple conductivity and well CSV tables.

    Conductivity CSV requires a ``conductivity`` column. Well CSV requires
    ``id`` and ``node`` and accepts optional ``pumping_rate`` and unit columns.
    """
    conductivity_file = Path(conductivity_path)
    wells_file = Path(wells_path)

    with conductivity_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "conductivity" not in rows[0]:
        raise ValueError("conductivity CSV must contain a conductivity column")
    conductivity = [float(row["conductivity"]) for row in rows]

    with wells_file.open(newline="", encoding="utf-8-sig") as handle:
        well_rows = list(csv.DictReader(handle))
    if well_rows and not {"id", "node"}.issubset(well_rows[0]):
        raise ValueError("wells CSV must contain id and node columns")
    wells = [
        Well(
            id=row["id"],
            node=int(row["node"]),
            pumping_rate=float(row.get("pumping_rate") or 0.0),
            pumping_rate_unit=row.get("pumping_rate_unit") or "m^3/s",
            x=float(row["x"]) if row.get("x") else None,
            y=float(row["y"]) if row.get("y") else None,
        )
        for row in well_rows
    ]

    return DarcyDataset(
        name=name,
        grid=grid,
        conductivity=conductivity,
        conductivity_unit=conductivity_unit,
        wells=wells,
        provenance=[
            Provenance(
                source="local CSV files",
                metadata={
                    "conductivity_file": str(conductivity_file),
                    "wells_file": str(wells_file),
                },
                transformations=["parsed numeric CSV fields"],
            )
        ],
    )
