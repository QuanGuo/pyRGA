"""Portable visual artifacts for pyRGA simulation results."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .models import ForwardResult


def _color(value: float, minimum: float, maximum: float) -> str:
    """Small perceptually ordered blue-to-yellow palette without plot deps."""
    ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    ratio = min(1.0, max(0.0, ratio))
    stops = ((35, 46, 90), (38, 130, 142), (93, 190, 126), (238, 220, 74))
    position = ratio * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    fraction = position - index
    rgb = tuple(
        round(stops[index][channel] * (1 - fraction) + stops[index + 1][channel] * fraction)
        for channel in range(3)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def forward_result_svg(result: ForwardResult, well_index: int = 0) -> str:
    """Render one simulated hydraulic-head field as standalone SVG."""
    if not 0 <= well_index < len(result.heads):
        raise IndexError("well_index is outside the simulated pumping tests")
    field = np.asarray(result.heads[well_index], dtype=float).reshape(
        result.grid.ny + 1, result.grid.nx + 1
    )
    minimum = float(field.min())
    maximum = float(field.max())
    cell = 24
    plot_width = field.shape[1] * cell
    plot_height = field.shape[0] * cell
    width = plot_width + 180
    height = plot_height + 100
    rectangles = []
    for row in range(field.shape[0]):
        for column in range(field.shape[1]):
            value = float(field[row, column])
            rectangles.append(
                f'<rect x="{50 + column * cell}" y="{50 + (field.shape[0] - row - 1) * cell}" '
                f'width="{cell}" height="{cell}" fill="{_color(value, minimum, maximum)}">'
                f'<title>{value:.6g} m</title></rect>'
            )
    title = html.escape(f"Hydraulic head — {result.well_ids[well_index]}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<text x="50" y="28" font-family="system-ui" font-size="17" fill="#172033">{title}</text>'
        + "".join(rectangles)
        + f'<text x="{plot_width + 70}" y="70" font-family="system-ui" font-size="12" fill="#344054">max {maximum:.5g} m</text>'
        + f'<text x="{plot_width + 70}" y="92" font-family="system-ui" font-size="12" fill="#344054">min {minimum:.5g} m</text>'
        + f'<text x="50" y="{plot_height + 74}" font-family="system-ui" font-size="12" fill="#667085">'
        + html.escape(f"{result.grid.length_x:g} × {result.grid.length_y:g} {result.grid.length_unit}")
        + "</text></svg>"
    )


def write_forward_report(result: ForwardResult, directory: str | Path) -> Path:
    """Write an SVG plus self-contained HTML reproducibility report."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    svg = forward_result_svg(result)
    (output / "hydraulic-head.svg").write_text(svg, encoding="utf-8")
    metadata = json.dumps(result.model_dump(mode="json"), indent=2)
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(result.dataset_name)} — pyRGA result</title>
<style>body{{font:16px system-ui;color:#172033;background:#eef2f6;margin:0}}main{{max-width:1000px;margin:32px auto;padding:28px;background:white;border-radius:16px;box-shadow:0 8px 30px #10182818}}.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metric{{padding:14px 18px;background:#f2f4f7;border-radius:10px}}pre{{overflow:auto;background:#101828;color:#d0d5dd;padding:18px;border-radius:10px}}img{{max-width:100%;height:auto}}</style>
</head><body><main><h1>{html.escape(result.dataset_name)}</h1>
<p>Steady-state Darcy-flow result generated with pyRGA {html.escape(result.solver_version)}.</p>
<div class="metrics"><div class="metric"><strong>{result.summary.minimum_head:.5g} m</strong><br>minimum head</div><div class="metric"><strong>{result.summary.maximum_head:.5g} m</strong><br>maximum head</div><div class="metric"><strong>{result.summary.pumping_well_count}</strong><br>pumping tests</div></div>
<h2>Hydraulic head</h2><img src="hydraulic-head.svg" alt="Hydraulic-head field">
<details><summary>Machine-readable result</summary><pre>{html.escape(metadata)}</pre></details>
</main></body></html>"""
    destination = output / "index.html"
    destination.write_text(report, encoding="utf-8")
    return destination
