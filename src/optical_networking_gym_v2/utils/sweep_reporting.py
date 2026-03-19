from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .experiment_utils import float_mean, float_std


Scalar = str | int | float | bool


def date_prefix(now: datetime | None = None) -> str:
    current = datetime.now() if now is None else now
    return current.strftime("%d-%m")


def build_sweep_output_paths(
    *,
    base_dir: Path,
    sweep_name: str,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    prefix = date_prefix(now)
    return (
        Path(base_dir) / f"{prefix}-{sweep_name}-episodes.csv",
        Path(base_dir) / f"{prefix}-{sweep_name}-summary.csv",
    )


def build_summary_fieldnames(
    *,
    base_fields: Sequence[str],
    metric_names: Sequence[str],
) -> list[str]:
    summary_fields = list(base_fields)
    for metric_name in metric_names:
        summary_fields.append(f"{metric_name}_mean")
        summary_fields.append(f"{metric_name}_std")
    return summary_fields


def aggregate_summary_metrics(
    rows: Sequence[Mapping[str, Scalar]],
    *,
    metric_names: Sequence[str],
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        aggregated[f"{metric_name}_mean"] = float_mean(values)
        aggregated[f"{metric_name}_std"] = float_std(values)
    return aggregated


def write_csv_rows(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Scalar]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "Scalar",
    "aggregate_summary_metrics",
    "build_summary_fieldnames",
    "build_sweep_output_paths",
    "date_prefix",
    "write_csv_rows",
]
