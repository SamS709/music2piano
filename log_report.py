#!/usr/bin/env python3
"""Summarize Lightning training metrics by log version."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


def parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def find_metric_rows(
    rows: Iterable[dict[str, str]],
    metric: str,
) -> list[tuple[int, int, float]]:
    result = []
    for row in rows:
        value = parse_float(row.get(metric))
        if value is None:
            continue
        epoch = parse_int(row.get("epoch"))
        step = parse_int(row.get("step"))
        result.append(
            (
                epoch if epoch is not None else -1,
                step if step is not None else -1,
                value,
            )
        )
    return result


def summarize_version(log_dir: Path) -> dict[str, str | int]:
    metrics_path = log_dir / "metrics.csv"
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    train_rows = find_metric_rows(rows, "train_loss")
    val_rows = find_metric_rows(rows, "val_loss")
    lr_columns = [
        name
        for name in (rows[0].keys() if rows else [])
        if name and name.startswith("lr-")
    ]
    lr_rows = []
    for row in rows:
        for lr_column in lr_columns:
            lr = parse_float(row.get(lr_column))
            if lr is not None:
                lr_rows.append(lr)
                break

    steps = [
        parsed
        for row in rows
        if (parsed := parse_int(row.get("step"))) is not None
    ]
    epochs = [
        parsed
        for row in rows
        if (parsed := parse_int(row.get("epoch"))) is not None
    ]

    best_train = min(train_rows, key=lambda item: item[2]) if train_rows else None
    last_train = max(train_rows, key=lambda item: (item[0], item[1])) if train_rows else None
    best_val = min(val_rows, key=lambda item: item[2]) if val_rows else None
    last_val = max(val_rows, key=lambda item: (item[0], item[1])) if val_rows else None

    checkpoint_dir = log_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.is_dir() else []
    best_checkpoint = ""
    best_checkpoint_match = None
    for checkpoint in checkpoints:
        match = re.search(r"val_loss=(-?\d+(?:\.\d+)?)", checkpoint.name)
        if match:
            value = float(match.group(1))
            if best_checkpoint_match is None or value < best_checkpoint_match:
                best_checkpoint_match = value
                best_checkpoint = checkpoint.name

    version_match = re.fullmatch(r"version_(\d+)", log_dir.name)
    if version_match:
        version = int(version_match.group(1))
    else:
        source_version = next(
            (
                parent
                for parent in log_dir.parents
                if re.fullmatch(r"version_\d+", parent.name)
            ),
            None,
        )
        if source_version is not None:
            relative = log_dir.relative_to(source_version)
            version = f"{source_version.name}/{relative.as_posix()}"
        else:
            version = log_dir.name

    return {
        "version": version,
        "completed_epochs": max(epochs) + 1 if epochs else 0,
        "last_epoch": max(epochs) if epochs else "",
        "max_step": max(steps) if steps else 0,
        "train_initial": format_metric(train_rows[0][2] if train_rows else None),
        "train_last": format_metric(last_train[2] if last_train else None),
        "train_best": format_metric(best_train[2] if best_train else None),
        "train_best_epoch": best_train[0] if best_train else "",
        "val_initial": format_metric(val_rows[0][2] if val_rows else None),
        "val_last": format_metric(last_val[2] if last_val else None),
        "val_best": format_metric(best_val[2] if best_val else None),
        "val_best_epoch": best_val[0] if best_val else "",
        "lr_initial": format_metric(lr_rows[0] if lr_rows else None),
        "lr_last": format_metric(lr_rows[-1] if lr_rows else None),
        "lr_max": format_metric(max(lr_rows) if lr_rows else None),
        "checkpoint_count": len(checkpoints),
        "best_checkpoint": best_checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a CSV summary of all Lightning training versions.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("checkpoints/lightning_logs"),
        help="Lightning logs directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("training_report.csv"),
        help="output CSV path",
    )
    args = parser.parse_args()

    metrics_paths = sorted(
        {
            metrics_path.parent
            for metrics_path in args.logs.rglob("metrics.csv")
        },
        key=lambda path: str(path),
    )
    summaries = [
        summarize_version(path)
        for path in metrics_paths
        if (path / "metrics.csv").is_file()
    ]

    if not summaries:
        raise SystemExit(f"No metrics.csv found under {args.logs}")

    fieldnames = list(summaries[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} training versions to {args.output}")


if __name__ == "__main__":
    main()
