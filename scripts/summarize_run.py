"""Summarize a run directory after training and optional evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact JSON summary from a FloodNet run directory."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        action="append",
        default=[],
        help="Optional evaluation directory containing metrics.json; repeatable.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return loaded


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def summarize_run(run_dir: Path, evaluation_dirs: list[Path]) -> dict[str, Any]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_config = _read_json(resolved_run_dir / "resolved_config.json")
    train_summary = _read_json(resolved_run_dir / "train_summary.json")
    runtime = _read_json(resolved_run_dir / "runtime_metadata.json")
    overfit_gate = _read_json(resolved_run_dir / "overfit_gate.json")
    history = _read_history(resolved_run_dir / "history.csv")

    best_history_row: dict[str, Any] | None = None
    if history:
        best_history_row = max(
            history,
            key=lambda row: _float(row, "validation_miou10")
            if _float(row, "validation_miou10") is not None
            else float("-inf"),
        )

    evaluations: list[dict[str, Any]] = []
    for evaluation_dir in evaluation_dirs:
        metrics = _read_json(evaluation_dir.expanduser().resolve() / "metrics.json")
        if metrics is not None:
            evaluations.append(metrics)

    experiment = (resolved_config or {}).get("experiment", {})
    model = (resolved_config or {}).get("model", {})
    data = (resolved_config or {}).get("data", {})
    return {
        "run_dir": str(resolved_run_dir),
        "run_id": experiment.get("run_id"),
        "kind": experiment.get("kind"),
        "model": model.get("name"),
        "backbone": model.get("name"),
        "crop_size": data.get("crop_size"),
        "history_rows": len(history),
        "best_history_row": best_history_row,
        "train_summary": train_summary,
        "runtime": runtime,
        "overfit_gate": overfit_gate,
        "evaluations": evaluations,
        "checkpoints": {
            "best": str(resolved_run_dir / "best.pt")
            if (resolved_run_dir / "best.pt").is_file()
            else None,
            "last": str(resolved_run_dir / "last.pt")
            if (resolved_run_dir / "last.pt").is_file()
            else None,
        },
    }


def main() -> int:
    args = parse_args()
    summary = summarize_run(args.run_dir, args.evaluation_dir)
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
