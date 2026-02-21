from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkConfig, BenchmarkReport, EvaluationDataset, dataclass_to_dict


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    payload = load_json(path)
    return BenchmarkConfig.from_dict(payload)


def load_dataset(path: str | Path) -> EvaluationDataset:
    payload = load_json(path)
    return EvaluationDataset.from_dict(payload)


def write_report(report: BenchmarkReport, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dataclass_to_dict(report), file, indent=2)


def write_text(content: str, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
