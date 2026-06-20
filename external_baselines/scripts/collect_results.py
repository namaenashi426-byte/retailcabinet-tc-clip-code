#!/usr/bin/env python
"""Collect TC-CLIP and external-baseline metrics into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = SCRIPT_DIR.parent
EXPERIMENT_FILE = EXTERNAL_ROOT / "experiments" / "baselines.yaml"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ACC_RE = re.compile(r"Acc@1\s+([0-9.]+)")
STAR_ACC_RE = re.compile(r"\*\s+Acc@1\s+([0-9.]+)")
MMACTION_ACC_RE = re.compile(r"\bacc/top1:\s*([0-9.]+)", re.I)
ACC_COLON_RE = re.compile(r"(?:accuracy_top1|Accuracy)[^0-9]+([0-9.]+)", re.I)
F1_RE = re.compile(
    r"Macro-F1\s+([0-9.]+)\s+Balanced Acc\s+([0-9.]+)"
    r"(?:\s+Clips/s\s+([0-9.]+)\s+ms/clip\s+([0-9.]+)\s+Peak Mem\s+([0-9.]+)\s+MB)?"
)
STAGE_METADATA_KEYS = {"class_balanced_loss", "class_weights"}


def load_experiment_file(path: Path = EXPERIMENT_FILE) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible and PyYAML is not installed."
            ) from exc
        return yaml.safe_load(text)


def read_text(path: Path) -> str:
    return ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="ignore"))


def parse_metrics(text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "acc": None,
        "macro_f1": None,
        "bacc": None,
        "clips_s": None,
        "ms_clip": None,
        "peak_mem_mb": None,
    }
    mmaction_acc_matches = MMACTION_ACC_RE.findall(text)
    star_acc_matches = STAR_ACC_RE.findall(text)
    acc_matches = ACC_RE.findall(text)
    if mmaction_acc_matches:
        metrics["acc"] = float(mmaction_acc_matches[-1])
    elif star_acc_matches:
        metrics["acc"] = float(star_acc_matches[-1])
    elif acc_matches:
        metrics["acc"] = float(acc_matches[-1])
    else:
        generic_acc = ACC_COLON_RE.findall(text)
        if generic_acc:
            metrics["acc"] = float(generic_acc[-1])

    f1_matches = F1_RE.findall(text)
    if f1_matches:
        macro_f1, bacc, clips_s, ms_clip, peak_mem = f1_matches[-1]
        metrics["macro_f1"] = float(macro_f1)
        metrics["bacc"] = float(bacc)
        if clips_s:
            metrics["clips_s"] = float(clips_s)
            metrics["ms_clip"] = float(ms_clip)
            metrics["peak_mem_mb"] = float(peak_mem)
    return metrics


def parse_stage_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines()[:80]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in STAGE_METADATA_KEYS:
            metadata[key] = value.strip()
    return metadata


def resolved_class_balanced_loss(
    config: dict[str, Any],
    model: dict[str, Any],
    metadata: dict[str, str] | None = None,
) -> str:
    if metadata and metadata.get("class_balanced_loss"):
        return metadata["class_balanced_loss"]
    value = bool(model.get("class_balanced_loss", config["defaults"].get("class_balanced_loss", False)))
    return str(value)


def resolved_class_weights(
    metadata: dict[str, str] | None = None,
) -> str:
    if metadata and metadata.get("class_weights"):
        return metadata["class_weights"]
    return ""


def find_logs_for_model(model: dict[str, Any], seed: int) -> list[Path]:
    backend = model.get("backend")
    if backend == "tc_clip_existing":
        logs: list[Path] = []
        for pattern in model.get("log_globs", []):
            logs.extend(sorted(EXTERNAL_ROOT.glob(pattern.format(seed=seed))))
        # Avoid accidentally collecting single-view logs for the multi-view rows.
        return [path for path in logs if "test_svi" not in path.as_posix()]

    candidates: list[Path] = []
    stable_log_dir = EXTERNAL_ROOT / "run_logs" / model["model_id"] / f"seed{seed}"
    for pattern in (
        "test_clip*_crop*.log",
        "val_clip*_crop*.log",
        "train.log",
        "smoke.log",
        "*.log",
    ):
        candidates.extend(sorted(stable_log_dir.glob(pattern)))

    if not model.get("ignore_work_dir_fallback", False):
        work_dir = EXTERNAL_ROOT / "work_dirs" / model["model_id"] / f"seed{seed}"
        for suffix in ("*.log", "*.txt", "*.json"):
            candidates.extend(sorted(work_dir.rglob(suffix)))

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        key = path.resolve().as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)
    return unique_candidates


def collect_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    default_seeds = config["defaults"].get("seeds", [1024, 1025, 1026])
    for model in config["models"]:
        seeds = model.get("seeds", default_seeds)
        for seed in seeds:
            logs = find_logs_for_model(model, int(seed))
            if not logs:
                rows.append(row_for_missing(config, model, int(seed)))
                continue
            parsed_any = False
            for log_path in logs:
                text = read_text(log_path)
                metrics = parse_metrics(text)
                if any(value is not None for value in metrics.values()):
                    metadata = parse_stage_metadata(text)
                    rows.append(row_for_metrics(config, model, int(seed), log_path, metrics, metadata))
                    parsed_any = True
                    break
            if not parsed_any:
                rows.append(row_for_missing(config, model, int(seed), logs[0]))
    return rows


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def row_for_metrics(
    config: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    log_path: Path,
    metrics: dict[str, float | None],
    metadata: dict[str, str],
) -> dict[str, str]:
    return {
        "model_id": model["model_id"],
        "display_name": model.get("display_name", model["model_id"]),
        "backend": model.get("backend", ""),
        "seed": str(seed),
        "frames": str(model.get("frames", "")),
        "class_balanced_loss": resolved_class_balanced_loss(config, model, metadata),
        "class_weights": resolved_class_weights(metadata),
        "acc": _fmt(metrics["acc"]),
        "macro_f1": _fmt(metrics["macro_f1"]),
        "bacc": _fmt(metrics["bacc"]),
        "clips_s": _fmt(metrics["clips_s"]),
        "ms_clip": _fmt(metrics["ms_clip"]),
        "peak_mem_mb": _fmt(metrics["peak_mem_mb"]),
        "status": "parsed",
        "log_path": log_path.resolve().as_posix(),
    }


def row_for_missing(
    config: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    log_path: Path | None = None,
) -> dict[str, str]:
    return {
        "model_id": model["model_id"],
        "display_name": model.get("display_name", model["model_id"]),
        "backend": model.get("backend", ""),
        "seed": str(seed),
        "frames": str(model.get("frames", "")),
        "class_balanced_loss": resolved_class_balanced_loss(config, model),
        "class_weights": "",
        "acc": "",
        "macro_f1": "",
        "bacc": "",
        "clips_s": "",
        "ms_clip": "",
        "peak_mem_mb": "",
        "status": "missing_or_unparsed",
        "log_path": "" if log_path is None else log_path.resolve().as_posix(),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "display_name",
        "backend",
        "seed",
        "frames",
        "class_balanced_loss",
        "class_weights",
        "acc",
        "macro_f1",
        "bacc",
        "clips_s",
        "ms_clip",
        "peak_mem_mb",
        "status",
        "log_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "parsed":
            grouped[
                (
                    row["model_id"],
                    row.get("class_balanced_loss", ""),
                    row.get("class_weights", ""),
                )
            ].append(row)

    out: list[dict[str, str]] = []
    metric_names = ["acc", "macro_f1", "bacc", "clips_s", "ms_clip", "peak_mem_mb"]
    for (model_id, class_balanced_loss, class_weights), group in sorted(grouped.items()):
        item = {
            "model_id": model_id,
            "display_name": group[0]["display_name"],
            "backend": group[0]["backend"],
            "frames": group[0]["frames"],
            "class_balanced_loss": class_balanced_loss,
            "class_weights": class_weights,
            "n": str(len(group)),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric]]
            item[f"{metric}_mean"] = "" if not values else f"{mean(values):.3f}"
            item[f"{metric}_std"] = "" if len(values) < 2 else f"{stdev(values):.3f}"
        out.append(item)
    return out


def write_aggregate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXTERNAL_ROOT / "results" / "summary.csv")
    args = parser.parse_args()

    config = load_experiment_file()
    rows = collect_rows(config)
    out_path = args.out.resolve()
    write_csv(out_path, rows)
    aggregate_rows = aggregate(rows)
    aggregate_path = out_path.with_name(out_path.stem + "_agg.csv")
    write_aggregate_csv(aggregate_path, aggregate_rows)
    print(f"Wrote {out_path}")
    print(f"Wrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
