#!/usr/bin/env python
"""Prepare RetailCabinet-4 annotations for external baseline projects.

The TC-CLIP split files already use the common "relative_path label" format.
This script copies that format into external_baselines/annotations/retail4 and
adds absolute-path CSV files plus validation metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = SCRIPT_DIR.parent
TC_ROOT_DEFAULT = EXTERNAL_ROOT.parent
SPLIT_NAMES = ("train", "val", "test")


def _read_retail4_root(default_yaml: Path) -> str | None:
    if not default_yaml.exists():
        return None
    text = default_yaml.read_text(encoding="utf-8")
    match = re.search(r"retail4:\s.*?^\s{2}root:\s*(.+?)\s*$", text, re.S | re.M)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def read_labels(label_file: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with label_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            labels[int(row[0])] = row[1].strip()
    return labels


def read_split(split_file: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_no, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid split line {split_file}:{line_no}: {raw!r}")
        rel_path = parts[0].replace("\\", "/")
        rows.append((rel_path, int(parts[1])))
    return rows


def base_video_id(rel_path: str) -> str:
    stem = Path(rel_path).stem
    stem = stem.replace("__rev", "")
    return stem


def write_split_outputs(
    split_name: str,
    rows: list[tuple[str, int]],
    labels: dict[int, str],
    data_root: Path,
    out_dir: Path,
) -> dict:
    rel_txt = out_dir / f"{split_name}.txt"
    abs_txt = out_dir / f"{split_name}_absolute.txt"
    csv_file = out_dir / f"{split_name}.csv"

    counts = Counter(label for _, label in rows)
    missing: list[str] = []
    rel_seen = Counter(rel for rel, _ in rows)

    with rel_txt.open("w", encoding="utf-8", newline="\n") as rel_handle, \
            abs_txt.open("w", encoding="utf-8", newline="\n") as abs_handle, \
            csv_file.open("w", encoding="utf-8-sig", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow([
            "split",
            "rel_path",
            "abs_path",
            "label",
            "class_name",
            "exists",
            "is_reverse",
            "base_id",
        ])
        for rel_path, label in rows:
            abs_path = (data_root / rel_path).resolve()
            exists = abs_path.exists()
            if not exists:
                missing.append(str(abs_path))
            class_name = labels.get(label, f"class_{label}")
            rel_handle.write(f"{rel_path} {label}\n")
            abs_handle.write(f"{abs_path.as_posix()} {label}\n")
            writer.writerow([
                split_name,
                rel_path,
                abs_path.as_posix(),
                label,
                class_name,
                int(exists),
                int("__rev" in Path(rel_path).stem),
                base_video_id(rel_path),
            ])

    return {
        "num_samples": len(rows),
        "class_counts": {str(k): counts.get(k, 0) for k in sorted(labels)},
        "missing_count": len(missing),
        "missing_examples": missing[:10],
        "duplicate_relative_paths": [path for path, n in rel_seen.items() if n > 1],
        "base_ids": sorted({base_video_id(rel) for rel, _ in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tc-root", type=Path, default=TC_ROOT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=EXTERNAL_ROOT / "annotations" / "retail4")
    parser.add_argument("--strict", action="store_true", help="Fail if any referenced video is missing.")
    args = parser.parse_args()

    tc_root = args.tc_root.resolve()
    default_yaml_candidates = [
        tc_root / "configs" / "common" / "default.yaml",
        tc_root / "code" / "configs" / "common" / "default.yaml",
    ]
    default_yaml = next(
        (path for path in default_yaml_candidates if path.exists()),
        default_yaml_candidates[0],
    )
    data_root = args.data_root
    if data_root is None:
        configured_root = _read_retail4_root(default_yaml)
        if not configured_root:
            raise RuntimeError(f"Could not infer retail4.root from {default_yaml}")
        data_root = Path(configured_root)
    data_root = data_root.resolve()

    split_dir_candidates = [
        tc_root / "datasets_splits",
        tc_root / "code" / "datasets_splits",
    ]
    split_dir = next(
        (path for path in split_dir_candidates if (path / "train.txt").exists()),
        split_dir_candidates[0],
    )
    label_file_candidates = [
        tc_root / "labels" / "retail4_cjj_labels.csv",
        tc_root / "code" / "labels" / "retail4_cjj_labels.csv",
    ]
    label_file = next(
        (path for path in label_file_candidates if path.exists()),
        label_file_candidates[0],
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(label_file)
    if labels != {0: "drink", 1: "open", 2: "purchase", 3: "putback"}:
        print(f"[warn] Unexpected label map: {labels}")

    metadata = {
        "dataset": "RetailCabinet-4",
        "subset": "cjj",
        "tc_root": tc_root.as_posix(),
        "data_root": data_root.as_posix(),
        "label_file": label_file.as_posix(),
        "labels": {str(k): v for k, v in sorted(labels.items())},
        "splits": {},
        "split_overlap": {},
    }

    all_base_ids: dict[str, set[str]] = {}
    total_missing = 0
    for split_name in SPLIT_NAMES:
        rows = read_split(split_dir / f"{split_name}.txt")
        unknown_labels = sorted({label for _, label in rows if label not in labels})
        if unknown_labels:
            raise ValueError(f"{split_name} contains unknown labels: {unknown_labels}")
        info = write_split_outputs(split_name, rows, labels, data_root, out_dir)
        metadata["splits"][split_name] = {k: v for k, v in info.items() if k != "base_ids"}
        all_base_ids[split_name] = set(info["base_ids"])
        total_missing += info["missing_count"]

    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left >= right:
                continue
            overlap = sorted(all_base_ids[left] & all_base_ids[right])
            metadata["split_overlap"][f"{left}_vs_{right}_base_id_count"] = len(overlap)
            metadata["split_overlap"][f"{left}_vs_{right}_base_id_examples"] = overlap[:10]

    (out_dir / "classes.txt").write_text(
        "\n".join(labels[i] for i in sorted(labels)) + "\n",
        encoding="utf-8",
    )
    (out_dir / "label_map.txt").write_text(
        "\n".join(f"{i} {labels[i]}" for i in sorted(labels)) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name"])
        for label_id in sorted(labels):
            writer.writerow([label_id, labels[label_id]])
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stats_file = out_dir / "split_stats.csv"
    with stats_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "class_name", "count"])
        for split_name, split_info in metadata["splits"].items():
            for label, count in split_info["class_counts"].items():
                writer.writerow([split_name, label, labels[int(label)], count])

    print(f"Wrote RetailCabinet-4 annotations to {out_dir}")
    print(json.dumps(metadata["splits"], indent=2, ensure_ascii=False))
    if total_missing and args.strict:
        print(f"[error] Missing videos: {total_missing}")
        return 2
    if total_missing:
        print(f"[warn] Missing videos: {total_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
