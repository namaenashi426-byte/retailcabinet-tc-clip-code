#!/usr/bin/env python
"""Clone or update external baseline repositories listed in baselines.yaml."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXTERNAL_ROOT = SCRIPT_DIR.parent
EXPERIMENT_FILE = EXTERNAL_ROOT / "experiments" / "baselines.yaml"


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


def run(command: list[str], cwd: Path, dry_run: bool) -> int:
    printable = " ".join(command)
    print(f"[cwd] {cwd}")
    print(f"[cmd] {printable}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd))
    return completed.returncode


def selected_repositories(config: dict[str, Any], selection: str) -> list[dict[str, Any]]:
    repos = config.get("repositories", [])
    if selection == "all":
        return repos
    if selection == "required":
        return [repo for repo in repos if repo.get("required")]
    wanted = {item.strip() for item in selection.split(",") if item.strip()}
    return [repo for repo in repos if repo["repo_id"] in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="required",
        help="required, all, or comma-separated repo ids from baselines.yaml",
    )
    parser.add_argument("--update", action="store_true", help="Run git pull for existing repos.")
    parser.add_argument("--full", action="store_true", help="Clone full history instead of --depth 1.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_experiment_file()
    repos_root = EXTERNAL_ROOT / "repos"
    repos_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for repo in selected_repositories(config, args.repo):
        target = (EXTERNAL_ROOT / repo["path"]).resolve()
        if target.exists():
            print(f"[skip] {repo['repo_id']} already exists at {target}")
            if args.update:
                failures += run(["git", "pull", "--ff-only"], target, args.dry_run) != 0
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone"]
        if not args.full:
            command.extend(["--depth", "1"])
        command.extend([repo["url"], str(target)])
        failures += run(command, EXTERNAL_ROOT, args.dry_run) != 0

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
