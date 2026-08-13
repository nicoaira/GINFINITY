#!/usr/bin/env python3
"""Regenerate or verify PACKAGE_MANIFEST.json for this source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = {
    ".git", ".pytest_cache", ".venv", "__pycache__", "build",
    "conda-dist", "dist",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "PACKAGE_MANIFEST.json"
        and not any(part in EXCLUDED_DIRECTORIES for part in path.parts)
        and not any(part.endswith(".egg-info") for part in path.parts)
        and not path.name.endswith((".pyc", ".pyo", "~"))
    )


def _project_identity() -> tuple[str, str]:
    text = (ROOT / "pyproject.toml").read_text()
    section = re.search(
        r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ValueError("pyproject.toml has no [project] table")
    values = {}
    for key in ("name", "version"):
        match = re.search(
            rf'(?m)^{key}\s*=\s*"([^"]+)"\s*$', section.group(1))
        if match is None:
            raise ValueError(f"pyproject.toml [project] has no {key}")
        values[key] = match.group(1)
    return values["name"], values["version"]


def _manifest() -> dict:
    name, version = _project_identity()
    return {
        "package": name,
        "version": version,
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in _source_files()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="fail instead of writing when the manifest is stale",
    )
    args = parser.parse_args()
    path = ROOT / "PACKAGE_MANIFEST.json"
    rendered = json.dumps(_manifest(), indent=2) + "\n"
    if args.check:
        if not path.is_file() or path.read_text() != rendered:
            raise SystemExit(
                "PACKAGE_MANIFEST.json is stale; run scripts/update_manifest.py")
        print("PACKAGE_MANIFEST.json is current")
        return 0
    path.write_text(rendered)
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
