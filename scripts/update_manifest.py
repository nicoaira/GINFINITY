#!/usr/bin/env python3
"""Regenerate or verify PACKAGE_MANIFEST.json for this source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
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


def _manifest() -> dict:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return {
        "package": project["name"],
        "version": project["version"],
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
