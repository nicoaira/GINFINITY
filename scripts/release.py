#!/usr/bin/env python3
"""Preflight a GINFINITY release and publish it by creating a GitHub release.

Uploads stay in GitHub Actions. This script only checks the tree, creates
``v{version}``, and runs ``gh release create``, which starts the PyPI and
Anaconda.org workflows.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BRANCH = "main"
INSTALL_TEMPLATE = """\
Install:

```
pip install ginfinity=={version}
conda install -c nicolas.aira -c conda-forge ginfinity={version}
```
"""


class ReleaseError(Exception):
    """A release preflight or publish step failed."""


def project_version(root: Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text()
    section = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ReleaseError("pyproject.toml has no [project] table")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', section.group(1))
    if match is None:
        raise ReleaseError("pyproject.toml [project] has no version")
    return match.group(1)


def release_tag(version: str) -> str:
    return f"v{version}"


def collect_versions(root: Path = ROOT) -> dict[str, str]:
    recipe = (root / "conda-recipe" / "meta.yaml").read_text()
    recipe_match = re.search(
        r'{% set version = "([^"]+)" %}', recipe)
    init_text = (root / "src" / "ginfinity" / "__init__.py").read_text()
    init_match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$', init_text)
    model = json.loads((root / "src" / "ginfinity" / "data" / "model.json").read_text())
    return {
        "pyproject.toml": project_version(root),
        "conda-recipe/meta.yaml": (
            recipe_match.group(1) if recipe_match else ""),
        "src/ginfinity/__init__.py": (
            init_match.group(1) if init_match else ""),
        "src/ginfinity/data/model.json": str(
            model.get("package_version") or ""),
    }


def version_errors(root: Path = ROOT) -> list[str]:
    versions = collect_versions(root)
    expected = versions["pyproject.toml"]
    errors = [
        f"{name} has {value!r}, expected {expected!r}"
        for name, value in versions.items()
        if value != expected
    ]
    if extract_changelog((root / "CHANGELOG.md").read_text(), expected) is None:
        errors.append(f"CHANGELOG.md has no '## {expected}' section")
    return errors


def extract_changelog(text: str, version: str) -> str | None:
    pattern = re.compile(
        rf"(?ms)^## {re.escape(version)}(?:[ \t][^\n]*)?\n(.*?)(?=^## |\Z)")
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def default_notes(root: Path, version: str) -> str:
    body = extract_changelog((root / "CHANGELOG.md").read_text(), version)
    if body is None:
        raise ReleaseError(f"CHANGELOG.md has no '## {version}' section")
    return (
        f"Package version {version}.\n\n{body}\n\n"
        + INSTALL_TEMPLATE.format(version=version)
    )


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            f"{' '.join(command)} failed"
            + (f": {detail}" if detail else ""))
    return result


def _git(*args: str, check: bool = True) -> str:
    return _run(["git", *args], check=check).stdout.strip()


def _require_tools(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise ReleaseError("missing executable(s): " + ", ".join(missing))


def preflight_git(root: Path = ROOT) -> str:
    _require_tools("git")
    if not (root / ".git").exists():
        raise ReleaseError(f"{root} is not a git checkout")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != CANONICAL_BRANCH:
        raise ReleaseError(f"publish from {CANONICAL_BRANCH}, not {branch}")
    status = _git("status", "--porcelain")
    dirty = [line for line in status.splitlines() if not line.startswith("??")]
    if dirty:
        raise ReleaseError(
            "working tree has uncommitted changes:\n" + "\n".join(dirty))
    untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]
    if untracked:
        print(
            "warning: untracked files (not included in the release):\n  "
            + "\n  ".join(untracked),
            file=sys.stderr,
        )
    _git("fetch", "origin", CANONICAL_BRANCH, "--tags")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", f"origin/{CANONICAL_BRANCH}")
    if head != remote:
        ahead, behind = _git(
            "rev-list", "--left-right", "--count",
            f"origin/{CANONICAL_BRANCH}...HEAD",
        ).split()
        raise ReleaseError(
            f"{CANONICAL_BRANCH} differs from origin/{CANONICAL_BRANCH} "
            f"(ahead {ahead}, behind {behind}); push or pull first")
    return head


def preflight_tag(version: str) -> str:
    tag = release_tag(version)
    local = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}", check=False)
    if local:
        raise ReleaseError(f"local tag {tag} already exists")
    remote = _git("ls-remote", "--tags", "origin", f"refs/tags/{tag}")
    if remote:
        raise ReleaseError(f"origin already has {tag}")
    return tag


def preflight_manifest() -> None:
    _run([sys.executable, str(ROOT / "scripts" / "update_manifest.py"), "--check"])


def run_tests() -> None:
    _run([sys.executable, "-m", "pytest", "-q"])


def run_build() -> None:
    _run([sys.executable, "-m", "build"])
    artifacts = sorted((ROOT / "dist").glob("ginfinity-*"))
    if not artifacts:
        raise ReleaseError("python -m build produced no ginfinity artifacts")
    _run([sys.executable, "-m", "twine", "check", "--strict", *map(str, artifacts)])


def confirm(prompt: str, *, assume_yes: bool) -> None:
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise ReleaseError("refusing to publish without a TTY; pass --yes")
    answer = input(f"{prompt} [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise ReleaseError("aborted")


def create_release(tag: str, title: str, notes: str, *, target: str) -> str:
    _require_tools("git", "gh")
    _git("tag", "-a", tag, "-m", title, target)
    try:
        _git("push", "origin", tag)
    except ReleaseError:
        _git("tag", "-d", tag, check=False)
        raise
    try:
        result = _run([
            "gh", "release", "create", tag,
            "--title", title,
            "--notes", notes,
            "--verify-tag",
        ])
    except ReleaseError as error:
        raise ReleaseError(
            f"{error}. Tag {tag} is on origin; finish with "
            f"`gh release create {tag} --verify-tag`") from error
    url = (result.stdout or "").strip()
    return url or f"https://github.com/nicoaira/GINFINITY/releases/tag/{tag}"


def retry_conda(tag: str, *, replace: bool) -> None:
    _require_tools("gh")
    command = [
        "gh", "workflow", "run", "Publish to Anaconda.org",
        "--ref", CANONICAL_BRANCH,
        "-f", f"release_ref={tag}",
        "-f", f"force={'true' if replace else 'false'}",
    ]
    _run(command)


def print_workflow_hints(head: str) -> None:
    print(
        "Publishing workflows start from the GitHub release:\n"
        "  https://github.com/nicoaira/GINFINITY/actions/workflows/publish-pypi.yml\n"
        "  https://github.com/nicoaira/GINFINITY/actions/workflows/publish-conda.yml"
    )
    listed = _run(
        ["gh", "run", "list", "--commit", head, "--limit", "10",
         "--json", "databaseId,name,url,status"],
        check=False,
    )
    if listed.returncode != 0 or not listed.stdout.strip():
        return
    try:
        runs = json.loads(listed.stdout)
    except json.JSONDecodeError:
        return
    interesting = [
        run for run in runs
        if run.get("name") in {"Publish to PyPI", "Publish to Anaconda.org", "CI"}
    ]
    if not interesting:
        return
    print("Recent runs on this commit:")
    for run in interesting:
        print(f"  {run['name']}: {run.get('url', '')} ({run.get('status', '?')})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the tree and publish ginfinity by creating a GitHub release")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run every check and print the tag, then exit")
    parser.add_argument(
        "--yes", action="store_true",
        help="do not ask for confirmation")
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="do not run pytest")
    parser.add_argument(
        "--skip-build", action="store_true",
        help="do not run python -m build / twine check")
    parser.add_argument(
        "--notes-file", type=Path,
        help="use this file as the GitHub release notes")
    parser.add_argument(
        "--retry-conda", action="store_true",
        help="re-run only the Anaconda.org workflow for the current version tag")
    parser.add_argument(
        "--replace", action="store_true",
        help="with --retry-conda, replace an existing Anaconda.org build")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.replace and not args.retry_conda:
            raise ReleaseError("--replace is only valid with --retry-conda")
        version = project_version()
        tag = release_tag(version)
        errors = version_errors()
        if errors:
            raise ReleaseError("version mismatch:\n  " + "\n  ".join(errors))
        head = preflight_git()
        if args.retry_conda:
            remote = _git("ls-remote", "--tags", "origin", f"refs/tags/{tag}")
            if not remote:
                raise ReleaseError(
                    f"origin has no {tag}; create the GitHub release first")
            print(
                f"Retry Anaconda.org publish of {tag} "
                f"(replace={args.replace})")
            if args.dry_run:
                print("dry run: not starting the workflow")
                return 0
            confirm("Start the Anaconda.org workflow?", assume_yes=args.yes)
            retry_conda(tag, replace=args.replace)
            print("started Publish to Anaconda.org")
            print_workflow_hints(head)
            return 0

        preflight_tag(version)
        preflight_manifest()
        if not args.skip_tests:
            run_tests()
        if not args.skip_build:
            run_build()
        if args.notes_file is not None:
            notes = args.notes_file.read_text()
        else:
            notes = default_notes(ROOT, version)
        title = f"GINFINITY {tag}"
        print(f"Ready to publish ginfinity {version}")
        print(f"  tag:    {tag}")
        print(f"  commit: {head}")
        print("  starts: Publish to PyPI, Publish to Anaconda.org")
        if args.dry_run:
            print("dry run: not creating the tag or GitHub release")
            print("--- release notes ---")
            print(notes)
            return 0
        confirm(f"Create GitHub release {tag}?", assume_yes=args.yes)
        url = create_release(tag, title, notes, target=head)
        print(url)
        print_workflow_hints(head)
        return 0
    except ReleaseError as error:
        print(f"release: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
