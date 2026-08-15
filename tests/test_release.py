import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _release():
    path = ROOT / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("ginfinity_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_uses_v_prefix():
    assert _release().release_tag("1.1.0") == "v1.1.0"


def test_collect_versions_agree_in_this_checkout():
    module = _release()
    versions = module.collect_versions(ROOT)
    assert len(set(versions.values())) == 1
    assert versions["pyproject.toml"] == module.project_version(ROOT)
    assert module.version_errors(ROOT) == []


def test_extract_changelog_returns_the_named_section():
    module = _release()
    text = (
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "- pending\n\n"
        "## 1.1.0 - 2026-08-15\n\n"
        "- sliced graphs\n\n"
        "## 1.0.1 - 2026-08-13\n\n"
        "- republish\n"
    )
    assert module.extract_changelog(text, "1.1.0") == "- sliced graphs"
    assert module.extract_changelog(text, "9.9.9") is None


def test_default_notes_include_install_commands():
    module = _release()
    notes = module.default_notes(ROOT, module.project_version(ROOT))
    version = module.project_version(ROOT)
    assert f"pip install ginfinity=={version}" in notes
    assert f"ginfinity={version}" in notes


def test_replace_requires_retry_conda():
    module = _release()
    assert module.main(["--replace"]) == 2


def test_parse_args_defaults():
    args = _release().parse_args([])
    assert args.dry_run is False
    assert args.retry_conda is False
    assert args.replace is False
    assert args.skip_tests is False
