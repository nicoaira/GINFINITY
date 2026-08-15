# Publishing GINFINITY

The repository produces a platform-independent wheel, a source distribution,
and a `noarch: python` conda package.

## Release script

`scripts/release.py` is the supported way to start a publish. Keep it in
the git tree: it holds no credentials, and every checkout should run the
same checks. It does **not** upload to PyPI or Anaconda.org. It verifies
the tree, creates `v{version}`, and runs `gh release create`. That
release event starts both GitHub Actions workflows.

Before running it, bump the version in `pyproject.toml`,
`conda-recipe/meta.yaml`, `src/ginfinity/__init__.py`, and
`src/ginfinity/data/model.json` `package_version`, add a `CHANGELOG.md`
section, run `python scripts/update_manifest.py`, commit, and push
`main`.

```bash
python scripts/release.py --dry-run
python scripts/release.py
```

The script refuses a dirty tree, a branch other than `main`, a `main`
that does not match `origin/main`, mismatched version files, a stale
`PACKAGE_MANIFEST.json`, or a tag that already exists. It runs pytest
and `python -m build` / `twine check` unless you pass `--skip-tests` or
`--skip-build`. Use `--yes` in a non-interactive shell.

To retry only Anaconda.org after the GitHub release exists:

```bash
python scripts/release.py --retry-conda
python scripts/release.py --retry-conda --replace   # overwrite that version
```

## Release checks

The script above runs these. To run them by hand from the repository root:

```bash
python scripts/update_manifest.py
python scripts/update_manifest.py --check
python -m pytest
python -m build
python -m twine check --strict dist/*
```

Inspect the wheel before upload and confirm that it contains `encoder.pt`,
`LICENSE`, and `LICENSE-WEIGHTS`. Install the wheel and source distribution in
clean Python 3.10–3.12 environments before publishing.

## PyPI

The GitHub workflow in `.github/workflows/publish-pypi.yml` uses PyPI Trusted
Publishing. Configure a PyPI publisher with these values:

- owner: `nicoaira`
- repository: `GINFINITY`
- workflow: `publish-pypi.yml`
- environment: `pypi`

Publish a GitHub release only after its tag points to the exact release commit.
The tag name must be `v` plus the version in `pyproject.toml`, for example
`v1.1.0`. Prefer `python scripts/release.py`, which creates that release.
The same release also starts the Anaconda.org workflow. The PyPI
workflow builds both distribution formats, validates them with Twine, and
publishes without a long-lived PyPI token.

For a manual first upload, use an account-scoped PyPI API token and run:

```bash
python -m twine upload dist/*
```

Never commit API tokens or a credential-bearing `.pypirc` file.

## TestPyPI

Register a TestPyPI Trusted Publisher with these values:

- owner: `nicoaira`
- repository: `GINFINITY`
- workflow: `publish-testpypi.yml`
- environment: `testpypi`

Run the **Publish to TestPyPI** workflow manually and retain its default
`v1.0.1` source ref. Test installation requires PyPI as an additional index for
runtime dependencies:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  ginfinity==1.0.1
```

## Personal Anaconda channel

Build and test the recipe locally with:

```bash
conda build conda-recipe --output-folder conda-dist -c conda-forge
```

The `publish-conda.yml` workflow publishes to the personal Anaconda.org
channel `nicolas.aira`. It requires:

- repository environment: `anaconda`
- repository variable: `ANACONDA_USER=nicolas.aira`
- environment secret: `ANACONDA_API_TOKEN`

It starts automatically when a GitHub release is published, from that release
tag. The tag, `pyproject.toml`, and `conda-recipe/meta.yaml` must all carry the
same version. A manual retry is available from **Actions → Publish to
Anaconda.org → Run workflow**. Start it from `main` and set **Git tag to
build and publish** to the release tag, for example `v1.0.1`. Enable
**Replace an existing Anaconda.org build of this version** only when you
intend to overwrite that version.

Users install with:

```bash
conda install -c nicolas.aira -c conda-forge ginfinity
```

## Previous 1.0.0 artifacts

`1.0.1` is the first tag that PyPI and Anaconda.org share. Existing `1.0.0`
files stay published. Yank PyPI `1.0.0` if it should no longer be installed.
Anaconda.org can replace its `1.0.0` build from the `v1.0.0` tag with the
workflow replace option; prefer installing `1.0.1`.
