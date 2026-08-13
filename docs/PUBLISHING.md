# Publishing GINFINITY

The repository produces a platform-independent wheel, a source distribution,
and a `noarch: python` conda package.

## Release checks

Run from the repository root:

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
`v1.0.0`. The same release also starts the Anaconda.org workflow. The PyPI
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
`v1.0.0` source ref. Test installation requires PyPI as an additional index for
runtime dependencies:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  ginfinity==1.0.0
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
same version. A manual **Actions → Publish to Anaconda.org → Run workflow**
retry is available; choose the release tag, not `main`. Enable **Replace an
existing Anaconda.org build of this version** only when you intend to overwrite
that version.

Users install with:

```bash
conda install -c nicolas.aira -c conda-forge ginfinity
```

## Replacing an already published 1.0.0

Anaconda.org can replace a build. Point `v1.0.0` at the commit you want, then
run **Publish to Anaconda.org** on that tag with the replace option enabled.

PyPI does not allow replacing files for a version that already exists. Yank
`1.0.0` if it must not be installed, then publish a new version such as
`1.0.1` from a new tag. Do not expect every current `1.0.0` artifact to become
bit-identical on PyPI.
