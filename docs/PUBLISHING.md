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
The workflow builds both distribution formats, validates them with Twine, and
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

The manual `publish-conda.yml` workflow requires:

- repository environment: `anaconda`
- repository variable: `ANACONDA_USER`
- environment secret: `ANACONDA_API_TOKEN`

Set `ANACONDA_USER` to the owner of the personal Anaconda.org channel. The
workflow builds the package, runs the recipe tests, and uploads only the conda
artifact created by that run.
