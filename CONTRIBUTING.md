# Contributing

Changes must preserve the public input contract, checkpoint integrity checks,
and deterministic CPU output for an identical execution layout.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . pytest build
.venv/bin/python -m pytest -q
.venv/bin/python -m build
```

Do not replace `src/ginfinity/data/encoder.pt` manually. Model artifacts and
their JSON metadata must be generated together by the repository release
builder, reviewed by hash, and exercised through wheel installation before
merge. Add a changelog entry for every user-visible change.
