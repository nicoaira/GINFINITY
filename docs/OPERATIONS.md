# Operations guide

## Reproducible installation

Pin a release tag or full commit in the Git URL. Build a wheel once in a trusted
pipeline and promote the same wheel across environments:

```bash
python -m build GINFINITY
python -m pip install --require-hashes -r deployment-requirements.txt
```

The validated dependency family is CPython 3.10–3.12, NumPy 2.1.x, and PyTorch
2.6.x, with Safetensors 0.5–0.7 for graph-shard interchange. A deployment lock
is included at `requirements.lock`.

## Startup checks

`Ginfinity.load()` verifies the checkpoint SHA-256 and its architecture metadata
before loading weights. Treat a `ModelIntegrityError` as a deployment failure;
do not retry with integrity checking disabled.

## Resource planning

Encoding is linear in RNA length. Use `encode_many` and tune
`max_batch_nodes` to fit the available CPU/GPU memory. The default of 60,000
nodes is conservative for the 306,436-parameter model.

For staged execution, use `build-graphs` on CPU workers and `embed-graphs` on
accelerator workers. Graph content checksums and full numerical validation are
opt-in so trusted local or shared-storage pipelines do not pay for a second
full-artifact scan.

Local Smith–Waterman alignment is quadratic in the two sequence lengths. Its
resource controls live in the separate `ginfinity-sw` package.

## Observability

The CLI manifest records package/model versions, model hash, normalized input
hash, output hash, device, record dimensions, and elapsed time. Store it with
the `.npz` output for traceability.

## Concurrency

A loaded instance is safe for serialized inference. For concurrent request
handling, give each worker its own `Ginfinity` instance or protect access to a
shared instance with the service framework's execution lock.

## Upgrade procedure

1. Pin the new package version in a staging environment.
2. Confirm `ginfinity info` and startup integrity checks.
3. Replay representative inputs and compare shapes, finite values, and expected
   application behavior.
4. Promote the already-built wheel and retain the prior version for rollback.
