import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import ginfinity.api as api_module

from ginfinity import (Ginfinity, ModelIntegrityError, RNA,
                        default_alignment_parameters)


def test_bundled_model_loads_and_is_deterministic():
    encoder = Ginfinity.load()
    record = RNA("rna", "ACGUACGU", "((....))")
    first = encoder.encode(record)
    second = encoder.encode(record)
    assert first.shape == (8, 128)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert encoder.info()["parameter_count"] == 306_436


def test_checkpoint_loading_is_restricted_to_weights_only(monkeypatch):
    original = api_module.torch.load
    observed = {}

    def wrapped(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(api_module.torch, "load", wrapped)
    Ginfinity.load()
    assert observed["weights_only"] is True


def test_batch_preserves_order_and_rejects_duplicates():
    encoder = Ginfinity.load()
    first = RNA("first", "ACGU", "....")
    second = RNA("second", "GGAA", "(())")
    outputs = encoder.encode_many([first, second])
    assert [value.shape[0] for value in outputs] == [4, 4]
    with pytest.raises(ValueError, match="duplicate"):
        encoder.encode_many([first, first])


def test_cuda_requires_explicit_policy_acknowledgement():
    with pytest.raises(ValueError, match="allow_nondeterministic_cuda"):
        Ginfinity.load(device="cuda")


def test_alignment_parameters_are_complete():
    values = default_alignment_parameters()
    assert set(values) == {
        "mu", "sigma", "gamma", "score_min", "score_max",
        "gap_open", "gap_extend", "score_offset"}


def test_checkpoint_tampering_is_rejected(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src/ginfinity/data"
    destination = tmp_path / "model"
    shutil.copytree(source, destination)
    checkpoint = destination / "encoder.pt"
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        Ginfinity.load(model_dir=destination)


def test_metadata_is_json_serializable():
    json.dumps(Ginfinity.load().info())
