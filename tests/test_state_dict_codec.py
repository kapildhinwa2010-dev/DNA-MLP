import numpy as np
import pytest

from dna_mlp import encode_state_dict, decode_state_dict


def _state_dict(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "fc1.weight": rng.standard_normal((16, 8)),
        "fc1.bias": rng.standard_normal(16),
        "fc2.weight": rng.standard_normal((4, 16)),
        "fc2.bias": rng.standard_normal(4),
    }


def test_state_dict_roundtrip_shapes():
    sd = _state_dict()
    records = encode_state_dict(sd, bits=8)
    assert set(records) == set(sd)
    out = decode_state_dict(records)
    for name, tensor in sd.items():
        assert out[name].shape == tensor.shape


def test_state_dict_roundtrip_error_small():
    sd = _state_dict()
    out = decode_state_dict(encode_state_dict(sd, bits=8))
    for name, tensor in sd.items():
        assert np.abs(tensor - out[name]).max() < 0.05


def test_state_dict_rejects_3d():
    with pytest.raises(ValueError):
        encode_state_dict({"bad": np.zeros((2, 2, 2))})


def test_state_dict_bias_flattened_back():
    sd = {"b": np.arange(5, dtype=np.float64)}
    out = decode_state_dict(encode_state_dict(sd, bits=16))
    assert out["b"].shape == (5,)
    assert np.allclose(out["b"], sd["b"])
