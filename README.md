# DNA-MLP

DNA-MLP is a small NumPy-based, lossy quantization codec for two-dimensional matrices and simple MLP-style parameter dictionaries. It stores quantized integer codes with per-matrix scale metadata and reconstructs arrays within quantization error. This is an early-stage utility, not a neural-network training or inference framework.

## Implemented functionality

- `encode(matrix, bits=8)` converts a non-empty 2-D array to a `GeneRecord`. Supported precision is 1–16 bits; values are uniformly quantized over that matrix's minimum-to-maximum range. Codes use `uint8` through 8 bits and `uint16` above 8 bits.
- `decode(record)` reconstructs the array from its codes and metadata.
- `encode_state_dict(state_dict, bits=8)` encodes each 1-D or 2-D value independently, with an independent range for each value. One-dimensional values are encoded as single-row matrices.
- `decode_state_dict(records)` reconstructs those values and flattens encoded single-row entries.
- `codec_stats(matrix, bits=8)` reports a rough code-size estimate, ratio, and maximum absolute error. Its “original” size assumes four bytes per element, and encoded size is code bytes plus a fixed 16-byte allowance; these are estimates, not measurements of a serialized file.
- `GeneRecord.to_dict()` and `GeneRecord.from_dict()` expose a dictionary representation containing the NumPy code array and metadata. They do not implement file or JSON serialization.

## Requirements and use

Python 3.9 or newer and NumPy 1.24 or newer are declared in `pyproject.toml`. From a clone of this repository, install the declared development/runtime requirements with:

```bash
python -m pip install -r requirements.txt
```

The module can be imported from the repository directory. For example:

```python
import numpy as np
from dna_mlp import decode, decode_state_dict, encode, encode_state_dict

weights = np.array([[0.1, -0.3], [0.8, 0.2]])
record = encode(weights, bits=8)
reconstructed = decode(record)

parameters = {
    "layer.weight": weights,
    "layer.bias": np.array([0.05, -0.1]),
}
records = encode_state_dict(parameters, bits=8)
restored_parameters = decode_state_dict(records)
```

Quantization is lossy except when the values happen to be exactly represented by the chosen levels. Increasing `bits` increases the number of quantization levels but does not make this a trained or model-aware compression scheme.

## Tests and CI

The test suite is run with:

```bash
python -m pytest -q
```

At the inspected repository commit, `ddbddbd0b83de55e03f64a9c8b65844476480966`, this command completed with **16 passed**. This is a local test run, not a claim about the status of a remote GitHub Actions run. The CI workflow defines tests, compilation, and a small import/shape smoke check on Python 3.10, 3.11, and 3.12. A separate CodeQL workflow is configured; its latest remote result was not checked.

## Limitations

- Only non-empty 2-D matrices can be passed to `encode`; state-dictionary values must be 1-D or 2-D. Convolutional or recurrent tensors and higher-dimensional arrays are not supported by this implementation.
- Each matrix or state-dictionary entry is quantized independently using its own min/max range. No model-level calibration, per-layer fitting beyond this per-entry scaling, training, or inference is implemented.
- `decode_state_dict` flattens any decoded single-row record, so a genuine 2-D value with shape `(1, n)` does not retain that shape.
- `codec_stats` is an arithmetic estimate based on a float32 input-size assumption and a fixed metadata allowance. The repository does not provide a packed file format or a benchmark demonstrating real storage savings.
- The tests exercise basic round-trips, shape handling, precision bounds, and validation; they do not establish model quality, speed, or production suitability.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for the complete terms.
