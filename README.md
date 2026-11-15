# DNA-MLP

[![CI](https://github.com/kapildhinwa2010-dev/DNA-MLP/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-MLP/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/DNA-MLP/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-MLP/actions/workflows/codeql.yml)

DNA quantized codec tuned for MLP weights: whole state-dict encode/decode.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

DNA-MLP is the MLP-specialized member of the DNA codec family
(lineage: the private DNA repository). encode()/decode() quantize a
float matrix into a GeneRecord; the MLP-specific part is
encode_state_dict()/decode_state_dict(): every 2-D weight and 1-D
bias is encoded independently with its own scale (biases as (1, n)
rows and returned flattened), so mixed-magnitude layers each get
their own quantization range.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from dna_mlp import encode_state_dict, decode_state_dict

records = encode_state_dict({"W1": W1, "b1": b1, "W2": W2}, bits=8)
restored = decode_state_dict(records)
```

## API

- encode(matrix, bits) / decode(record) / codec_stats(matrix, bits)
- encode_state_dict(state_dict, bits) - name -> GeneRecord
- decode_state_dict(records) - biases flattened back to (n,)

## Honest results and limitations

- Lossy: error equals the quantization step of each tensor.
- Only 1-D and 2-D tensors are accepted; conv or recurrent tensors
  need DNA-CNN / DNA-RNN.
- "Ratio" in codec_stats() is float32-bytes / encoded-bytes
  ARITHMETIC - it counts stored codes, not a real serialized file.
- No per-layer fitting: the reserv.dean-part-2 log records that
  generated-weight reconstructions of trained models were not usable
  without per-layer fitting.

## License

MIT - see [LICENSE](LICENSE).
