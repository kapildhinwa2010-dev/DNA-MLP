# DNA-MLP - DNA codec tuned for MLP weight matrices
# Part of the Kapildhinwa2010-dev Toolkit.

"""DNA-Net - encode/decode codec for weight matrices ("genes").

Lineage: the DNA repository. A float matrix is encoded into a compact gene
record (quantized codes + scale metadata) and decoded back; the codec is
deterministic and supports lossless round-trips within quantization error.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "encode", "decode", "GeneRecord", "codec_stats",
    "encode_state_dict", "decode_state_dict",
]


class GeneRecord:
    """Compact encoded representation of one weight matrix."""

    __slots__ = ("codes", "lo", "scale", "shape", "bits")

    def __init__(self, codes: np.ndarray, lo: float, scale: float, shape, bits: int) -> None:
        self.codes = codes
        self.lo = lo
        self.scale = scale
        self.shape = shape
        self.bits = bits

    def nbytes(self) -> int:
        return int(self.codes.nbytes + 16)

    def to_dict(self) -> dict:
        return {
            "codes": self.codes,
            "lo": self.lo,
            "scale": self.scale,
            "shape": self.shape,
            "bits": self.bits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneRecord":
        return cls(data["codes"], float(data["lo"]), float(data["scale"]),
                   tuple(data["shape"]), int(data["bits"]))


def encode(matrix: np.ndarray, bits: int = 8) -> GeneRecord:
    """Quantize a matrix into a GeneRecord."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo = float(matrix.min())
    hi = float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    dtype = np.uint16 if bits > 8 else np.uint8
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels).astype(dtype)
    return GeneRecord(codes, lo, scale, matrix.shape, bits)


def decode(record: GeneRecord) -> np.ndarray:
    """Reconstruct the matrix from a GeneRecord."""
    if not isinstance(record, GeneRecord):
        raise TypeError("decode expects a GeneRecord")
    values = np.asarray(record.codes, dtype=np.float64) * record.scale + record.lo
    return values.reshape(record.shape)


def codec_stats(matrix: np.ndarray, bits: int = 8) -> dict:
    """Report compression statistics for encoding ``matrix``."""
    record = encode(matrix, bits=bits)
    original_bytes = matrix.size * 4  # float32
    return {
        "original_bytes": int(original_bytes),
        "encoded_bytes": record.nbytes(),
        "ratio": original_bytes / record.nbytes(),
        "max_abs_error": float(np.abs(matrix - decode(record)).max()),
        "bits": bits,
    }


def encode_state_dict(state_dict: dict, bits: int = 8) -> dict:
    """Encode an MLP state dict (2-D weights, 1-D biases) record-by-record.

    Each tensor is encoded independently with its own scale: weight
    matrices keep their 2-D shape, biases are encoded as (1, n) rows.
    Returns a dict mapping parameter name to GeneRecord.
    """
    records = {}
    for name, tensor in state_dict.items():
        arr = np.asarray(tensor, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("state_dict tensors must be 1-D or 2-D: " + name)
        records[name] = encode(arr, bits=bits)
    return records


def decode_state_dict(records: dict) -> dict:
    """Reconstruct a state dict from encode_state_dict output.

    Biases encoded as (1, n) are returned flattened to (n,).
    """
    out = {}
    for name, record in records.items():
        arr = decode(record)
        if arr.shape[0] == 1 and len(record.shape) == 2 and record.shape[0] == 1:
            arr = arr.reshape(-1)
        out[name] = arr
    return out