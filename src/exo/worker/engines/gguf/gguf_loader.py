"""GGUF file format loader and parser.

This module parses GGUF files to extract model metadata, layer information,
and tensor specifications needed for distributed inference sharding.

GGUF Format Reference:
- Magic: "GGUF" (4 bytes)
- Version: uint32
- Tensor count: uint64
- Metadata KV count: uint64
- Metadata key-value pairs
- Tensor info
- Tensor data
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO, Optional, cast

from loguru import logger

type GGUFValue = str | int | float | bool | list[GGUFValue]


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """GGML tensor quantization types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 29


@dataclass
class TensorInfo:
    """Information about a tensor in the GGUF file."""

    name: str
    n_dims: int
    dims: list[int]
    dtype: GGMLType
    offset: int

    @property
    def layer_number(self) -> Optional[int]:
        """Extract layer number from tensor name if applicable."""
        # Common patterns: "blk.0.attn_q.weight", "layers.0.self_attn.q_proj.weight"
        import re

        match = re.search(r"(?:blk|layers?|block)\.(\d+)\.", self.name)
        if match:
            return int(match.group(1))
        return None

    @property
    def is_layer_tensor(self) -> bool:
        """Check if this tensor belongs to a specific layer."""
        return self.layer_number is not None


@dataclass
class GGUFModelInfo:
    """Complete information about a GGUF model."""

    path: Path
    version: int
    tensor_count: int
    architecture: str
    quantization: Optional[str]
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    context_length: int
    rope_freq_base: float
    metadata: dict[str, GGUFValue] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        """Get total file size."""
        return self.path.stat().st_size

    def get_layer_tensors(self, layer_idx: int) -> list[TensorInfo]:
        """Get all tensors belonging to a specific layer."""
        return [t for t in self.tensors if t.layer_number == layer_idx]

    def get_layers_range(self, start_layer: int, end_layer: int) -> list[TensorInfo]:
        """Get tensors for a range of layers (for sharding)."""
        return [
            t
            for t in self.tensors
            if t.layer_number is not None and start_layer <= t.layer_number < end_layer
        ]

    def get_non_layer_tensors(self) -> list[TensorInfo]:
        """Get tensors that don't belong to a specific layer (embeddings, output, etc.)."""
        return [t for t in self.tensors if t.layer_number is None]


def _read_exact(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated GGUF: expected {size} bytes, got {len(data)}")
    return data


def _read_integer(f: BinaryIO, size: int, *, signed: bool = False) -> int:
    return int.from_bytes(_read_exact(f, size), byteorder="little", signed=signed)


def _read_string(f: BinaryIO) -> str:
    length = _read_integer(f, 8)
    return _read_exact(f, length).decode("utf-8")


def _read_value(f: BinaryIO, value_type: GGUFValueType) -> GGUFValue:
    integer_sizes = {
        GGUFValueType.UINT8: (1, False),
        GGUFValueType.INT8: (1, True),
        GGUFValueType.UINT16: (2, False),
        GGUFValueType.INT16: (2, True),
        GGUFValueType.UINT32: (4, False),
        GGUFValueType.INT32: (4, True),
        GGUFValueType.UINT64: (8, False),
        GGUFValueType.INT64: (8, True),
    }
    if value_type in integer_sizes:
        size, signed = integer_sizes[value_type]
        return _read_integer(f, size, signed=signed)
    if value_type == GGUFValueType.FLOAT32:
        return cast(tuple[float], struct.unpack("<f", _read_exact(f, 4)))[0]
    if value_type == GGUFValueType.FLOAT64:
        return cast(tuple[float], struct.unpack("<d", _read_exact(f, 8)))[0]
    if value_type == GGUFValueType.BOOL:
        return _read_integer(f, 1) != 0
    if value_type == GGUFValueType.STRING:
        return _read_string(f)
    if value_type == GGUFValueType.ARRAY:
        elem_type = GGUFValueType(_read_integer(f, 4))
        count = _read_integer(f, 8)
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF value type: {value_type}")


def _metadata_int(metadata: dict[str, GGUFValue], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"GGUF metadata {key} must be an integer")
            return value
    return default


def _metadata_float(
    metadata: dict[str, GGUFValue], *keys: str, default: float
) -> float:
    for key in keys:
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"GGUF metadata {key} must be numeric")
            return float(value)
    return default


def load_gguf_metadata(path: Path, load_tensors: bool = False) -> GGUFModelInfo:
    """Load and parse GGUF file metadata.

    Args:
        path: Path to the GGUF file
        load_tensors: If True, also load tensor info (slower but needed for sharding)

    Returns:
        GGUFModelInfo with all extracted metadata
    """
    with open(path, "rb") as f:
        # Read magic
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic}")

        # Read version
        version = _read_integer(f, 4)
        if version < 2:
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Read counts
        tensor_count = _read_integer(f, 8)
        metadata_kv_count = _read_integer(f, 8)

        logger.debug(
            f"GGUF v{version}: {tensor_count} tensors, {metadata_kv_count} metadata keys"
        )

        # Read metadata
        metadata: dict[str, GGUFValue] = {}
        for _ in range(metadata_kv_count):
            key = _read_string(f)
            value_type = GGUFValueType(_read_integer(f, 4))
            value = _read_value(f, value_type)
            metadata[key] = value

        # Extract common fields with defaults
        architecture = metadata.get("general.architecture", "unknown")

        if not isinstance(architecture, str):
            raise ValueError("GGUF general.architecture must be a string")
        n_layers = _metadata_int(
            metadata,
            f"{architecture}.block_count",
            f"{architecture}.num_hidden_layers",
            "general.block_count",
        )
        n_heads = _metadata_int(
            metadata,
            f"{architecture}.attention.head_count",
            f"{architecture}.num_attention_heads",
        )
        n_kv_heads = _metadata_int(
            metadata,
            f"{architecture}.attention.head_count_kv",
            f"{architecture}.num_key_value_heads",
            default=n_heads,
        )
        hidden_size = _metadata_int(
            metadata, f"{architecture}.embedding_length", f"{architecture}.hidden_size"
        )
        intermediate_size = _metadata_int(
            metadata,
            f"{architecture}.feed_forward_length",
            f"{architecture}.intermediate_size",
        )
        vocab_size = _metadata_int(metadata, f"{architecture}.vocab_size")
        context_length = _metadata_int(
            metadata,
            f"{architecture}.context_length",
            "general.context_length",
            default=4096,
        )
        rope_freq_base = _metadata_float(
            metadata,
            f"{architecture}.rope.freq_base",
            f"{architecture}.rope_freq_base",
            default=10000.0,
        )
        file_type = _metadata_int(metadata, "general.file_type")
        quantization = _file_type_to_quant_name(file_type)

        # Load tensor info if requested
        tensors: list[TensorInfo] = []
        if load_tensors:
            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = _read_integer(f, 4)
                dims = [_read_integer(f, 8) for _ in range(n_dims)]
                dtype = GGMLType(_read_integer(f, 4))
                offset = _read_integer(f, 8)

                tensors.append(
                    TensorInfo(
                        name=name,
                        n_dims=n_dims,
                        dims=dims,
                        dtype=dtype,
                        offset=offset,
                    )
                )

        return GGUFModelInfo(
            path=path,
            version=version,
            tensor_count=tensor_count,
            architecture=architecture,
            quantization=quantization,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            context_length=context_length,
            rope_freq_base=rope_freq_base,
            metadata=metadata,
            tensors=tensors,
        )


def _file_type_to_quant_name(file_type: int) -> Optional[str]:
    """Convert GGUF file type to quantization name."""
    mapping = {
        0: "ALL_F32",
        1: "MOSTLY_F16",
        2: "MOSTLY_Q4_0",
        3: "MOSTLY_Q4_1",
        7: "MOSTLY_Q8_0",
        8: "MOSTLY_Q5_0",
        9: "MOSTLY_Q5_1",
        10: "MOSTLY_Q2_K",
        11: "MOSTLY_Q3_K_S",
        12: "MOSTLY_Q3_K_M",
        13: "MOSTLY_Q3_K_L",
        14: "MOSTLY_Q4_K_S",
        15: "MOSTLY_Q4_K_M",
        16: "MOSTLY_Q5_K_S",
        17: "MOSTLY_Q5_K_M",
        18: "MOSTLY_Q6_K",
        19: "MOSTLY_IQ2_XXS",
        20: "MOSTLY_IQ2_XS",
        21: "MOSTLY_IQ3_XXS",
        22: "MOSTLY_IQ1_S",
        23: "MOSTLY_IQ4_NL",
        24: "MOSTLY_IQ3_S",
        25: "MOSTLY_IQ2_S",
        26: "MOSTLY_IQ4_XS",
        27: "MOSTLY_IQ1_M",
        28: "MOSTLY_BF16",
    }
    return mapping.get(file_type)


def get_layer_count(path: Path) -> int:
    """Quick function to get layer count from a GGUF file.

    Args:
        path: Path to the GGUF file

    Returns:
        Number of layers in the model
    """
    info = load_gguf_metadata(path, load_tensors=False)
    return info.n_layers


async def load_gguf_metadata_async(
    path: Path, load_tensors: bool = False
) -> GGUFModelInfo:
    """Async version of load_gguf_metadata."""
    import asyncio

    return await asyncio.get_event_loop().run_in_executor(
        None, load_gguf_metadata, path, load_tensors
    )
