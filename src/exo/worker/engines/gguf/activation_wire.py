"""Versioned activation payloads; never unpickle data received from a socket."""

import base64
from math import prod
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

MAX_PACKET_BYTES = 64 * 1024 * 1024


class ArrayPayload(BaseModel):
    shape: list[Annotated[int, Field(ge=0)]] = Field(max_length=8)
    dtype: str
    data: str = Field(max_length=MAX_PACKET_BYTES)
    model_config = ConfigDict(strict=True, extra="forbid")

    @classmethod
    def from_array(cls, value: np.ndarray) -> Self:
        if value.dtype.kind not in "biuf":
            raise ValueError("Only numeric or boolean activation arrays are supported")
        if value.nbytes > MAX_PACKET_BYTES:
            raise ValueError("Activation array exceeds packet size limit")
        return cls(
            shape=list(value.shape),
            dtype=value.dtype.str,
            data=base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        )

    def to_array(self) -> np.ndarray:
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biuf" or dtype.fields is not None:
            raise ValueError("Object and structured array dtypes are not allowed")
        raw = base64.b64decode(self.data, validate=True)
        if prod(self.shape) * dtype.itemsize != len(raw):
            raise ValueError("Activation shape and data length disagree")
        return np.frombuffer(raw, dtype=dtype).reshape(self.shape).copy()


class ActivationEnvelope(BaseModel):
    version: Annotated[int, Field(ge=1, le=1)] = 1
    request_id: str
    sequence_position: int
    hidden_states: ArrayPayload
    attention_mask: ArrayPayload | None = None
    position_ids: ArrayPayload | None = None
    is_final: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    model_config = ConfigDict(strict=True, extra="forbid")
