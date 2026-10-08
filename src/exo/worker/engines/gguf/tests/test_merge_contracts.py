"""CPU-only regression tests; these do not qualify llama.cpp or real GPUs."""

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from exo.worker.engines.gguf.activation_wire import ActivationEnvelope, ArrayPayload
from exo.worker.engines.gguf.distributed_pipeline import ActivationPacket
from exo.worker.engines.gguf.exo_integration import ollama_model_to_metadata
from exo.worker.engines.gguf.gguf_loader import load_gguf_metadata
from exo.worker.engines.gguf.llama_backend import LlamaModel
from exo.worker.engines.gguf.ollama_discovery import OllamaManifest, OllamaModel
from exo.worker.engines.gguf.ollama_distributed import (
    DistributedOllamaCluster,
    OllamaNode,
)


class FakeLlama:
    def __init__(self) -> None:
        self.closed = False
        self.stream = False

    def create_completion(self, prompt: str, **kwargs: object) -> object:
        self.stream = kwargs.get("stream") is True
        value = {"choices": [{"text": prompt, "finish_reason": "stop"}]}
        return iter([value]) if self.stream else value

    def create_chat_completion(self, **kwargs: object) -> object:
        self.stream = kwargs.get("stream") is True
        value = {
            "choices": [
                {"message": {"content": "hello"}, "delta": {"content": "hello"}}
            ]
        }
        return iter([value]) if self.stream else value

    def close(self) -> None:
        self.closed = True


def test_llama_boundary_handles_both_stream_modes_and_closes() -> None:
    raw = FakeLlama()
    model = LlamaModel(raw)
    assert model.complete("prompt").choices[0].text == "prompt"
    assert not raw.stream
    assert [chunk.choices[0].text for chunk in model.complete_stream("prompt")] == [
        "prompt"
    ]
    assert raw.stream
    assert (
        model.chat([{"role": "user", "content": "hello"}]).choices[0].message.content
        == "hello"
    )
    assert not raw.stream
    assert next(model.chat_stream([])).choices[0].delta.content == "hello"
    assert raw.stream
    model.close()
    assert raw.closed


class InvalidLlama(FakeLlama):
    def create_completion(self, prompt: str, **kwargs: object) -> object:
        invalid: dict[str, list[object]] = {"choices": []}
        return iter([invalid]) if kwargs.get("stream") else invalid


def test_llama_boundary_rejects_invalid_payloads() -> None:
    model = LlamaModel(InvalidLlama())
    with pytest.raises(ValidationError):
        model.complete("hello")
    with pytest.raises(ValidationError):
        list(model.complete_stream("hello"))


def test_activation_wire_roundtrip_preserves_arrays_and_metadata() -> None:
    packet = ActivationPacket(
        request_id="test",
        sequence_position=3,
        hidden_states=np.arange(12, dtype=np.float32).reshape(3, 4),
        position_ids=np.array([0, 1, 2], dtype=np.int64),
        attention_mask=np.array([True, False]),
        is_final=True,
        metadata={"token": "é", "finish_reason": "stop"},
    )
    framed = packet.serialize()
    assert int.from_bytes(framed[:4], "big") == len(framed) - 4
    restored = ActivationPacket.deserialize(framed[4:])
    np.testing.assert_array_equal(restored.hidden_states, packet.hidden_states)
    np.testing.assert_array_equal(restored.position_ids, packet.position_ids)
    np.testing.assert_array_equal(restored.attention_mask, packet.attention_mask)
    assert restored.hidden_states.dtype == packet.hidden_states.dtype
    assert restored.request_id == "test"
    assert restored.sequence_position == 3
    assert restored.is_final
    assert restored.metadata == packet.metadata


def test_activation_rejects_pickle_and_unsafe_array_types() -> None:
    with pytest.raises(ValidationError):
        ActivationPacket.deserialize(pickle.dumps({"request_id": "old wire format"}))
    with pytest.raises(ValueError, match="numeric"):
        ArrayPayload.from_array(np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="dtypes"):
        ArrayPayload(shape=[1], dtype="O", data="AAAAAAAAAAA=").to_array()
    with pytest.raises(ValueError, match="length"):
        ArrayPayload(shape=[999999999], dtype="<f4", data="AAAAAA==").to_array()
    envelope = ActivationEnvelope(
        request_id="test",
        sequence_position=0,
        hidden_states=ArrayPayload.from_array(np.array([1])),
    )
    data = envelope.model_dump()
    data["version"] = 2
    with pytest.raises(ValidationError):
        ActivationPacket.deserialize(json.dumps(data).encode())


def write_gguf(path: Path, *, invalid_layer_type: bool = False) -> None:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = (
        string("general.architecture")
        + struct.pack("<I", 8)
        + string("llama")
        + string("llama.block_count")
        + (
            struct.pack("<I", 8) + string("four")
            if invalid_layer_type
            else struct.pack("<II", 4, 4)
        )
        + string("llama.embedding_length")
        + struct.pack("<II", 4, 128)
    )
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 3) + metadata)


def test_gguf_metadata_and_model_card_use_actual_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "small.gguf"
    write_gguf(path)
    info = load_gguf_metadata(path)
    assert (info.architecture, info.n_layers, info.hidden_size) == ("llama", 4, 128)
    card = ollama_model_to_metadata(
        OllamaModel("test", "latest", path.stat().st_size, path, "test"), info
    )
    assert card.model_id == "ollama/test:latest"
    assert card.n_layers == 4
    assert card.hidden_size == 128
    assert card.storage_size.in_bytes == path.stat().st_size


def test_gguf_invalid_dimensions_and_truncation_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    write_gguf(path, invalid_layer_type=True)
    with pytest.raises(ValueError, match="must be an integer"):
        load_gguf_metadata(path)
    path.write_bytes(b"GGUF\x03")
    with pytest.raises(ValueError, match="Truncated"):
        load_gguf_metadata(path)
    with pytest.raises(ValueError, match="readable, positive"):
        ollama_model_to_metadata(OllamaModel("test", "latest", 5, path, "test"))


def test_manifest_rejects_path_as_digest() -> None:
    with pytest.raises(ValidationError):
        OllamaManifest.model_validate_json(
            '{"layers":[{"mediaType":"model","digest":"../../outside","size":1}]}'
        )


class RecordingOllamaNode(OllamaNode):
    def __init__(self) -> None:
        super().__init__(host="localhost", model_name="test")
        self.arguments: list[tuple[str, str | None, int, float]] = []

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ):
        self.arguments.append((prompt, model, max_tokens, temperature))
        yield prompt.upper()


async def test_parallel_ollama_passes_model_and_sampling_arguments_correctly() -> None:
    first, second = RecordingOllamaNode(), RecordingOllamaNode()
    cluster = DistributedOllamaCluster(nodes=[first, second], model_name="test")
    assert await cluster.generate_parallel(
        ["one", "two", "three"], max_tokens=12, temperature=0.25
    ) == ["ONE", "TWO", "THREE"]
    assert first.arguments == [("one", None, 12, 0.25), ("three", None, 12, 0.25)]
    assert second.arguments == [("two", None, 12, 0.25)]


async def test_rpc_workers_cannot_silently_run_as_local_inference(
    tmp_path: Path,
) -> None:
    from exo.worker.engines.gguf.gguf_inference import (
        GGUFInferenceEngine,
        GGUFNodeRole,
        GGUFRPCConfig,
    )

    engine = GGUFInferenceEngine(
        GGUFRPCConfig(
            model_path=tmp_path / "missing.gguf",
            role=GGUFNodeRole.MAIN,
            worker_addresses=["127.0.0.1:50052"],
        )
    )
    with pytest.raises(NotImplementedError, match="not connected"):
        await engine.start()
    assert not engine.is_running
