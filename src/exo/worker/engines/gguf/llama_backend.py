"""Validated boundary around the optional llama-cpp-python dependency.

Importing EXO does not require loading a CUDA/Metal shared library. Callers load
this dependency explicitly and validate its OpenAI-style completion payloads.
"""

import importlib
from collections.abc import Iterator
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class CompletionMessage(BaseModel):
    content: str | None = None
    model_config = ConfigDict(strict=True)


class CompletionChoice(BaseModel):
    text: str = ""
    finish_reason: Literal["stop", "length", "tool_calls", "function_call"] | None = (
        None
    )
    message: CompletionMessage = Field(default_factory=CompletionMessage)
    delta: CompletionMessage = Field(default_factory=CompletionMessage)
    model_config = ConfigDict(strict=True)


class Completion(BaseModel):
    choices: list[CompletionChoice] = Field(min_length=1)
    model_config = ConfigDict(strict=True)


@runtime_checkable
class RawLlamaModel(Protocol):
    def create_completion(self, prompt: str, **kwargs: object) -> object: ...
    def create_chat_completion(self, **kwargs: object) -> object: ...
    def close(self) -> None: ...


class LlamaFactory(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@runtime_checkable
class LlamaModule(Protocol):
    Llama: LlamaFactory


def load_llama_model(**kwargs: object) -> "LlamaModel":
    try:
        module = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise ImportError(
            "Install the optional llama-cpp-python backend to run GGUF inference"
        ) from exc
    if not isinstance(module, LlamaModule):
        raise TypeError("llama_cpp does not expose the expected Llama factory")
    model = module.Llama(**kwargs)
    if not isinstance(model, RawLlamaModel):
        raise TypeError(
            "The llama_cpp model is missing completion or lifecycle methods"
        )
    return LlamaModel(model)


class LlamaModel:
    def __init__(self, model: RawLlamaModel):
        self._model = model

    def complete(self, prompt: str, **kwargs: object) -> Completion:
        return Completion.model_validate(
            self._model.create_completion(prompt, stream=False, **kwargs)
        )

    def complete_stream(self, prompt: str, **kwargs: object) -> Iterator[Completion]:
        yield from self._validated_stream(
            self._model.create_completion(prompt, stream=True, **kwargs)
        )

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> Completion:
        return Completion.model_validate(
            self._model.create_chat_completion(
                messages=messages, stream=False, **kwargs
            )
        )

    def chat_stream(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> Iterator[Completion]:
        yield from self._validated_stream(
            self._model.create_chat_completion(messages=messages, stream=True, **kwargs)
        )

    @staticmethod
    def _validated_stream(value: object) -> Iterator[Completion]:
        if not isinstance(value, Iterator):
            raise TypeError("llama_cpp stream response must be an iterator")
        # The protocol check establishes iteration only; each item remains an
        # untrusted object until Pydantic validates its response structure.
        for item in cast(Iterator[object], value):
            yield Completion.model_validate(item)

    def close(self) -> None:
        self._model.close()
