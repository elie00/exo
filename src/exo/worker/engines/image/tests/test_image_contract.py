"""An absent optional model must not turn into a successful fake generation."""

from io import BytesIO

import pytest
from PIL import Image

from exo.worker.engines.image import image_generator
from exo.worker.engines.image.image_generator import MfluxBackend


def unavailable_module(name: str):
    raise ImportError(f"Test fixture: {name} unavailable")


async def test_missing_image_backend_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_generator.importlib, "import_module", unavailable_module)
    backend = MfluxBackend("flux-schnell")
    with pytest.raises(ImportError, match="mflux is required"):
        await backend.load_model()
    assert not backend.loaded
    with pytest.raises(RuntimeError, match="not been loaded"):
        await backend.generate("test", width=32, height=32)


async def test_explicit_placeholder_is_a_decodable_image() -> None:
    backend = MfluxBackend("placeholder")
    await backend.load_model()
    result = await backend.generate("UI preview", width=64, height=48, seed=0)
    with Image.open(BytesIO(result.data)) as image:
        assert image.size == (64, 48)
        assert image.format == "PNG"
    assert result.seed == 0


@pytest.mark.parametrize(
    "model,response_format",
    [("stable-diffusion-xl", "b64_json"), ("flux-schnell", "url")],
)
async def test_unimplemented_variants_fail_before_loading(
    model: str, response_format: str
) -> None:
    generator = image_generator.ImageGenerator()
    request = image_generator.ImageGenerationRequest(
        model=model, prompt="test", response_format=response_format
    )
    with pytest.raises(ValueError):
        await generator.generate(request)
    assert generator.backends == {}


def test_default_image_model_matches_supported_backend() -> None:
    request = image_generator.ImageGenerationRequest(prompt="test")
    assert request.model == "flux-schnell"
