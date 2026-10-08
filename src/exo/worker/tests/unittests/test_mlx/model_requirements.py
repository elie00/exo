"""Artifact preconditions for optional, real-model distributed tests."""

from pathlib import Path

from exo.worker.download.download_utils import ModelSafetensorsIndex


def missing_gpt_oss_files(model_path: Path) -> list[str]:
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        # An existing but invalid index is an error, not a reason to skip.
        index = ModelSafetensorsIndex.model_validate_json(index_path.read_bytes())
        if not index.weight_map:
            raise ValueError("GPT-OSS index has no weights")
        required.update(index.weight_map.values())
    return sorted(
        name
        for name in required
        if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0
    )
