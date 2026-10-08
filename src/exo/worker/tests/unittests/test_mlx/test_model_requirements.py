import json
from pathlib import Path

import pytest

from exo.worker.tests.unittests.test_mlx.model_requirements import missing_gpt_oss_files


def test_config_only_cache_is_not_a_complete_model(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": None,
                "weight_map": {
                    "layer.0": "model-00001.safetensors",
                    "layer.1": "model-00002.safetensors",
                },
            }
        )
    )
    missing = missing_gpt_oss_files(tmp_path)
    assert missing == [
        "model-00001.safetensors",
        "model-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for filename in missing:
        (tmp_path / filename).write_bytes(b"test readiness fixture")
    assert missing_gpt_oss_files(tmp_path) == []
    (tmp_path / "model-00002.safetensors").write_bytes(b"")
    assert missing_gpt_oss_files(tmp_path) == ["model-00002.safetensors"]


def test_invalid_existing_index_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata": null, "weight_map": {}}'
    )
    with pytest.raises(ValueError, match="no weights"):
        missing_gpt_oss_files(tmp_path)
