"""Discover and manage Ollama models installed locally.

This module scans the Ollama model directory to find installed models
and extracts their metadata for use in EXO's distributed inference system.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class OllamaModel:
    """Represents an Ollama model installed locally."""
    
    name: str
    tag: str
    size_bytes: int
    model_path: Path
    digest: str
    architecture: Optional[str] = None
    quantization: Optional[str] = None
    n_layers: Optional[int] = None
    
    @property
    def full_name(self) -> str:
        """Returns the full model name with tag (e.g., 'qwen3-coder:30b')."""
        return f"{self.name}:{self.tag}"
    
    @property
    def size_gb(self) -> float:
        """Returns the model size in gigabytes."""
        return self.size_bytes / (1024 ** 3)


def get_ollama_home() -> Path:
    """Get the Ollama home directory.
    
    Returns:
        Path to Ollama's home directory (default: ~/.ollama)
    """
    return Path(os.environ.get("OLLAMA_HOME", Path.home() / ".ollama"))


def get_ollama_models_dir() -> Path:
    """Get the Ollama models directory.
    
    Returns:
        Path to Ollama's models directory
    """
    return get_ollama_home() / "models"


def get_ollama_model_path(model_name: str, tag: str = "latest") -> Optional[Path]:
    """Get the path to a specific Ollama model's weights file.
    
    Args:
        model_name: Name of the model (e.g., 'qwen3-coder')
        tag: Model tag (e.g., '30b', 'latest')
    
    Returns:
        Path to the model's GGUF file, or None if not found
    """
    models_dir = get_ollama_models_dir()
    manifests_dir = models_dir / "manifests" / "registry.ollama.ai" / "library"
    
    manifest_path = manifests_dir / model_name / tag
    if not manifest_path.exists():
        logger.warning(f"Manifest not found for {model_name}:{tag}")
        return None
    
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Find the model layer (largest blob, typically the weights)
        blobs_dir = models_dir / "blobs"
        for layer in manifest.get("layers", []):
            media_type = layer.get("mediaType", "")
            if "model" in media_type:
                digest = layer.get("digest", "").replace(":", "-")
                blob_path = blobs_dir / digest
                if blob_path.exists():
                    return blob_path
        
        logger.warning(f"Model weights not found in manifest for {model_name}:{tag}")
        return None
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing manifest for {model_name}:{tag}: {e}")
        return None


def discover_ollama_models() -> list[OllamaModel]:
    """Discover all Ollama models installed locally.
    
    Scans the Ollama models directory and returns information about
    all installed models.
    
    Returns:
        List of OllamaModel objects representing installed models
    """
    models: list[OllamaModel] = []
    models_dir = get_ollama_models_dir()
    
    if not models_dir.exists():
        logger.info("Ollama models directory not found")
        return models
    
    manifests_dir = models_dir / "manifests" / "registry.ollama.ai" / "library"
    
    if not manifests_dir.exists():
        logger.info("Ollama manifests directory not found")
        return models
    
    blobs_dir = models_dir / "blobs"
    
    # Iterate through all model directories
    for model_dir in manifests_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Iterate through all tags for this model
        for tag_file in model_dir.iterdir():
            if tag_file.is_dir():
                continue
            
            tag = tag_file.name
            
            try:
                with open(tag_file, "r") as f:
                    manifest = json.load(f)
                
                # Find model weights blob
                model_digest = None
                model_size = 0
                
                for layer in manifest.get("layers", []):
                    media_type = layer.get("mediaType", "")
                    if "model" in media_type:
                        model_digest = layer.get("digest", "").replace(":", "-")
                        model_size = layer.get("size", 0)
                        break
                
                if model_digest:
                    blob_path = blobs_dir / model_digest
                    if blob_path.exists():
                        # Get actual file size if not in manifest
                        if model_size == 0:
                            model_size = blob_path.stat().st_size
                        
                        model = OllamaModel(
                            name=model_name,
                            tag=tag,
                            size_bytes=model_size,
                            model_path=blob_path,
                            digest=model_digest,
                        )
                        models.append(model)
                        logger.info(f"Discovered Ollama model: {model.full_name} ({model.size_gb:.1f} GB)")
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error parsing manifest for {model_name}:{tag}: {e}")
                continue
    
    return models


async def discover_ollama_models_async() -> list[OllamaModel]:
    """Async version of discover_ollama_models.
    
    Note: Currently just wraps the sync version, but could be made
    truly async with aiofiles if needed.
    """
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, discover_ollama_models)
