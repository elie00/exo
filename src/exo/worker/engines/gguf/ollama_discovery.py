"""Discover and manage Ollama models installed locally.

This module scans the Ollama model directory to find installed models
and extracts their metadata for use in EXO's distributed inference system.
"""

import json
import os
import platform
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
    
    Searches multiple locations based on OS and environment variables.
    
    Returns:
        Path to Ollama's home directory
    """
    # Check environment variables first
    if "OLLAMA_HOME" in os.environ:
        return Path(os.environ["OLLAMA_HOME"])
    
    if "OLLAMA_MODELS" in os.environ:
        # OLLAMA_MODELS points to the models dir, parent is home
        return Path(os.environ["OLLAMA_MODELS"]).parent
    
    # OS-specific defaults
    if platform.system() == "Darwin":
        return Path.home() / ".ollama"
    elif platform.system() == "Linux":
        # Try user directory first
        user_path = Path.home() / ".ollama"
        if user_path.exists():
            return user_path
        # System installation (via official install script)
        system_path = Path("/usr/share/ollama/.ollama")
        if system_path.exists():
            return system_path
        # Fallback to user path (will be created if models are pulled)
        return user_path
    else:
        # Windows or other
        return Path.home() / ".ollama"


def get_ollama_models_dir() -> Path:
    """Get the Ollama models directory.
    
    Returns:
        Path to Ollama's models directory
    """
    # Check if OLLAMA_MODELS is explicitly set
    if "OLLAMA_MODELS" in os.environ:
        return Path(os.environ["OLLAMA_MODELS"])
    
    return get_ollama_home() / "models"


def _find_models_dir() -> Optional[Path]:
    """Find the Ollama models directory by searching multiple locations.
    
    Returns:
        Path to models directory if found, None otherwise
    """
    # Check environment variable first
    if "OLLAMA_MODELS" in os.environ:
        path = Path(os.environ["OLLAMA_MODELS"])
        if path.exists():
            return path
    
    # List of possible locations to check
    candidates = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/models"),
        Path("/opt/ollama/models"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Found Ollama models at: {candidate}")
            return candidate
    
    return None


def get_ollama_model_path(model_name: str, tag: str = "latest") -> Optional[Path]:
    """Get the path to a specific Ollama model's weights file.
    
    Args:
        model_name: Name of the model (e.g., 'qwen3-coder')
        tag: Model tag (e.g., '30b', 'latest')
    
    Returns:
        Path to the model's GGUF file, or None if not found
    """
    models_dir = _find_models_dir()
    if models_dir is None:
        logger.warning("No Ollama models directory found")
        return None
    
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
    
    # Find models directory
    models_dir = _find_models_dir()
    if models_dir is None:
        logger.info("Ollama models directory not found in any standard location")
        logger.info("Searched: ~/.ollama/models, /usr/share/ollama/.ollama/models")
        return models
    
    logger.debug(f"Using Ollama models directory: {models_dir}")
    
    manifests_dir = models_dir / "manifests" / "registry.ollama.ai" / "library"
    
    if not manifests_dir.exists():
        logger.info(f"Ollama manifests directory not found at {manifests_dir}")
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
    
    
    # Scan for local GGUF files in multiple directories
    try:
        # Build list of directories to scan for local GGUF models
        local_model_dirs: list[Path] = [
            Path.home() / "exo" / "models",  # Default user dir
        ]
        
        # Add /mnt/models on Linux (common for attached storage)
        if platform.system() == "Linux":
            mnt_models = Path("/mnt/models")
            if mnt_models.exists():
                local_model_dirs.append(mnt_models)
        
        # Add custom paths from environment variable (colon-separated)
        if "EXO_GGUF_MODELS_PATH" in os.environ:
            for path in os.environ["EXO_GGUF_MODELS_PATH"].split(":"):
                custom_path = Path(path.strip())
                if custom_path.exists() and custom_path not in local_model_dirs:
                    local_model_dirs.append(custom_path)
        
        # Scan all directories for GGUF files
        for local_models_dir in local_model_dirs:
            if not local_models_dir.exists():
                continue
                
            logger.debug(f"Scanning for local GGUF models in: {local_models_dir}")
            
            for file_path in local_models_dir.glob("*.gguf"):
                if not file_path.is_file():
                    continue
                    
                model_name = file_path.stem
                size = file_path.stat().st_size
                
                # Create a synthetic OllamaModel
                model = OllamaModel(
                    name=model_name.lower().replace("_", "-"),
                    tag="local",
                    size_bytes=size,
                    model_path=file_path,
                    digest=f"local-{model_name}",
                )
                models.append(model)
                logger.info(f"Discovered Local GGUF model: {model.full_name} ({model.size_gb:.1f} GB)")
                
    except Exception as e:
        logger.error(f"Error scanning local GGUF models: {e}")

    return models


async def discover_ollama_models_async() -> list[OllamaModel]:
    """Async version of discover_ollama_models.
    
    Note: Currently just wraps the sync version, but could be made
    truly async with aiofiles if needed.
    """
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, discover_ollama_models)
