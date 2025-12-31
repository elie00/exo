"""Integration of GGUF/Ollama models with EXO's model system.

This module bridges the Ollama local models with EXO's model cards
and instance management system, allowing Ollama models to be used
in the distributed inference cluster.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.worker.engines.gguf.gguf_loader import GGUFModelInfo, load_gguf_metadata
from exo.worker.engines.gguf.ollama_discovery import OllamaModel, discover_ollama_models


def ollama_model_to_metadata(
    ollama_model: OllamaModel,
    gguf_info: Optional[GGUFModelInfo] = None,
) -> ModelMetadata:
    """Convert an Ollama model to EXO ModelMetadata.
    
    Args:
        ollama_model: The discovered Ollama model
        gguf_info: Optional pre-loaded GGUF metadata
    
    Returns:
        ModelMetadata compatible with EXO's model system
    """
    # Load GGUF info if not provided
    if gguf_info is None:
        try:
            gguf_info = load_gguf_metadata(ollama_model.model_path, load_tensors=False)
        except Exception as e:
            logger.warning(f"Failed to load GGUF metadata for {ollama_model.full_name}: {e}")
            gguf_info = None
    
    # Create model ID in EXO format
    model_id = ModelId(f"ollama/{ollama_model.full_name}")
    
    # Extract metadata from GGUF or use defaults
    n_layers = gguf_info.n_layers if gguf_info else 32
    hidden_size = gguf_info.hidden_size if gguf_info else 4096
    
    return ModelMetadata(
        model_id=model_id,
        pretty_name=f"{ollama_model.name.title()} {ollama_model.tag} (Ollama)",
        storage_size=Memory.from_bytes(ollama_model.size_bytes),
        n_layers=n_layers,
        hidden_size=hidden_size,
        supports_tensor=False,  # GGUF models use pipeline parallelism
    )


async def discover_and_register_ollama_models() -> list[ModelMetadata]:
    """Discover all Ollama models and create EXO-compatible metadata.
    
    Returns:
        List of ModelMetadata for all discovered Ollama models
    """
    from exo.worker.engines.gguf.ollama_discovery import discover_ollama_models_async
    
    ollama_models = await discover_ollama_models_async()
    
    metadata_list: list[ModelMetadata] = []
    
    for ollama_model in ollama_models:
        try:
            # Load GGUF metadata for each model
            gguf_info = load_gguf_metadata(ollama_model.model_path, load_tensors=False)
            
            metadata = ollama_model_to_metadata(ollama_model, gguf_info)
            metadata_list.append(metadata)
            
            logger.info(
                f"Registered Ollama model: {metadata.pretty_name} "
                f"({metadata.n_layers} layers, {metadata.storage_size})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to register Ollama model {ollama_model.full_name}: {e}")
            continue
    
    return metadata_list


def get_ollama_model_cards() -> dict[str, "OllamaModelCard"]:
    """Get model cards for all available Ollama models.
    
    This function synchronously discovers Ollama models and creates
    model cards compatible with EXO's MODEL_CARDS format.
    
    Returns:
        Dictionary of model short_id -> OllamaModelCard
    """
    from dataclasses import dataclass
    
    @dataclass
    class OllamaModelCard:
        short_id: str
        model_id: ModelId
        name: str
        description: str
        tags: list[str]
        metadata: ModelMetadata
        ollama_model: OllamaModel
        gguf_path: Path
    
    ollama_models = discover_ollama_models()
    cards: dict[str, OllamaModelCard] = {}
    
    for ollama_model in ollama_models:
        try:
            gguf_info = load_gguf_metadata(ollama_model.model_path, load_tensors=False)
            metadata = ollama_model_to_metadata(ollama_model, gguf_info)
            
            short_id = f"ollama-{ollama_model.name}-{ollama_model.tag}".lower().replace(":", "-")
            
            card = OllamaModelCard(
                short_id=short_id,
                model_id=metadata.model_id,
                name=metadata.pretty_name,
                description=f"Ollama model: {ollama_model.full_name} ({gguf_info.architecture if gguf_info else 'unknown'} architecture)",
                tags=["ollama", "gguf", "local"],
                metadata=metadata,
                ollama_model=ollama_model,
                gguf_path=ollama_model.model_path,
            )
            
            cards[short_id] = card
            
        except Exception as e:
            logger.warning(f"Failed to create card for {ollama_model.full_name}: {e}")
            continue
    
    return cards


class OllamaModelProvider:
    """Provider for Ollama models in EXO.
    
    This class manages the discovery, registration, and lifecycle
    of Ollama models within the EXO distributed inference system.
    """
    
    def __init__(self):
        self._models: dict[str, OllamaModel] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        self._gguf_info: dict[str, GGUFModelInfo] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the provider and discover available models."""
        if self._initialized:
            return
        
        from exo.worker.engines.gguf.ollama_discovery import discover_ollama_models_async
        
        ollama_models = await discover_ollama_models_async()
        
        for model in ollama_models:
            model_key = model.full_name
            self._models[model_key] = model
            
            try:
                gguf_info = load_gguf_metadata(model.model_path, load_tensors=False)
                self._gguf_info[model_key] = gguf_info
                self._metadata[model_key] = ollama_model_to_metadata(model, gguf_info)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_key}: {e}")
        
        self._initialized = True
        logger.info(f"OllamaModelProvider initialized with {len(self._models)} models")
    
    def list_models(self) -> list[str]:
        """List all available Ollama model names."""
        return list(self._models.keys())
    
    def get_model(self, name: str) -> Optional[OllamaModel]:
        """Get an Ollama model by name."""
        return self._models.get(name)
    
    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get EXO metadata for an Ollama model."""
        return self._metadata.get(name)
    
    def get_gguf_info(self, name: str) -> Optional[GGUFModelInfo]:
        """Get GGUF info for an Ollama model."""
        return self._gguf_info.get(name)
    
    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the path to an Ollama model's GGUF file."""
        model = self._models.get(name)
        return model.model_path if model else None


# Global provider instance
_ollama_provider: Optional[OllamaModelProvider] = None


async def get_ollama_provider() -> OllamaModelProvider:
    """Get the global Ollama model provider (singleton).
    
    Returns:
        Initialized OllamaModelProvider instance
    """
    global _ollama_provider
    
    if _ollama_provider is None:
        _ollama_provider = OllamaModelProvider()
        await _ollama_provider.initialize()
    
    return _ollama_provider
