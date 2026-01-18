# Plan d'amélioration EXO : Support GPU NVIDIA (Linux)

## 📊 État actuel

### macOS (fonctionnel)
- ✅ **Détection GPU** : `macmon` collecte métriques GPU (usage, temp, power)
- ✅ **Backend MLX Metal** : `mlx>=0.30.1` utilise GPU Apple Silicon automatiquement
- ✅ **Device selection** : `mx.set_default_device(mx.gpu)` + `mx.metal.is_available()`
- ✅ **Télémétrie** : `NodePerformanceProfile.system.gpu_usage` remonté dans `/state`

### Linux (✅ IMPLÉMENTÉ)
- ✅ **Détection GPU** : `nvidia_monitor.py` via pynvml détecte les GPU NVIDIA RTX
- ✅ **MLX CUDA** : Support via `mlx[cuda]>=0.30.1` avec `--extra cuda`
- ✅ **GGUF/llama.cpp CUDA** : `n_gpu_layers` calculé automatiquement selon VRAM
- ✅ **Placement GPU-aware** : VRAM utilisée pour placement si `prefer_gpu=True`
- ✅ **Télémétrie GPU** : VRAM, utilisation, température exposés via `/gpu/info`
- ✅ **API endpoint** : `/gpu/info` retourne l'état GPU de tous les nœuds


---

## 🎯 Objectifs

1. **Détecter les GPU NVIDIA** sur Linux (RTX 3050, etc.)
2. **Utiliser MLX avec CUDA** ou un backend alternatif (PyTorch CUDA)
3. **Exposer VRAM dans la topologie** pour le placement de modèles
4. **Collecter métriques GPU** (usage, temp, VRAM) comme sur macOS
5. **Permettre placement GPU-aware** (modèles sur GPU vs CPU selon disponibilité)

---

## 📋 Plan d'implémentation (par priorité)

### Phase 1 : Détection GPU NVIDIA (Fondation)

#### 1.1 Créer module `exo/worker/utils/nvidia.py`
**Fichier** : `src/exo/worker/utils/nvidia.py`

```python
"""Détection et métriques GPU NVIDIA via NVML/pynvml."""
import platform
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class NvidiaGPUInfo:
    """Informations sur un GPU NVIDIA."""
    device_id: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: int
    temperature: float
    power_usage_watts: float
    utilization_gpu_percent: float
    utilization_memory_percent: float

class NvidiaMonitor:
    """Monitor GPU NVIDIA (équivalent macmon pour macOS)."""
    
    @classmethod
    def is_available(cls) -> bool:
        """Vérifie si NVML est disponible et fonctionnel."""
        if not NVML_AVAILABLE:
            return False
        if platform.system().lower() != "linux":
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except:
            return False
    
    @classmethod
    def get_gpu_count(cls) -> int:
        """Retourne le nombre de GPU NVIDIA détectés."""
        if not cls.is_available():
            return 0
        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount()
        finally:
            pynvml.nvmlShutdown()
    
    @classmethod
    def get_gpu_info(cls, device_id: int = 0) -> Optional[NvidiaGPUInfo]:
        """Récupère les infos d'un GPU spécifique."""
        # Implémentation complète avec pynvml
        pass
    
    @classmethod
    async def get_metrics_async(cls) -> Optional[dict]:
        """Retourne métriques GPU au format similaire à macmon."""
        # Format compatible avec Metrics de macmon.py
        pass
```

**Dépendances à ajouter** :
- `pynvml` (wrapper Python pour NVML)
- Optionnel : `nvidia-ml-py` (alternative)

**Actions** :
- [ ] Créer `src/exo/worker/utils/nvidia.py`
- [ ] Ajouter `pynvml>=11.5.0` dans `pyproject.toml` (dépendance Linux-only)
- [ ] Tests unitaires pour détection GPU

---

#### 1.2 Intégrer détection GPU dans `profile.py`
**Fichier** : `src/exo/worker/utils/profile.py`

**Modifications** :
```python
# Ligne 30-35 : Ajouter fallback NVIDIA
async def get_metrics_async() -> Metrics | None:
    """Return detailed Metrics on macOS or NVIDIA GPU on Linux."""
    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()
    elif platform.system().lower() == "linux":
        # Nouveau : détection NVIDIA
        from .nvidia import NvidiaMonitor
        if NvidiaMonitor.is_available():
            return await NvidiaMonitor.get_metrics_async()
    return None
```

**Actions** :
- [ ] Modifier `get_metrics_async()` pour appeler `NvidiaMonitor`
- [ ] Adapter `start_polling_node_metrics()` pour utiliser métriques NVIDIA
- [ ] Mapper `NvidiaGPUInfo` → `SystemPerformanceProfile` (gpu_usage, temp, etc.)

---

#### 1.3 Exposer VRAM dans `NodePerformanceProfile`
**Fichier** : `src/exo/shared/types/profiling.py`

**Modifications** :
```python
class MemoryPerformanceProfile(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    # NOUVEAU : VRAM GPU
    gpu_vram_total: Memory | None = None
    gpu_vram_available: Memory | None = None
    gpu_vram_used: Memory | None = None

class SystemPerformanceProfile(CamelCaseModel):
    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0
    # NOUVEAU : Infos GPU détaillées
    gpu_count: int = 0
    gpu_names: list[str] = []
    gpu_power_watts: float = 0.0
```

**Actions** :
- [ ] Ajouter champs VRAM dans `MemoryPerformanceProfile`
- [ ] Ajouter `gpu_count`, `gpu_names` dans `SystemPerformanceProfile`
- [ ] Mettre à jour `apply.py` pour initialiser ces champs
- [ ] Mettre à jour dashboard pour afficher VRAM

---

### Phase 2 : Backend MLX CUDA (Utilisation GPU)

#### 2.1 Activer MLX CUDA dans `pyproject.toml`
**Fichier** : `pyproject.toml`

**Modifications** :
```toml
# Ligne 33 : Conditionner selon disponibilité CUDA
"mlx[cpu]>=0.30.1; sys_platform == 'linux' and not cuda_available()",
"mlx[cuda]>=0.30.1; sys_platform == 'linux' and cuda_available()",

# Ligne 56-59 : Décommenter et améliorer
[project.optional-dependencies]
cuda = [
    "mlx[cuda]>=0.30.1",
    "pynvml>=11.5.0",
]

# Ajouter fonction helper pour détecter CUDA
# (peut nécessiter setup.py ou pyproject.toml dynamique)
```

**Problème** : `pyproject.toml` ne supporte pas de logique Python dynamique.

**Solution alternative** :
- Créer `setup_cuda.py` qui détecte CUDA et installe `mlx[cuda]` si disponible
- Ou utiliser variable d'environnement `EXO_USE_CUDA=1` pour forcer CUDA

**Actions** :
- [ ] Créer script `scripts/detect_cuda.py` pour vérifier CUDA
- [ ] Modifier installation pour proposer `uv sync --extra cuda` si CUDA détecté
- [ ] Documenter dans README comment activer CUDA

---

#### 2.2 Détection automatique device MLX (GPU vs CPU)
**Fichier** : `src/exo/worker/engines/mlx/utils_mlx.py`

**Modifications** :
```python
# Ligne 164-174 : initialize_mlx()
def initialize_mlx(bound_instance: BoundInstance) -> Group:
    mx.random.seed(42)
    
    # NOUVEAU : Détecter device disponible
    device_type = _detect_mlx_device()
    if device_type == "gpu":
        mx.set_default_device(mx.gpu)
        logger.info("Using MLX GPU backend")
    elif device_type == "cuda":
        # MLX CUDA si disponible
        mx.set_default_device(mx.cuda)  # ou équivalent
        logger.info("Using MLX CUDA backend")
    else:
        mx.set_default_device(mx.cpu)
        logger.info("Using MLX CPU backend (fallback)")
    
    # ... reste du code

def _detect_mlx_device() -> str:
    """Détecte le meilleur device MLX disponible."""
    if mx.metal.is_available():
        return "gpu"
    # TODO: Vérifier mlx.cuda.is_available() si MLX CUDA existe
    # if hasattr(mx, 'cuda') and mx.cuda.is_available():
    #     return "cuda"
    return "cpu"
```

**Actions** :
- [ ] Ajouter `_detect_mlx_device()` dans `utils_mlx.py`
- [ ] Modifier `initialize_mlx()` pour sélectionner device automatiquement
- [ ] Tester avec MLX CUDA si disponible

---

#### 2.3 Alternative : Backend PyTorch CUDA (si MLX CUDA indisponible)
**Fichier** : `src/exo/worker/engines/pytorch/` (nouveau)

**Si MLX CUDA n'est pas mature**, créer un backend PyTorch alternatif :

```python
# src/exo/worker/engines/pytorch/utils_pytorch.py
import torch

def initialize_pytorch_cuda() -> bool:
    """Initialise PyTorch avec CUDA si disponible."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return True
    return False

def load_model_pytorch(model_path: str):
    """Charge modèle avec PyTorch CUDA."""
    # Implémentation similaire à load_mlx_items()
    pass
```

**Actions** :
- [ ] Évaluer si MLX CUDA est stable (vérifier docs MLX)
- [ ] Si non, créer backend PyTorch optionnel
- [ ] Ajouter `torch>=2.0.0` dans optional-dependencies

---

### Phase 3 : Placement GPU-aware

#### 3.1 Modifier placement pour considérer VRAM
**Fichier** : `src/exo/master/placement_utils.py`

**Modifications** :
```python
def filter_cycles_by_memory(
    cycles: list[list[NodeInfo]],
    required_memory: Memory,
    prefer_gpu: bool = True
) -> list[list[NodeInfo]]:
    """Filtre cycles selon mémoire disponible (RAM ou VRAM si GPU)."""
    valid_cycles = []
    for cycle in cycles:
        total_available = Memory()
        gpu_available = Memory()
        
        for node in cycle:
            if node.node_profile:
                # RAM système
                total_available += node.node_profile.memory.ram_available
                # VRAM GPU (si disponible)
                if node.node_profile.memory.gpu_vram_available:
                    gpu_available += node.node_profile.memory.gpu_vram_available
        
        # Préférer VRAM si disponible et si prefer_gpu=True
        effective_memory = gpu_available if (prefer_gpu and gpu_available.in_bytes > 0) else total_available
        
        if effective_memory >= required_memory:
            valid_cycles.append(cycle)
    
    return valid_cycles
```

**Actions** :
- [ ] Modifier `filter_cycles_by_memory()` pour utiliser VRAM
- [ ] Ajouter paramètre `prefer_gpu` dans `PlaceInstance` command
- [ ] Mettre à jour `place_instance()` pour passer `prefer_gpu=True` si GPU détecté

---

#### 3.2 Exposer préférence GPU dans API
**Fichier** : `src/exo/master/api.py`

**Modifications** :
```python
# Ligne 178-191 : place_instance()
async def place_instance(self, payload: PlaceInstanceParams):
    command = PlaceInstance(
        model_meta=await resolve_model_meta(payload.model_id),
        sharding=payload.sharding,
        instance_meta=payload.instance_meta,
        min_nodes=payload.min_nodes,
        prefer_gpu=payload.prefer_gpu,  # NOUVEAU
    )
    # ...
```

**Actions** :
- [ ] Ajouter `prefer_gpu: bool = True` dans `PlaceInstanceParams`
- [ ] Exposer dans `/instance/previews` les placements GPU vs CPU
- [ ] Dashboard : afficher indicateur "GPU" vs "CPU" pour chaque preview

---

### Phase 4 : Dashboard & Observabilité

#### 4.1 Afficher VRAM dans dashboard
**Fichier** : `dashboard/src/lib/components/` (TopologyGraph ou NodeCard)

**Modifications** :
- Afficher barre VRAM séparée de RAM si `gpu_vram_total` existe
- Afficher nom GPU (`gpu_names[0]`) dans tooltip node
- Indicateur visuel "GPU" vs "CPU" sur chaque node

**Actions** :
- [ ] Modifier composants dashboard pour lire `gpu_vram_*`
- [ ] Ajouter légende "VRAM" dans graphique mémoire
- [ ] Afficher température GPU si disponible

---

#### 4.2 Logs & métriques GPU
**Fichier** : `src/exo/worker/main.py`

**Modifications** :
- Logger au démarrage : "GPU detected: NVIDIA RTX 3050 (8GB VRAM)"
- Logger si fallback CPU : "Warning: GPU detected but MLX CUDA not available, using CPU"

**Actions** :
- [ ] Ajouter logs informatifs sur détection GPU
- [ ] Exposer métriques GPU dans `/state` (déjà fait via `NodePerformanceProfile`)

---

## 🔧 Dépendances à ajouter

### Obligatoires (Linux avec GPU)
```toml
# pyproject.toml
[project.optional-dependencies]
cuda = [
    "mlx[cuda]>=0.30.1",  # Si MLX CUDA stable
    "pynvml>=11.5.0",     # Détection GPU NVIDIA
]
# OU alternative PyTorch
pytorch-cuda = [
    "torch>=2.0.0",
    "pynvml>=11.5.0",
]
```

### Installation
```bash
# Détecter CUDA automatiquement
uv sync --extra cuda

# Ou manuellement
export EXO_USE_CUDA=1
uv sync --extra cuda
```

---

## 📊 Métriques de succès

1. ✅ **Détection** : `curl http://localhost:52415/state | jq '.topology.nodes[].node_profile.system.gpu_count'` → `1` (au lieu de `0`)
2. ✅ **VRAM exposée** : `jq '.topology.nodes[].node_profile.memory.gpu_vram_total'` → `8589934592` (8GB)
3. ✅ **Utilisation GPU** : Logs montrent "Using MLX CUDA backend" au démarrage
4. ✅ **Performance** : Inférence 5-10x plus rapide sur GPU vs CPU (mesurer tokens/sec)

---

## 🚨 Risques & Limitations

1. **MLX CUDA peut être instable** : Vérifier version MLX supporte CUDA avant déploiement
2. **Compatibilité drivers** : Nécessite NVIDIA drivers >= 525.x + CUDA >= 12.0
3. **Fallback CPU** : Si CUDA indisponible, exo doit continuer sur CPU sans crash
4. **Multi-GPU** : Plan futur pour utiliser plusieurs GPU sur un même node

---

## 📅 Estimation

- **Phase 1** (Détection) : 2-3 jours
- **Phase 2** (Backend CUDA) : 3-5 jours (selon stabilité MLX CUDA)
- **Phase 3** (Placement) : 1-2 jours
- **Phase 4** (Dashboard) : 1-2 jours

**Total** : ~1-2 semaines de dev + tests

---

## 🔗 Références

- [MLX CUDA Support](https://github.com/ml-explore/mlx) (vérifier si disponible)
- [pynvml Documentation](https://pypi.org/project/nvidia-ml-py/)
- [NVIDIA Management Library (NVML)](https://developer.nvidia.com/nvidia-management-library-nvml)
