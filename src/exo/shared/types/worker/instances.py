from enum import Enum

from pydantic import model_validator

from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    GGUFPipeline = "GGUFPipeline"  # For Ollama/GGUF models


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    ibv_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


class GGUFPipelineInstance(BaseInstance):
    """Instance for distributed GGUF/Ollama model inference.
    
    Uses pipeline parallelism where each node handles a subset of layers
    and activations are passed between nodes via TCP.
    """
    gguf_model_path: str  # Path to GGUF file (on each node)
    pipeline_ports: dict[NodeId, int]  # Port each node listens on for activations
    node_order: list[NodeId]  # Order of nodes in the pipeline


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | GGUFPipelineInstance


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
