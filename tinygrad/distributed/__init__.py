"""
Distributed training module for tinygrad.

Provides DeepSpeed-style distributed training capabilities with multi-node support.
"""

from .bootstrap import init_distributed, DistributedConfig, get_world_info, get_all_capabilities, reset_global_config
from .mesh import DeviceMesh, create_data_parallel_mesh, create_tensor_parallel_mesh, create_pipeline_parallel_mesh
from .rdma_config import RDMAConfig, patch_rdma_ip_assignment, RDMANetwork
from tinygrad.checkpoint import checkpoint, checkpoint_sequential, CheckpointFunction, CheckpointBuilder, create_checkpoint, is_checkpointed

# Try to patch RDMA IP assignment automatically
try:
    patch_rdma_ip_assignment()
except Exception as e:
    # RDMA module might not be available
    pass

__all__ = [
    # Bootstrap
    "init_distributed",
    "DistributedConfig",
    "get_world_info",
    "get_all_capabilities",
    "reset_global_config",

    # DeviceMesh
    "DeviceMesh",
    "create_data_parallel_mesh",
    "create_tensor_parallel_mesh",
    "create_pipeline_parallel_mesh",

    # RDMA
    "RDMAConfig",
    "patch_rdma_ip_assignment",
    "RDMANetwork",

    # Checkpointing
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointFunction",
    "CheckpointBuilder",
    "create_checkpoint",
    "is_checkpointed",
]