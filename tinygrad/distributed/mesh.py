"""
DeviceMesh for logical topology abstraction in tinygrad.

Provides a logical mesh abstraction over physical devices for distributed training.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Union
import itertools

from tinygrad.uop.ops import Ops
from .bootstrap import DistributedConfig, init_distributed, get_all_capabilities, HAS_MPI

class DeviceMesh:
    """
    Logical mesh of devices for distributed training.

    A DeviceMesh represents a logical topology of devices (GPUs) across
    potentially multiple nodes. It provides methods for sharding tensors
    and coordinating collective operations.

    Example:
        >>> config = init_distributed()
        >>> mesh = DeviceMesh(config=config)
        >>> # Shard a tensor along the first axis (data parallelism)
        >>> x = Tensor.randn(128, 256).shard_(mesh, axis=0)
    """

    def __init__(self,
                 shape: Optional[Tuple[int, ...]] = None,
                 devices: Optional[List[str]] = None,
                 config: Optional[DistributedConfig] = None):
        """
        Initialize a DeviceMesh.

        Args:
            shape: Logical mesh shape (e.g., (2, 4) for 2 nodes × 4 GPUs).
                   If None, auto-detected from distributed config.
            devices: List of device strings (e.g., ["AMD:0", "AMD:1", ...]).
                     If None, auto-discovered from system or built from config.
            config: DistributedConfig from bootstrap. If None, will try to
                    initialize distributed or create single-node mesh.
        """
        self.config = config
        self._global_config = init_distributed() if config is None else config

        if devices is not None:
            self.devices = devices
            self.shape = shape or (len(devices),)
        else:
            self._discover_devices()
            if shape is None:
                if self._global_config and self._global_config.num_nodes > 1:
                    self.shape = (self._global_config.num_nodes, self._global_config.gpus_per_node)
                else:
                    self.shape = (len(self.devices),)
            else:
                self.shape = shape

            total_devices = 1
            for dim in self.shape:
                total_devices *= dim

            if total_devices != len(self.devices):
                self._build_devices_from_topology()
                total_devices = 1
                for dim in self.shape:
                    total_devices *= dim
                if total_devices != len(self.devices):
                    raise ValueError(
                        f"Shape {self.shape} requires {total_devices} devices, "
                        f"but found {len(self.devices)} devices"
                    )

        self._coord_to_device: Dict[Tuple[int, ...], str] = {}
        self._device_to_coord: Dict[str, Tuple[int, ...]] = {}
        self._build_coordinate_mapping()
        self._submeshes: Dict[Tuple[int, ...], 'DeviceMesh'] = {}
        self._peer_groups: Dict[str, List[str]] = {}
        self._setup_peer_groups()

    def _discover_devices(self) -> None:
        """Discover available devices for this mesh."""
        if self._global_config is None or self._global_config.world_size <= 1:
            devices = []

            if os.getenv("CUDA_VISIBLE_DEVICES"):
                cuda_count = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
                devices.extend([f"NV:{i}" for i in range(cuda_count)])

            if os.getenv("ROCR_VISIBLE_DEVICES"):
                amd_count = len(os.getenv("ROCR_VISIBLE_DEVICES").split(","))
                devices.extend([f"AMD:{i}" for i in range(amd_count)])

            if not devices:
                devices = ["CPU:0"]

            self.devices = devices
        else:
            all_caps = get_all_capabilities()
            if all_caps and len(all_caps) > 0:
                self.devices = all_caps[0].get("gpus", [f"AMD:{i}" for i in range(self._global_config.gpus_per_node)])
            else:
                local_rank = self._global_config.local_rank
                self.devices = [f"AMD:{local_rank}"]

    def _build_devices_from_topology(self) -> None:
        """Build device list from exchanged capabilities."""
        if self._global_config is None or self._global_config.world_size <= 1:
            return

        all_caps = get_all_capabilities()
        if not all_caps:
            return

        self.devices = []
        for caps in all_caps:
            self.devices.extend(caps.get("gpus", []))

    def _build_coordinate_mapping(self) -> None:
        """Build mapping between mesh coordinates and devices."""
        ranges = [range(dim) for dim in self.shape]
        coordinates = list(itertools.product(*ranges))

        if len(coordinates) != len(self.devices):
            return

        for coord, device in zip(coordinates, self.devices):
            self._coord_to_device[coord] = device
            self._device_to_coord[device] = coord

    def _setup_peer_groups(self) -> None:
        """Set up peer groups on HCQCompiled devices."""
        try:
            from tinygrad.runtime.support.hcq import HCQCompiled
        except ImportError:
            return

        all_caps = get_all_capabilities() if self._global_config and self._global_config.world_size > 1 else []

        for caps in all_caps:
            node_name = caps.get("node", "")
            for gpu in caps.get("gpus", []):
                try:
                    dev = None
                    try:
                        from tinygrad.device import Device
                        dev = Device[gpu]
                    except (KeyError, IndexError):
                        continue

                    if hasattr(dev, 'peer_group'):
                        dev.peer_group = node_name
                        if node_name not in self._peer_groups:
                            self._peer_groups[node_name] = []
                        if gpu not in self._peer_groups[node_name]:
                            self._peer_groups[node_name].append(gpu)
                except Exception:
                    continue

        if not self._peer_groups:
            self._peer_groups["localhost"] = self.devices

    def get_device(self, *coords: int) -> str:
        """
        Get device at given mesh coordinates.

        Args:
            *coords: Mesh coordinates (e.g., (0, 1) for row 0, column 1)

        Returns:
            Device string.

        Raises:
            KeyError: If coordinates are out of bounds.
        """
        if len(coords) != len(self.shape):
            raise ValueError(
                f"Expected {len(self.shape)} coordinates, got {len(coords)}"
            )

        coord = tuple(coords)
        if coord not in self._coord_to_device:
            raise KeyError(f"Coordinates {coord} out of bounds for shape {self.shape}")

        return self._coord_to_device[coord]

    def get_coordinates(self, device: str) -> Tuple[int, ...]:
        """
        Get mesh coordinates for a device.

        Args:
            device: Device string.

        Returns:
            Mesh coordinates.

        Raises:
            KeyError: If device not in mesh.
        """
        if device not in self._device_to_coord:
            raise KeyError(f"Device {device} not in mesh")

        return self._device_to_coord[device]

    def submesh(self, dim: int, index: int) -> 'DeviceMesh':
        """
        Get a submesh along a dimension.

        Useful for creating sub-groups for collective operations.

        Args:
            dim: Dimension to slice along.
            index: Index in that dimension.

        Returns:
            DeviceMesh representing the submesh.

        Example:
            >>> mesh = DeviceMesh(shape=(2, 4))  # 2 nodes × 4 GPUs
            >>> node0 = mesh.submesh(dim=0, index=0)  # All GPUs on node 0
            >>> node1 = mesh.submesh(dim=0, index=1)  # All GPUs on node 1
        """
        key = (dim, index)
        if key in self._submeshes:
            return self._submeshes[key]

        sub_devices = []
        for device in self.devices:
            coords = self.get_coordinates(device)
            if coords[dim] == index:
                sub_devices.append(device)

        new_shape = tuple(s for i, s in enumerate(self.shape) if i != dim)

        submesh = DeviceMesh(shape=new_shape if new_shape else (len(sub_devices),), devices=sub_devices, config=self.config)
        self._submeshes[key] = submesh
        return submesh

    def all_submeshes(self, dim: int) -> List['DeviceMesh']:
        """
        Get all submeshes along a dimension.

        Args:
            dim: Dimension to slice along.

        Returns:
            List of DeviceMeshes, one for each index in the dimension.
        """
        return [self.submesh(dim, i) for i in range(self.shape[dim])]

    def peer_group(self, device: str) -> str:
        """
        Get peer group for a device.

        In tinygrad's HCQ system, devices in the same peer group use P2P
        (PCIe/NVLink/Infinity Fabric), while devices in different peer
        groups use RDMA or TCP.

        By default, devices on the same node are in the same peer group.

        Args:
            device: Device string.

        Returns:
            Peer group identifier (typically node name).
        """
        if self._global_config is None:
            return "localhost"

        if len(self.shape) >= 2:
            coords = self.get_coordinates(device)
            node_index = coords[0]
            return f"node_{node_index}"
        else:
            if self._global_config.num_nodes > 1:
                return f"node_{self._global_config.node_rank}"
            else:
                return "localhost"

    def is_same_peer_group(self, device1: str, device2: str) -> bool:
        """
        Check if two devices are in the same peer group.

        Args:
            device1: First device string.
            device2: Second device string.

        Returns:
            True if devices are in same peer group (can use P2P).
        """
        return self.peer_group(device1) == self.peer_group(device2)

    def get_node_devices(self, node_rank: Optional[int] = None) -> List[str]:
        """Get all devices on a specific node."""
        if node_rank is None:
            node_rank = self._global_config.node_rank if self._global_config else 0

        if len(self.shape) >= 2:
            return [self.get_device(node_rank, i) for i in range(self.shape[1])]
        else:
            return self.devices

    def allreduce(self, data: Any, op: str = "sum") -> Any:
        """
        Allreduce collective operation across devices in the mesh.
        
        Args:
            data: Input tensor (should be sharded or multi-device).
            op: Reduction operation ("sum", "max", "min", "prod").
        
        Returns:
            Tensor with allreduce applied across all devices.
        """
        if self._global_config is None or self._global_config.world_size <= 1:
            return data
        
        op_map = {
            "sum": Ops.ADD,
            "max": Ops.MAX,
            "min": None,
            "prod": None
        }
        if op not in op_map or op_map[op] is None:
            raise NotImplementedError(f"Allreduce op {op} not supported")
        
        if hasattr(data, "allreduce"):
            return data.allreduce(op_map[op], tuple(self.devices))
        return data

    def allgather(self, data: Any) -> Any:
        """
        Allgather collective operation across devices in the mesh.
        
        Args:
            data: Input tensor (sharded across devices).
        
        Returns:
            Tensor with all shards gathered.
        """
        if self._global_config is None or self._global_config.world_size <= 1:
            return data
        
        if hasattr(data, "mselect") and hasattr(data, "_unshard"):
            # Use multi op to unshard (allgather)
            return data._unshard(axis=self._global_config.local_rank)
        return data

    def barrier(self) -> None:
        """
        Barrier synchronization across all devices/ranks in the mesh.
        
        Wait for all ranks to reach this point before proceeding.
        """
        if self._global_config is None or self._global_config.world_size <= 1:
            return
        
        if HAS_MPI and self._global_config.backend == "mpi":
            from .bootstrap import MPI
            MPI.COMM_WORLD.barrier()
        elif self._global_config.backend == "slurm":
            # Simple debug print for Slurm (no real barrier yet)
            print(f"[Slurm barrier] Rank {self._global_config.rank}")
        else:
            # Fallback: nothing to do (single-node/local)
            pass

    def __repr__(self) -> str:
        return (f"DeviceMesh(shape={self.shape}, devices={self.devices}, "
                f"config={self._global_config})")

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self):
        return iter(self.devices)

    def __getitem__(self, idx: int) -> str:
        return self.devices[idx]

import os

def create_data_parallel_mesh() -> DeviceMesh:
    """
    Create a DeviceMesh for data parallelism.

    All devices are arranged in a 1D mesh for data parallel sharding.

    Returns:
        DeviceMesh for data parallelism.
    """
    config = init_distributed()
    return DeviceMesh(config=config)

def create_tensor_parallel_mesh() -> DeviceMesh:
    """
    Create a DeviceMesh for tensor parallelism.

    Devices are arranged to optimize for tensor parallel operations.
    For multi-node, tries to keep tensor parallel groups within nodes
    to minimize cross-node communication.

    Returns:
        DeviceMesh for tensor parallelism.
    """
    config = init_distributed()

    if config is None:
        return DeviceMesh()

    if config.num_nodes > 1:
        shape = (config.num_nodes, config.gpus_per_node)
    else:
        shape = (config.gpus_per_node,)

    return DeviceMesh(shape=shape, config=config)

def create_pipeline_parallel_mesh(stages: int) -> DeviceMesh:
    """
    Create a DeviceMesh for pipeline parallelism.

    Devices are arranged for pipeline parallel stages.

    Args:
        stages: Number of pipeline stages.

    Returns:
        DeviceMesh for pipeline parallelism.

    Raises:
        ValueError: If number of devices not divisible by stages.
    """
    config = init_distributed()

    if config is None:
        mesh = DeviceMesh()
        devices = mesh.devices
    else:
        all_caps = get_all_capabilities()
        devices = []
        for caps in all_caps:
            devices.extend(caps.get("gpus", []))

    if len(devices) % stages != 0:
        raise ValueError(
            f"Number of devices ({len(devices)}) must be divisible by "
            f"pipeline stages ({stages})"
        )

    devices_per_stage = len(devices) // stages
    shape = (stages, devices_per_stage)

    return DeviceMesh(shape=shape, devices=devices, config=config)

__all__ = [
    "DeviceMesh",
    "create_data_parallel_mesh",
    "create_tensor_parallel_mesh",
    "create_pipeline_parallel_mesh",
]