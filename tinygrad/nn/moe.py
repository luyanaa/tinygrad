"""
Mixture of Experts (MoE) for tinygrad.

Provides distributed MoE layer with expert parallelism across multiple GPUs/nodes.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Callable
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv

class MoELayer:
    """
    Distributed Mixture of Experts layer.

    Routes tokens to different expert subnetworks, with experts placed on different devices.
    Uses all-to-all communication for token routing across devices.

    For multi-node MoE, tokens are sent to remote experts via RDMA/TCP.
    """

    def __init__(self, num_experts: int, dim: int, hidden_dim: int,
                 activated_experts: int = 1, bias: bool = False,
                 devices: Optional[List[str]] = None,
                 gate_bias: bool = False):
        """
        Args:
            num_experts: Total number of experts.
            dim: Input dimension.
            hidden_dim: Hidden dimension.
            activated_experts: Number of experts to activate per token (top-k).
            bias: Whether to use bias in linear layers.
            devices: Devices to place experts on. If None, all experts on same device.
            gate_bias: Whether to use bias in gate.
        """
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activated_experts = activated_experts
        self.devices = devices

        self.gate = Tensor.uniform(num_experts, dim, low=-0.1, high=0.1)

        self.experts_per_device = 1
        if devices is not None:
            self.experts_per_device = (num_experts + len(devices) - 1) // len(devices)

        self.up_proj = Tensor.uniform(num_experts, hidden_dim, dim, low=-0.1, high=0.1)
        self.gate_proj = Tensor.uniform(num_experts, hidden_dim, dim, low=-0.1, high=0.1)
        self.down_proj = Tensor.uniform(num_experts, dim, hidden_dim, low=-0.1, high=0.1)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass with optional distributed expert routing.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        if self.devices is None or len(self.devices) == 1:
            return self._local_forward(x)

        return self._distributed_forward(x)

    def _local_forward(self, x: Tensor) -> Tensor:
        """Forward pass with all experts on same device."""
        bsz, seqlen, dim = x.shape

        gate_out = x @ self.gate.transpose()
        gate_out = gate_out.reshape(bsz * seqlen, self.num_experts).softmax(-1)

        topk_probs, topk_indices = gate_out.topk(self.activated_experts, dim=-1)

        x_flat = x.reshape(bsz * seqlen, dim)
        out = Tensor.zeros(bsz * seqlen, dim)
        for k in range(self.activated_experts):
            expert_idx = topk_indices[:, k]
            expert_weight = topk_probs[:, k].reshape(-1, 1)

            for e in range(self.num_experts):
                mask = (expert_idx == e).cast(dtypes.float).reshape(-1, 1)

                expert_input = mask * x_flat

                up = (expert_input @ self.up_proj[e].permute(1, 0))
                gate = (expert_input @ self.gate_proj[e].permute(1, 0))
                hidden = (up * gate.silu())

                down = (hidden @ self.down_proj[e].permute(1, 0))

                out += mask * (down * expert_weight)

        return out.reshape(bsz, seqlen, -1)

    def _distributed_forward(self, x: Tensor) -> Tensor:
        """Forward pass with experts distributed across devices."""
        raise NotImplementedError("Distributed MoE requires all-to-all primitive")

class GroupedMoELayer(MoELayer):
    """
    Grouped Mixture of Experts with shared experts.

    Some experts are shared across all tokens, while others are routed.
    """

    def __init__(self, num_experts: int, num_shared_experts: int, dim: int, hidden_dim: int,
                 activated_experts: int = 1, bias: bool = False,
                 devices: Optional[List[str]] = None):
        super().__init__(num_experts, dim, hidden_dim, activated_experts, bias, devices)

        self.num_shared_experts = num_shared_experts
        self.shared_up = Tensor.uniform(num_shared_experts, hidden_dim, dim, low=-0.1, high=0.1)
        self.shared_down = Tensor.uniform(num_shared_experts, dim, hidden_dim, low=-0.1, high=0.1)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass with shared experts computed for all tokens."""
        bsz, seqlen, dim = x.shape

        shared_out = Tensor.zeros(bsz, seqlen, dim)
        for e in range(self.num_shared_experts):
            up = x @ self.shared_up[e].permute(1, 0)
            hidden = up.silu()
            shared_out += x @ self.hidden @ self.shared_down[e].permute(1, 0)

        routed_out = self._local_forward(x)

        return shared_out + routed_out

class AllToAll:
    """
    All-to-all communication primitive for distributed MoE.

    Each device sends different data to every other device.
    In MoE context, this is used for token routing: each token is sent to
    the device holding the selected expert.
    """

    def __init__(self, devices: List[str]):
        """
        Args:
            devices: List of devices participating in all-to-all.
        """
        self.devices = devices
        self.n_devices = len(devices)

    def forward(self, data: Tensor, expert_mapping: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Send data to appropriate experts via all-to-all.

        Args:
            data: Input tensor of shape (total_tokens, dim).
            expert_mapping: Tensor of shape (total_tokens,) with expert indices.

        Returns:
            Tuple of (tokens_per_expert, data_per_expert) where each is a list
            of tensors, one per expert.
        """
        raise NotImplementedError("All-to-all requires RDMA/TCP integration")

    def backward(self, grad_output: List[Tensor]) -> Tensor:
        """
        Gather gradients back via reverse all-to-all.

        Args:
            grad_output: List of gradients, one per expert.

        Returns:
            Combined gradient tensor.
        """
        raise NotImplementedError("All-to-all backward requires RDMA/TCP integration")

def all_to_all(tokens: Tensor, expert_ids: Tensor, num_experts: int,
               devices: List[str]) -> Tuple[List[Tensor], List[int]]:
    """
    All-to-all token routing for distributed MoE.

    Routes each token to the device containing its selected expert.

    Args:
        tokens: Input tokens of shape (batch, seq_len, dim).
        expert_ids: Expert indices for each token, shape (batch, seq_len, activated_experts).
        num_experts: Total number of experts.
        devices: List of devices.

    Returns:
        Tuple of (tokens_list, expert_counts) where tokens_list is a list of tensors
        (one per target device) and expert_counts indicates how many tokens go to each device.
    """
    raise NotImplementedError("all_to_all requires multi-device support")

def all_to_all_reverse(grads: List[Tensor], expert_counts: List[int],
                       devices: List[str]) -> Tensor:
    """
    Reverse all-to-all for gradients.

    Gathers gradients from all devices back to the original token locations.

    Args:
        grads: List of gradient tensors, one per source device.
        expert_counts: Number of tokens that came from each device.
        devices: List of devices.

    Returns:
        Combined gradient tensor.
    """
    raise NotImplementedError("all_to_all_reverse requires multi-device support")

class PipelineStage:
    """
    Pipeline parallelism stage.

    Wraps a set of layers to form a stage in pipeline parallelism.
    Handles forward/backward pass and communication between stages.
    """

    def __init__(self, layers: List[Callable], stage_id: int, num_stages: int,
                 devices: Optional[List[str]] = None):
        """
        Args:
            layers: List of layers/modules in this stage.
            stage_id: Index of this stage in the pipeline.
            num_stages: Total number of pipeline stages.
            devices: Devices to place this stage on.
        """
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.devices = devices

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through this stage."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through this stage."""
        grad = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
            else:
                grad = grad
        return grad

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class PipelineSchedule:
    """
    Pipeline parallelism scheduler.

    Manages micro-batch scheduling for pipeline parallelism (GPipe/PipeDream).
    """

    def __init__(self, stages: List[PipelineStage], num_microbatches: int = 1,
                 schedule_type: str = "gpipe"):
        """
        Args:
            stages: List of pipeline stages.
            num_microbatches: Number of microbatches for pipeline.
            schedule_type: "gpipe" for GPipe or "pipedream" for PipeDream.
        """
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.schedule_type = schedule_type

    def forward_backward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Run forward and backward pass through the pipeline.

        Args:
            inputs: List of input tensors (one per microbatch).

        Returns:
            List of output tensors.
        """
        raise NotImplementedError("Pipeline scheduling requires distributed support")

__all__ = [
    "MoELayer",
    "GroupedMoELayer",
    "AllToAll",
    "all_to_all",
    "all_to_all_reverse",
    "PipelineStage",
    "PipelineSchedule",
]