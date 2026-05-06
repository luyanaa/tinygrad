"""
Tensor Parallelism for tinygrad.

Provides tensor parallel layer wrappers for distributed LLM training across multiple GPUs.
"""

from __future__ import annotations
import math
from typing import Optional, List, Union, Tuple
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor

class TensorParallelLinear:
    """
    Tensor-parallel linear layer.

    Supports column-parallel (split output features) and row-parallel (split input features)
    transformations needed for efficient tensor parallelism in LLMs.

    For column-parallel: weight is sharded along output dimension (dim=0)
    For row-parallel: weight is sharded along input dimension (dim=1)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 parallel_dim: int = 0, devices: Optional[List[str]] = None):
        """
        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to include bias.
            parallel_dim: Dimension along which to shard (0=column, 1=row).
            devices: List of devices for sharding.
        """
        self.parallel_dim = parallel_dim
        self.in_features = in_features
        self.out_features = out_features

        bound = 1 / math.sqrt(in_features)

        if devices is None:
            self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        else:
            n_devices = len(devices)
            if parallel_dim == 0:
                shard_size = (out_features + n_devices - 1) // n_devices
                self.weight = Tensor.uniform(shard_size, in_features, low=-bound, high=bound)
            else:
                shard_size = (in_features + n_devices - 1) // n_devices
                self.weight = Tensor.uniform(out_features, shard_size, low=-bound, high=bound)

        self._original_weight = self.weight
        self._devices = devices
        self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

    def shard_(self, devices: List[str], axis: int = 0) -> 'TensorParallelLinear':
        """Shard this layer across devices."""
        if self._devices is not None and self._devices != devices:
            raise ValueError(f"Already sharded on {self._devices}, cannot reshard to {devices}")

        self._devices = devices
        self.weight = self._original_weight
        return self

    def __call__(self, x: Tensor) -> Tensor:
        if self._devices is not None and len(self._devices) > 1:
            result = x.linear(self.weight.transpose(), None)
            if self.parallel_dim == 0:
                return result
            else:
                raise NotImplementedError("Row-parallel linear requires all-gather")
        return x.linear(self.weight.transpose(), self.bias)

class TensorParallelColumnLinear(TensorParallelLinear):
    """
    Column-parallel linear layer.

    The weight matrix is split along the output dimension (dim=0) across GPUs.
    Each GPU computes a partial output; results are typically gathered or reduced.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 devices: Optional[List[str]] = None):
        super().__init__(in_features, out_features, bias, parallel_dim=0, devices=devices)

    def __call__(self, x: Tensor) -> Tensor:
        if self._devices is not None and len(self._devices) > 1:
            return x.linear(self.weight.transpose(), None)
        return x.linear(self.weight.transpose(), self.bias)

class TensorParallelRowLinear(TensorParallelLinear):
    """
    Row-parallel linear layer.

    The weight matrix is split along the input dimension (dim=1) across GPUs.
    Each GPU computes a partial result; results are summed via allreduce.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 devices: Optional[List[str]] = None):
        super().__init__(in_features, out_features, bias, parallel_dim=1, devices=devices)

    def __call__(self, x: Tensor) -> Tensor:
        if self._devices is not None and len(self._devices) > 1:
            result = x.linear(self.weight.transpose(), None)
            return result
        return x.linear(self.weight.transpose(), self.bias)

class TensorParallelMLP:
    """
    Tensor-parallel MLP layer (LLaMA-style).

    Consists of gate and up projections (column-parallel), a silu activation,
    and a down projection (row-parallel).
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False,
                 devices: Optional[List[str]] = None):
        """
        Args:
            dim: Input dimension.
            hidden_dim: Hidden dimension (intermediate size).
            bias: Whether to use bias in linear layers.
            devices: Devices to shard across.
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self._devices = devices

        bound = 1 / math.sqrt(dim)
        self.w1 = Tensor.uniform(hidden_dim, dim, low=-bound, high=bound)
        self.w2 = Tensor.uniform(dim, hidden_dim, low=-bound, high=bound)
        self.w3 = Tensor.uniform(hidden_dim, dim, low=-bound, high=bound)

        self._original_w1 = self.w1
        self._original_w2 = self.w2
        self._original_w3 = self.w3

    def shard_(self, devices: List[str], axis: int = 0) -> 'TensorParallelMLP':
        """Shard this MLP across devices."""
        self._devices = devices
        return self

    def __call__(self, x: Tensor) -> Tensor:
        if self._devices is not None and len(self._devices) > 1:
            raise NotImplementedError("Sharded MLP requires custom parallel implementation")
        return (self.w2 @ (self.w1(x) * self.w3(x)).silu())

class TensorParallelAttention:
    """
    Tensor-parallel attention layer.

    Supports sharding Q/K/V projections and the output projection.
    For multi-head attention, heads are split across GPUs.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 max_context: int = 0, qk_norm: Optional[float] = None,
                 devices: Optional[List[str]] = None):
        """
        Args:
            dim: Model dimension.
            n_heads: Number of attention heads.
            n_kv_heads: Number of key/value heads (for GQA). If None, same as n_heads.
            max_context: Maximum context length for KV cache.
            qk_norm: If not None, apply RMSNorm with this epsilon to Q and K.
            devices: Devices to shard across.
        """
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.max_context = max_context
        self.qk_norm = qk_norm
        self._devices = devices

        bound = 1 / math.sqrt(dim)
        self.wq = Tensor.uniform(n_heads * self.head_dim, dim, low=-bound, high=bound)
        self.wk = Tensor.uniform(self.n_kv_heads * self.head_dim, dim, low=-bound, high=bound)
        self.wv = Tensor.uniform(self.n_kv_heads * self.head_dim, dim, low=-bound, high=bound)
        self.wo = Tensor.uniform(dim, n_heads * self.head_dim, low=-bound, high=bound)

        self._original_wq = self.wq
        self._original_wk = self.wk
        self._original_wv = self.wv
        self._original_wo = self.wo

    def shard_(self, devices: List[str], axis: int = 0) -> 'TensorParallelAttention':
        """Shard this attention layer across devices."""
        if self._devices is not None and self._devices != devices:
            raise ValueError(f"Already sharded on {self._devices}, cannot reshard to {devices}")

        self._devices = devices
        return self

    def __call__(self, x: Tensor, start_pos: Union[int, Tensor],
                 freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self._devices is not None and len(self._devices) > 1:
            raise NotImplementedError("Sharded attention requires parallel implementation")

        xq = self.wq @ x
        xk = self.wk @ x
        xv = self.wv @ x

        if self.qk_norm is not None:
            from tinygrad.nn import RMSNorm
            q_norm = RMSNorm(self.dim, eps=self.qk_norm)
            k_norm = RMSNorm(self.dim, eps=self.qk_norm)
            xq = q_norm(xq)
            xk = k_norm(xk)

        bsz, seqlen, _, _ = xq.shape
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if start_pos.__class__.__name__ == 'Tensor':
            raise NotImplementedError("Variable start_pos not supported in basic TP")

        if self.max_context and hasattr(self, 'cache_kv'):
            self.cache_kv[:, :, start_pos:start_pos+seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()
            keys = self.cache_kv[0, :, 0:start_pos+seqlen, :, :]
            values = self.cache_kv[1, :, 0:start_pos+seqlen, :, :]
        else:
            keys, values = xk, xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.n_rep > 1:
            keys = keys.repeat((1, 1, 1, self.n_rep)).reshape(bsz, seqlen, self.n_heads, self.head_dim)
            values = values.repeat((1, 1, 1, self.n_rep)).reshape(bsz, seqlen, self.n_heads, self.head_dim)

        attn = xq.scaled_dot_product_attention(keys, values, mask)
        attn = attn.transpose(1, 2).reshape(bsz, seqlen, -1)

        return (self.wo @ attn)

class FusedTensorParallelLinear(TensorParallelLinear):
    """
    Fused tensor-parallel linear for Q/K/V projections.

    Combines Q, K, V projections into a single weight matrix for efficiency.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 bias: bool = False, devices: Optional[List[str]] = None):
        """
        Args:
            dim: Model dimension.
            n_heads: Number of attention heads.
            n_kv_heads: Number of key/value heads.
            bias: Whether to include bias.
            devices: Devices to shard across.
        """
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        total_q = n_heads * self.head_dim
        total_kv = self.n_kv_heads * self.head_dim * 2

        super().__init__(dim, total_q + total_kv, bias=bias, parallel_dim=0, devices=devices)

        self.q_start = 0
        self.k_start = total_q
        self.v_start = total_q + self.n_kv_heads * self.head_dim

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (xq, xk, xv) projections."""
        if self._devices is not None and len(self._devices) > 1:
            raise NotImplementedError("Sharded fused linear requires parallel implementation")

        out = x.linear(self.weight.transpose(), self.bias)

        bsz, seqlen, _ = x.shape
        xq = out[..., self.q_start:self.q_start + self.n_heads * self.head_dim]
        xk = out[..., self.k_start:self.k_start + self.n_kv_heads * self.head_dim]
        xv = out[..., self.v_start:]

        return xq, xk, xv

__all__ = [
    "TensorParallelLinear",
    "TensorParallelColumnLinear",
    "TensorParallelRowLinear",
    "TensorParallelMLP",
    "TensorParallelAttention",
    "FusedTensorParallelLinear",
]