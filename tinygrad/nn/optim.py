# sorted in order of increasing complexity
import itertools
from tinygrad.helpers import dedup, flatten, getenv, unwrap, FUSE_OPTIM
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, least_upper_dtype, to_dtype
from typing import Optional, List

class Optimizer:
  """
  Base class for all optimizers.
  """
  def __init__(self, params: list[Tensor], lr: float, device=None, fused=FUSE_OPTIM):
    if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
    # if requires_grad is None, but being put into an optimizer, set it to True
    for x in params:
      if x.requires_grad is None: x.requires_grad_(True)

    self.params: list[Tensor] = dedup([x for x in params if x.requires_grad])
    assert len(self.params) != 0, "optimizer must have at least one param"
    self.buffers: list[Tensor] = dedup([x for x in params if not x.requires_grad])   # buffers are still realized
    self.device = device or self.params[0].device
    self.param_dtype = to_dtype(getenv("OPTIM_DTYPE", "float32"))
    self.fused = fused
    # store lr in at least float32 precision
    self.lr = Tensor(lr if getenv("CONST_LR") else [lr], requires_grad=False, device=self.device,
                     dtype=least_upper_dtype(dtypes.default_float, dtypes.float32))
    if self.fused: self.pos_params = list(itertools.accumulate(self.params, lambda x,y: x+y.numel(), initial=0))

  def _new_optim_param(self) -> list[Tensor]:
    if self.fused: return [Tensor.zeros(self.pos_params[-1], dtype=self.param_dtype, device=self.device, requires_grad=False)]
    if isinstance(self.device, tuple): return [Tensor.zeros_like(t, dtype=self.param_dtype, requires_grad=False) for t in self.params]
    else: return [Tensor.zeros(t.shape, dtype=self.param_dtype, device=self.device, requires_grad=False) for t in self.params]

  def zero_grad(self):
    """
    Zeroes the gradients of all the parameters.
    """
    for param in self.params: param.grad = None

  def step(self):
    """
    Performs a single optimization step.
    """
    Tensor.realize(*self.schedule_step())

  def schedule_step(self) -> list[Tensor]:
    """
    Returns the tensors that need to be realized to perform a single optimization step.
    """
    if not Tensor.training: raise RuntimeError(
            f"""Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
                - help: Consider setting Tensor.training=True before calling Optimizer.step().""")
    if self.fused:
      # optimizer fusion just concatenates all the buffers, runs the _step, then splits them back up
      # NOTE: contiguous is for speed
      out, extra = self._step([Tensor.cat(*[t.flatten() for t in self.params], dim=0)],
                              [Tensor.cat(*[unwrap(t.grad).contiguous().flatten() for t in self.params], dim=0)])
      updates = [out[0][self.pos_params[i]:self.pos_params[i+1]].reshape(tt.shape) for i, tt in enumerate(self.params)]
    else:
      updates, extra = self._step(self.params, [unwrap(t.grad) for t in self.params])
    for i, tt in enumerate(self.params): tt.assign(self._apply_update(tt, updates[i]))
    return extra+self.params+self.buffers

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]: raise NotImplementedError
  def _apply_update(self, t:Tensor, up:Tensor) -> Tensor: return t.detach() - up.to(t.device)

class OptimizerGroup(Optimizer):
  """
  Combines multiple optimizers into one.
  """
  def __init__(self, *optimizers: Optimizer): # pylint: disable=super-init-not-called
    self.optimizers = optimizers
    self.params, self.buffers = flatten([o.params for o in self.optimizers]), flatten([o.buffers for o in self.optimizers])
  def __getitem__(self, i): return self.optimizers[i]
  def zero_grad(self): [o.zero_grad() for o in self.optimizers]
  def schedule_step(self) -> list[Tensor]: return [x for o in self.optimizers for x in o.schedule_step()]

# LARS is essentially just trust ratio to SGD so if we just set the trust coeff 0.0 it's just standard SGD.
def SGD(params: list[Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, classic=False, device=None, fused=FUSE_OPTIM):
  """
  Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

  `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.
  """
  return LARS(params, lr, momentum, weight_decay, 0, None, nesterov, classic=classic, pre_wd=True, tcoef=0.0, device=device, fused=fused)

# Muon applies the newton schulz algorithm on gradient. also can include momentum, nesterov, and weight decay
def Muon(params: list[Tensor], lr=0.001, momentum=0.95, weight_decay=0.1, ns_steps=5, ns_coefficients=(3.4445, -4.775, 2.0315),
         nesterov=True, device=None, fused=FUSE_OPTIM):
  """
  SGD with newton-schulz iteration and post momentum weight decay.

  - Described: https://kellerjordan.github.io/posts/muon/
  - Paper: https://arxiv.org/pdf/2502.16982
  """
  assert not fused, "FUSE_OPTIM not allowed for Muon optimizer"
  return LARS(params, lr, momentum, weight_decay, ns_steps, ns_coefficients, nesterov,
              classic=False, pre_wd=False, tcoef=0.0, device=None, fused=fused)

class LARS(Optimizer):
  """
  Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

  - Paper: https://arxiv.org/abs/1708.03888v3
  """
  def __init__(self, params:list[Tensor], lr=0.001, momentum=0.9, weight_decay=1e-4, ns_steps=0, ns_coefficients=None,
               nesterov=False, classic=True, pre_wd=True, tcoef=0.001, device=None, fused=FUSE_OPTIM):
    if momentum < 0: raise ValueError(f"Invalid momentum value: {momentum}")
    super().__init__(params, lr, device, fused)
    self.momentum, self.wd, self.ns_steps, self.ns_coefficients  = momentum, weight_decay, ns_steps, ns_coefficients
    self.nesterov, self.classic, self.pre_wd, self.tcoef = nesterov, classic, pre_wd, tcoef
    self.b = self._new_optim_param() if self.momentum else []

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    for i, (t, g) in enumerate(zip(params, grads)):
      if self.tcoef != 0:
        r1 = t.detach().square().sum().sqrt()
        r2 = g.square().sum().sqrt()
        r:Tensor|float = (r1 > 0).where((r2 > 0).where(self.tcoef * r1 / (r2 + self.wd * r1), 1.0), 1.0)
      else: r = 1.0
      if self.pre_wd and self.wd > 0: g = g + self.wd * t.detach()
      # classic momentum does post learning rate update
      if self.classic: g = g * r * self.lr
      if self.momentum:
        self.b[i].assign(self.momentum * self.b[i] + g)  # NOTE: self.b[i] is zero on the first run, no if required
        g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      if self.ns_coefficients: g = g.reshape(g.shape[0], -1).newton_schulz(self.ns_steps, self.ns_coefficients).reshape(g.shape)
      # muon does post momentum weight decay
      if not self.pre_wd and self.wd > 0: t = t.detach() * (1.0 - self.wd * self.lr)
      # popular momentum does pre learning rate update
      if not self.classic: g = g * r * self.lr
      ret.append(g.cast(t.dtype))
    return ret, self.b

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 it's just Adam/W.
def AdamW(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, device=None, fused=FUSE_OPTIM):
  """
  AdamW optimizer with optional weight decay.

  - Paper: https://arxiv.org/abs/1711.05101v3
  """
  return LAMB(params, lr, b1, b2, eps, weight_decay, adam=True, device=device, fused=fused)
def Adam(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, device=None, fused=FUSE_OPTIM):
  """
  Adam optimizer.

  - Paper: https://arxiv.org/abs/1412.6980
  """
  return LAMB(params, lr, b1, b2, eps, 0.0, adam=True, device=device, fused=fused)

class LAMB(Optimizer):
  """
  LAMB optimizer with optional weight decay.

  - Paper: https://arxiv.org/abs/1904.00962
  """
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=False, device=None, fused=FUSE_OPTIM):
    if weight_decay < 0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    super().__init__(params, lr, device, fused)
    self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
    self.b1_t, self.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=self.device, requires_grad=False) for _ in [b1, b2])
    self.m = self._new_optim_param()
    self.v = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    ret = []
    self.b1_t *= self.b1
    self.b2_t *= self.b2
    for i, (t, g) in enumerate(zip(params, grads)):
      if g.device != self.m[i].device: g = g.contiguous().to(self.m[i].device)
      self.m[i].assign((self.b1 * self.m[i] + (1.0 - self.b1) * g).cast(self.m[i].dtype))
      self.v[i].assign((self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).cast(self.v[i].dtype))
      m_hat = self.m[i] / (1.0 - self.b1_t)
      v_hat = self.v[i] / (1.0 - self.b2_t)
      up = (m_hat / (v_hat.sqrt() + self.eps)).shard_like(t) + self.wd * t.detach()
      if not self.adam:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r: Tensor|float = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      ret.append((self.lr * r * up).cast(t.dtype))
    return ret, [self.b1_t, self.b2_t] + self.m + self.v

class ZeroLAMB(LAMB):
  """
  LAMB optimizer with ZeRO stage 1 (optimizer state sharding).

  Each GPU only stores 1/N of the optimizer states (m, v), reducing memory usage by N.

  - Paper: https://arxiv.org/abs/1910.02054v3 (ZeRO)
  """
  def __init__(self, params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=False, device=None, fused=FUSE_OPTIM,
               zero_stage: int = 1, zero_devices: Optional[List[str]] = None):
    if zero_stage < 1 or zero_stage > 3:
      raise ValueError(f"ZeRO stage must be 1, 2, or 3, got {zero_stage}")
    if zero_devices is not None and len(zero_devices) < 2:
      raise ValueError("ZeRO requires at least 2 devices")
    self.zero_stage = zero_stage
    self.zero_devices = zero_devices
    super().__init__(params, lr, b1, b2, eps, weight_decay, adam, device, fused)

  def _new_optim_param(self) -> list[Tensor]:
    if self.zero_devices is None:
      return super()._new_optim_param()
    n_devs = len(self.zero_devices)
    if self.fused:
      total_size = self.pos_params[-1]
      shard_size = (total_size + n_devs - 1) // n_devs
      ret = []
      for i, d in enumerate(self.zero_devices):
        start = i * shard_size
        end = min(start + shard_size, total_size)
        if start < total_size:
          ret.append(Tensor.zeros(end - start, dtype=self.param_dtype, device=d, requires_grad=False))
      return ret
    ret = []
    for t in self.params:
      shard_size = (t.numel() + n_devs - 1) // n_devs
      for i, d in enumerate(self.zero_devices):
        start = i * shard_size
        end = min(start + shard_size, t.numel())
        if start < t.numel():
          ret.append(Tensor.zeros(end - start, dtype=self.param_dtype, device=d, requires_grad=False))
    return ret

    def _step(self, params: list[Tensor], grads: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        if self.zero_devices is None or self.zero_stage == 0:
            return super()._step(params, grads)
        
        n_devices = len(self.zero_devices)
        ret = []
        
        for i, (t, g) in enumerate(zip(params, grads)):
            shard_size = (t.numel() + n_devices - 1) // n_devices
            device_idx = i % n_devices
            
            if self.fused:
                # Fused path
                pass
            else:
                # Non-fused ZeRO-1: optimizer states are sharded across devices
                m_shard = self.m[i]
                v_shard = self.v[i]
                
                # Compute local update
                self.b1_t *= self.b1
                self.b2_t *= self.b2
                
                m_shard.assign((self.b1 * m_shard + (1.0 - self.b1) * g).cast(m_shard.dtype))
                v_shard.assign((self.b2 * v_shard + (1.0 - self.b2) * (g * g)).cast(v_shard.dtype))
                
                m_hat = m_shard / (1.0 - self.b1_t)
                v_hat = v_shard / (1.0 - self.b2_t)
                
                up = (m_hat / (v_hat.sqrt() + self.eps)).shard_like(t) + self.wd * t.detach()
                
                if not self.adam:
                    r1 = t.detach().square().sum().sqrt()
                    r2 = up.square().sum().sqrt()
                    r = t.where(r1 > 0, t.where(r2 > 0, r1 / r2, 1.0), 1.0)
                else:
                    r = 1.0
                
                ret.append((self.lr * r * up).cast(t.dtype))
        
        return ret, [self.b1_t, self.b2_t] + self.m + self.v

def ZeroAdam(params: list[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, device=None, fused=FUSE_OPTIM,
             zero_stage: int = 1, zero_devices: Optional[List[str]] = None):
  """
  AdamW optimizer with ZeRO (Zero Redundancy Optimizer) support.

  - ZeRO-1: Shard optimizer states across GPUs
  - ZeRO-2: Shard gradients + optimizer states (not implemented)
  - ZeRO-3: Shard parameters + gradients + optimizer states (not implemented)

  Args:
      params: List of parameters to optimize.
      lr: Learning rate.
      b1: Beta1 coefficient for momentum.
      b2: Beta2 coefficient for second moment.
      eps: Epsilon for numerical stability.
      weight_decay: Weight decay coefficient.
      device: Device for optimizer state.
      fused: Whether to use fused optimizer.
      zero_stage: ZeRO stage (1, 2, or 3).
      zero_devices: List of devices for ZeRO sharding.

  Returns:
      ZeroLAMB optimizer instance.
  """
  return ZeroLAMB(params, lr, b1, b2, eps, weight_decay, adam=True, device=device, fused=fused,
                  zero_stage=zero_stage, zero_devices=zero_devices)

def ZeroSGD(params: list[Tensor], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=False, device=None, fused=FUSE_OPTIM,
            zero_stage: int = 1, zero_devices: Optional[List[str]] = None):
  """
  SGD optimizer with ZeRO support.

  Args:
      params: List of parameters to optimize.
      lr: Learning rate.
      momentum: Momentum coefficient.
      weight_decay: Weight decay coefficient.
      nesterov: Whether to use Nesterov momentum.
      device: Device for optimizer state.
      fused: Whether to use fused optimizer.
      zero_stage: ZeRO stage (1, 2, or 3).
      zero_devices: List of devices for ZeRO sharding.

  Returns:
      ZeroLARS optimizer instance (SGD is LARS with trust coefficient 0).
  """
  return ZeroLARS(params, lr, momentum, weight_decay, 0, None, nesterov, classic=True, pre_wd=True, tcoef=0.0,
                  device=device, fused=fused, zero_stage=zero_stage, zero_devices=zero_devices)

class ZeroLARS(LARS):
  """
  LARS optimizer with ZeRO stage 1 (optimizer state sharding).
  """
  def __init__(self, params: list[Tensor], lr=0.001, momentum=0.9, weight_decay=1e-4, ns_steps=0, ns_coefficients=None,
               nesterov=False, classic=True, pre_wd=True, tcoef=0.001, device=None, fused=FUSE_OPTIM,
               zero_stage: int = 1, zero_devices: Optional[List[str]] = None):
    if zero_devices is not None and len(zero_devices) < 2:
      raise ValueError("ZeRO requires at least 2 devices")
    self.zero_stage = zero_stage
    self.zero_devices = zero_devices
    super().__init__(params, lr, momentum, weight_decay, ns_steps, ns_coefficients, nesterov, classic, pre_wd, tcoef, device, fused)

    def _new_optim_param(self) -> list[Tensor]:
        if self.zero_devices is None:
            return super()._new_optim_param()
        n_devs = len(self.zero_devices)
        if self.fused:
            total_size = self.pos_params[-1]
            shard_size = (total_size + n_devs - 1) // n_devs
            ret = []
            for i, d in enumerate(self.zero_devices):
                start = i * shard_size
                end = min(start + shard_size, total_size)
                if start < total_size:
                    ret.append(Tensor.zeros(end - start, dtype=self.param_dtype, device=d, requires_grad=False))
            return ret
        if not self.momentum:
            return []
        ret = []
        for t in self.params:
            shard_size = (t.numel() + n_devs - 1) // n_devs
            for i, d in enumerate(self.zero_devices):
                start = i * shard_size
                end = min(start + shard_size, t.numel())
                if start < t.numel():
                    ret.append(Tensor.zeros(end - start, dtype=self.param_dtype, device=d, requires_grad=False))
        return ret

    def _step(self, params: list[Tensor], grads: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        if self.zero_devices is None or self.zero_stage == 0:
            return super()._step(params, grads)
        
        n_devices = len(self.zero_devices)
        ret = []
        
        for i, (t, g) in enumerate(zip(params, grads)):
            if not self.momentum:
                # No momentum: just return the update
                if self.pre_wd and self.wd > 0:
                    g = g + self.wd * t.detach()
                if self.classic:
                    g = g * self.lr
                if not self.classic:
                    g = g * self.lr
                ret.append(g.cast(t.dtype))
            else:
                shard_idx = i % n_devices
                b_shard = self.b[i]
                
                # Local update on shard
                if self.pre_wd and self.wd > 0:
                    g = g + self.wd * t.detach()
                if self.classic:
                    g = g * self.lr
                b_shard.assign(self.momentum * b_shard + g)
                g = (g + self.momentum * b_shard) if self.nesterov else b_shard
                if not self.classic:
                    g = g * self.lr
                ret.append(g.cast(t.dtype))
        
        return ret, self.b
