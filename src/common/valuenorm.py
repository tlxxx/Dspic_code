"""ValueNorm."""
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass

@jax.tree_util.register_pytree_node_class
@dataclass
class ValueNormState:
    running_mean: jnp.ndarray
    running_mean_sq: jnp.ndarray
    debiasing_term: jnp.ndarray  # scalar

    # PyTree 协议
    def tree_flatten(self):
        children = (self.running_mean, self.running_mean_sq, self.debiasing_term)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        rm, rmsq, deb = children
        return cls(rm, rmsq, deb)

@jax.jit
def _vn_mean_var(state: ValueNormState, epsilon: float):
    deb = jnp.clip(state.debiasing_term, a_min=epsilon)
    mean = state.running_mean / deb
    mean_sq = state.running_mean_sq / deb
    var = jnp.clip(mean_sq - mean * mean, a_min=1e-2)  # 与原版同样的下界
    return mean, var

@partial(jax.jit, static_argnames=["norm_axes", "per_element_update"])
def _vn_update(state: ValueNormState,
               x: jnp.ndarray,
               beta: float,
               norm_axes: int,
               per_element_update: bool) -> ValueNormState:
    x = jnp.asarray(x, jnp.float32)
    axes = tuple(range(norm_axes))
    b_mean = jnp.mean(x, axis=axes)
    b_sqmean = jnp.mean(x * x, axis=axes)

    # 与原实现一致：可选按样本元素数调整 EMA 权重
    weight = beta ** (math.prod(x.shape[:norm_axes]) if per_element_update else 1)

    rm = state.running_mean * weight + b_mean * (1.0 - weight)
    rmsq = state.running_mean_sq * weight + b_sqmean * (1.0 - weight)
    deb = state.debiasing_term * weight + (1.0 - weight)
    return ValueNormState(rm, rmsq, deb)

@partial(jax.jit, static_argnames=["norm_axes"])
def _vn_normalize(state: ValueNormState, x: jnp.ndarray, norm_axes: int, epsilon: float):
    x = jnp.asarray(x, jnp.float32)
    mean, var = _vn_mean_var(state, epsilon)
    b = (None,) * norm_axes
    return (x - mean[b]) / jnp.sqrt(var)[b]

@partial(jax.jit, static_argnames=["norm_axes"])
def _vn_denormalize(state: ValueNormState, x: jnp.ndarray, norm_axes: int, epsilon: float):
    x = jnp.asarray(x, jnp.float32)
    mean, var = _vn_mean_var(state, epsilon)
    b = (None,) * norm_axes
    return x * jnp.sqrt(var)[b] + mean[b]

class ValueNorm:
    """
    JAX 版本的 ValueNorm：
      - self.state: 保存 running_mean / running_mean_sq / debiasing_term
      - update(x): 更新 state（内部赋回新 state）
      - normalize(x) / denormalize(x)
      - running_mean_var()
    """
    def __init__(self,
                 input_shape,
                 norm_axes: int = 1,
                 beta: float = 0.99999,
                 per_element_update: bool = False,
                 epsilon: float = 1e-5
                 ):
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.beta = beta
        self.per_element_update = per_element_update
        self.epsilon = epsilon

        # 初始化为全零，与 torch 版一致
        rm = jnp.zeros(self.input_shape, jnp.float32)
        rmsq = jnp.zeros(self.input_shape, jnp.float32)
        deb = jnp.array(0.0, jnp.float32)
        self.state = ValueNormState(rm, rmsq, deb)

    def running_mean_var(self):
        return _vn_mean_var(self.state, self.epsilon)

    def update(self, input_vector):
        # x = jnp.asarray(input_vector, jnp.float32)
        self.state = _vn_update(self.state, input_vector,
                                self.beta, self.norm_axes,
                                self.per_element_update)

    def normalize(self, input_vector):
        # x = jnp.asarray(input_vector, jnp.float32)
        return _vn_normalize(self.state, input_vector, self.norm_axes, self.epsilon)

    def denormalize(self, input_vector):
        # x = jnp.asarray(input_vector, jnp.float32)
        return _vn_denormalize(self.state, input_vector, self.norm_axes, self.epsilon)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        serialized = serialization.to_bytes(self.state)  # 只保存 state（running_mean 等）

        filepath = os.path.join(save_dir, "value_normalizer.msgpack")
        with open(filepath, "wb") as f:
            f.write(serialized)

        print(f"Value normalizer state saved to {save_dir}")

    def restore(self, model_dir):
        filepath = os.path.join(model_dir, 'value_normalizer.msgpack')
        with open(filepath, 'rb') as f:
            raw = f.read()
        self.state = serialization.from_bytes(self.state, raw)

        print(f"Critic state restored from {model_dir}")