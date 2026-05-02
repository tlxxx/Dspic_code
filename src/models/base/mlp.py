import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Dict, Any, Callable
from jax.nn import initializers
from src.utils.models_tools import get_active_func, get_init_method

"""MLP modules."""
def calculate_gain_jax(activation_func_str: str) -> float:
    if activation_func_str in ['relu', 'leaky_relu']:
        return jnp.sqrt(2.0)
    elif activation_func_str == 'tanh':
        return 5.0 / 3.0
    elif activation_func_str in ['sigmoid', 'linear']:
        return 1.0
    else:
        return 1.0


class MLPLayer(nn.Module):
    input_dim: int 
    hidden_sizes: Sequence[int]
    initialization_method: str
    activation_func: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        active_func = get_active_func(self.activation_func)
        init_method = get_init_method(self.initialization_method)
        gain = calculate_gain_jax(self.activation_func)
        kernel_initializer = init_method(scale=gain)
        bias_initializer = initializers.constant(0.0)

        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(
                features=hidden_size,
                kernel_init=kernel_initializer,
                bias_init=bias_initializer,
                name=f"fc_{i}" 
            )(x)
            x = active_func(x)
            x = nn.LayerNorm(name=f"ln_{i}")(x)

        return x

class MLPBase(nn.Module):
    args: Dict[str, Any]
    obs_shape: Sequence[int]

    def setup(self):
        self.use_feature_normalization = self.args["use_feature_normalization"]
        obs_dim = self.obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(name="feature_norm")
            
        self.mlp = MLPLayer(
            input_dim = obs_dim, 
            hidden_sizes=self.args["hidden_sizes"],
            initialization_method=self.args["initialization_method"],
            activation_func=self.args["activation_func"]
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        
        x = self.mlp(x)
        
        return x