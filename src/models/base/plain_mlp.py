import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from src.utils.models_tools import get_active_func


class PlainMLP(nn.Module):
    """Plain MLP"""
    sizes: Sequence[int]
    activation_func: str
    final_activation_func: str = "identity"

    @nn.compact
    def __call__(self, x):
        act_mid = get_active_func(self.activation_func)
        act_last = get_active_func(self.final_activation_func)
        for i in range(len(self.sizes) - 1):
            x = nn.Dense(self.sizes[i + 1], name=f"dense_{i}")(x)
            x = act_mid(x) if i < len(self.sizes) - 2 else act_last(x)
        return x