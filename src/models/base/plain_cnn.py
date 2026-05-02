import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from src.utils.models_tools import get_active_func
from src.models.base.flatten import Flatten
from typing import Tuple, List

class PlainCNN(nn.Module):
    obs_shape: List[int]
    hidden_size: int
    activation_func: str
    kernel_size: int = 3
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        act = get_active_func(self.activation_func)
        x = x.astype(jnp.float32) / 255.0

        x = jnp.transpose(x, (0, 2, 3, 1))  # [N, W, H, C]

        x = nn.Conv(
            features=self.hidden_size // 4,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding="VALID",  
            use_bias=True,
        )(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = act(x)
        x = Flatten()(x)
        x = nn.Dense(self.hidden_size)(x)
        x = act(x)

        return x

class PlainCNN(nn.Module):
    """Plain CNN"""

    def __init__(
        self, obs_shape, hidden_size, activation_func, kernel_size=3, stride=1
    ):
        super().__init__()
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        layers = [
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=hidden_size // 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            get_active_func(activation_func),
            Flatten(),
            nn.Linear(
                hidden_size
                // 4
                * (input_width - kernel_size + stride)
                * (input_height - kernel_size + stride),
                hidden_size,
            ),
            get_active_func(activation_func),
        ]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x
