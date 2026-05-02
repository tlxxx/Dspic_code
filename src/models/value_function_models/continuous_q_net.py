import jax
import jax.numpy as jnp
from flax import linen as nn
import gymnasium as gym
from typing import Sequence, Dict, Any
from src.models.base.plain_cnn import PlainCNN
from src.models.base.plain_mlp import PlainMLP
from src.utils.envs_tools import get_shape_from_obs_space


def get_combined_dim(cent_obs_feature_dim, act_spaces):
    """Get the combined dimension of central observation and individual actions."""
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class ContinuousQNet(nn.Module):
    """Q Network for continuous and discrete action space. Outputs the q value given global states and actions.
    Note that the name ContinuousQNet emphasizes its structure that takes observations and actions as input and outputs
    the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space.
    """
    args: Dict[str, Any]
    cent_obs_space: gym.Space
    act_spaces: Sequence[gym.Space]

    def setup(self):
        self.activation_func = self.args["activation_func"]
        hidden_sizes = self.args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(self.cent_obs_space)
        self.sizes = (
            [get_combined_dim(cent_obs_shape[0], self.act_spaces)]
            + list(hidden_sizes)
            + [1]
        )

    @nn.compact
    def __call__(self, cent_obs, actions):
        concat_x = jnp.concatenate([cent_obs, actions], axis=-1)
        q_values = PlainMLP(self.sizes, self.activation_func)(concat_x)
        return q_values
