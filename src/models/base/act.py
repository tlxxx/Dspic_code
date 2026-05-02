import jax
import jax.numpy as jnp
from flax import linen as nn
import gymnasium as gym
from typing import Any, List, Optional, Dict
from src.models.base.distributions import Categorical, DiagGaussian


class ACTLayer(nn.Module):
    action_space: Any  # gym.Space
    inputs_dim: int
    initialization_method: str
    gain: float
    args: Optional[Dict[str, Any]] = None

    def setup(self):
        self.action_space_type = self.action_space.__class__.__name__
        self.multidiscrete_action = False

        if self.action_space_type == "Discrete":
            self.action_outs = Categorical(
                    self.inputs_dim,
                    num_outputs=self.action_space.n,
                    initialization_method=self.initialization_method,
                    gain=self.gain
                )
        elif self.action_space_type == "Box":
            self.action_outs = DiagGaussian(
                    self.inputs_dim, 
                    num_outputs=self.action_space.shape[0],
                    initialization_method=self.initialization_method,
                    gain=self.gain,
                    args=self.args
                )
        # elif self.action_space_type == "MultiDiscrete":
        #     self.multidiscrete_action = True
        #     action_dims = self.action_space.nvec
        #     self.action_outs = [
        #         Categorical(
        #             self.inputs_dim, 
        #             num_outputs=action_dim,
        #             initialization_method=self.initialization_method,
        #             gain=self.gain
        #         ) for action_dim in action_dims
        #     ]
        else:
            assert 0

    def __call__(self, x: jnp.ndarray, available_actions: Optional[jnp.ndarray] = None):
        if self.multidiscrete_action:
            return [action_out(x, available_actions) for action_out in self.action_outs]
        else: 
            return self.action_outs(x, available_actions)