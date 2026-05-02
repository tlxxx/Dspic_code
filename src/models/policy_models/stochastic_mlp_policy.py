from typing import Sequence, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
import gymnasium as gym
from functools import partial
from src.utils.envs_tools import get_shape_from_obs_space
from src.utils.discrete_util import gumbel_softmax
from src.models.base.mlp import MLPBase
from src.models.base.act import ACTLayer
from src.models.base.distributions import FixedCategorical


class StochasticMlpPolicy(nn.Module):
    """Stochastic policy model that only uses MLP network. Outputs actions given observations."""
    args: dict
    obs_space: gym.Space
    action_space: gym.Space
    
    @nn.compact
    def __call__(self, obs, available_actions=None):
        obs = obs.astype(jnp.float32)
        if available_actions is not None:
            available_actions = available_actions.astype(jnp.float32)

        # actor_features = self.base(obs)
        # obs = nn.LayerNorm()(obs)
        x = nn.Dense(256, name="fc1")(obs)
        x = nn.relu(x)
        x = nn.Dense(256, name="fc2")(x)
        actor_features = nn.relu(x)
        logits = nn.Dense(self.action_space.n)(actor_features)
        if available_actions is not None:
            logits = jnp.where(available_actions == 0, -1e10, logits)

        return logits
    
    @staticmethod
    @partial(jax.jit, static_argnames=["stochastic"])
    def sample_action_withologp(actor_params, obs, key, actor_state, available_actions=None, stochastic=True):
        logits = actor_state.apply_fn({"params": actor_params}, obs, available_actions)
        dist = FixedCategorical(logits=logits)
        if stochastic:
            actions = dist.sample(seed=key)
        else:
            actions = dist.mode()
        logp = dist.log_probs(actions)

        return actions, logp
        
    @staticmethod
    @partial(jax.jit, static_argnames=["stochastic"])
    def sample_action_withlogp(actor_params, obs, key, actor_state, available_actions=None, stochastic=True):
        logits = actor_state.apply_fn({"params": actor_params}, obs, available_actions)
        actions = gumbel_softmax(key, logits, hard=True)
        # logp_actions = jnp.sum(actions * logits, axis=-1, keepdims=True)
        logp_actions = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1, keepdims=True)
        return actions, logp_actions