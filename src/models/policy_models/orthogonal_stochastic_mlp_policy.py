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


class OrthogonalStochasticMlpPolicy(nn.Module):
    """Stochastic policy model that only uses MLP network. Outputs actions given observations."""
    args: dict
    obs_space: gym.Space
    action_space: gym.Space
    P_dim: int 

    def setup(self):
        self.hidden_sizes = self.args["hidden_sizes"]
        obs_shape = get_shape_from_obs_space(self.obs_space)

        self.base = MLPBase(args=self.args, obs_shape=obs_shape)
        self.act_layer = ACTLayer(
            action_space=self.action_space,
            inputs_dim=self.hidden_sizes[-1],
            initialization_method=self.args["initialization_method"],
            gain=self.args["gain"],
            args=self.args
        )  

    @nn.compact
    def __call__(self, obs, P, available_actions=None):
        obs = obs.astype(jnp.float32)
        if available_actions is not None:
            available_actions = available_actions.astype(jnp.float32)
        
        P = P.reshape(-1, self.P_dim, self.P_dim)
        obs = nn.Dense(self.P_dim)(obs)
        x = obs.reshape(-1, self.P_dim, 1)
        h = jnp.matmul(P, x).reshape(-1, self.P_dim)
        x = jnp.concatenate((x.reshape(-1, self.P_dim), h), axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        actor_features = nn.relu(x)
        logits = nn.Dense(self.action_space.n)(actor_features)
        if available_actions is not None:
            logits = jnp.where(available_actions == 0, -1e10, logits)

        return logits
    
    @staticmethod
    @partial(jax.jit, static_argnames=["stochastic"])
    def sample_action_withologp(obs, P, key, actor_params, actor_state, available_actions=None, stochastic=True):
        logits = actor_state.apply_fn({"params": actor_params}, obs, P, available_actions)
        dist = FixedCategorical(logits=logits)
        if stochastic:
            actions = dist.sample(seed=key)
        else:
            actions = dist.mode()
        
        log_p = dist.log_probs(actions)

        return actions, log_p
        
    @staticmethod
    @partial(jax.jit, static_argnames=["stochastic"])
    def sample_action_withlogp(obs, P, key, actor_params, actor_state, available_actions=None, stochastic=True):
        logits = actor_state.apply_fn({"params": actor_params}, obs, P, available_actions)
        actions = gumbel_softmax(key, logits, hard=True)
        logp_actions = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1, keepdims=True)
        return actions, logp_actions
    

class Encoder(nn.Module):
    """MLP encoder network."""
    args: dict
    P_dim: int 

    def setup(self):
        self.hidden_sizes = self.args["hidden_sizes"]

    @nn.compact
    def __call__(self, role):
        P = nn.Dense(self.hidden_sizes[0])(role)
        P = nn.relu(P)
        P = nn.Dense(self.P_dim * self.P_dim)(P)
        P = P.reshape(-1, self.P_dim, self.P_dim)

        norms = jnp.linalg.norm(P, axis=(1, 2), keepdims=True)
        P_normalized = P / norms
        
        return P_normalized
    
    def attention_layer(self, role):
        # Use MultiHeadDotProductAttention to capture the relationship between agents
        attention = nn.MultiHeadDotProductAttention(
            num_heads=4,  # Number of attention heads
            dtype=jnp.float32,
            qkv_features=self.hidden_sizes[0],  # Hidden dimension size for query, key, and value
            out_features=self.hidden_sizes[0]  # Output dimension of attention layer
        )
        
        # Apply attention on the input role, assuming role shape is (n_agents, role_dim)
        return attention(role, role)

    @staticmethod
    @jax.jit
    def cal_encoder(latent, params, encoder_state):
        P = encoder_state.apply_fn({"params": params}, latent)
        return P