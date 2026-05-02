from typing import Sequence, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
import gymnasium as gym
from functools import partial
from src.utils.envs_tools import get_shape_from_obs_space
from src.models.base.plain_cnn import PlainCNN
from src.models.base.plain_mlp import PlainMLP

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class OrthogonalGaussianPolicy(nn.Module):
    """Squashed Gaussian policy network for HASAC with hyper-net."""
    args: dict
    obs_space: gym.Space
    action_space: gym.Space
    n_agents: int
    P_dim: int

    def setup(self):
        self.hidden_sizes = self.args["hidden_sizes"]
        self.activation_func = self.args["activation_func"]
        self.final_activation_func = self.args.get("final_activation_func", "identity")
        self.obs_shape = get_shape_from_obs_space(self.obs_space)
        self.action_dim = self.action_space.shape[0]
        self.act_limit = self.action_space.high[0]

    @nn.compact
    def __call__(self, obs, P):
        feat_dim = self.obs_shape[0]
        P = P.reshape(-1, self.P_dim, self.P_dim)
        obs = nn.Dense(self.P_dim)(obs)
        x = obs.reshape(-1, self.P_dim, 1)
        h = jnp.matmul(P, x).reshape(-1, self.P_dim)
        x = jnp.concatenate((x.reshape(-1, self.P_dim), h), axis=-1)

        x = PlainMLP(
            sizes=[2 * self.P_dim] + [*self.hidden_sizes],
            activation_func=self.activation_func,
            final_activation_func=self.final_activation_func,
        )(x)

        mu = nn.Dense(self.action_dim)(x)  
        log_std = nn.Dense(self.action_dim)(x)  
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std


    @staticmethod
    @jax.jit
    def get_dist(params, obs, latent, actor_state):
        mu, log_std = actor_state.apply_fn({"params": params}, obs, latent)
        return mu, log_std

    @staticmethod
    @partial(jax.jit, static_argnames=["act_limit", "stochastic"])
    def sample_action_withlogp(params, obs, latent, key, act_limit, actor_state, stochastic: bool = True):
        mu, log_std = actor_state.apply_fn({"params": params}, obs, latent)
        std = jnp.exp(log_std)
        if stochastic:
            eps = jax.random.normal(key, shape=mu.shape)
            pre_tanh = mu + std * eps
        else:
            pre_tanh = mu

        log2pi = jnp.log(2.0 * jnp.pi)
        logp = -0.5 * (((pre_tanh - mu) / std) ** 2 + 2.0 * log_std + log2pi)
        logp_pi = jnp.sum(logp, axis=-1, keepdims=True)  # [N,1]
        correction = jnp.sum(
            2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)),
            axis=-1, keepdims=True
        )
        logp_pi = logp_pi - correction

        pi_action = jnp.tanh(pre_tanh) * act_limit  # [N, A]
        return pi_action, logp_pi

    @staticmethod
    @partial(jax.jit, static_argnames=["act_limit", "stochastic"])
    def sample_action_withologp(params, obs, latent, key, act_limit, actor_state, stochastic: bool = True):
        mu, log_std = actor_state.apply_fn({"params": params}, obs, latent)
        std = jnp.exp(log_std)
        if stochastic:
            eps = jax.random.normal(key, shape=mu.shape)
            pre_tanh = mu + std * eps
        else:
            pre_tanh = mu

        pi_action = jnp.tanh(pre_tanh) * act_limit  # [N, A]
        return pi_action, None
    
class Encoder(nn.Module):
    """MLP encoder network."""
    args: dict
    P_dim: int 

    def setup(self):
        self.hidden_sizes = self.args["hidden_sizes"]

    @nn.compact
    def __call__(self, role):
        # attention_output = self.attention_layer(role)
        P = nn.Dense(self.hidden_sizes[0])(role)
        P = nn.relu(P)
        P = nn.Dense(self.P_dim * self.P_dim)(P)
        P = P.reshape(-1, self.P_dim, self.P_dim)

        norms = jnp.linalg.norm(P, axis=(1, 2), keepdims=True)
        P_normalized = P / (norms + 1e-8)
        
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