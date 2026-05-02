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


class SquashedGaussianPolicy(nn.Module):
    """Squashed Gaussian policy network for HASAC."""
    args: dict
    obs_space: gym.Space
    action_space: gym.Space

    def setup(self):
        self.hidden_sizes = self.args["hidden_sizes"]
        self.activation_func = self.args["activation_func"]
        self.final_activation_func = self.args.get("final_activation_func", "identity")
        self.obs_shape = get_shape_from_obs_space(self.obs_space)
        self.action_dim = self.action_space.shape[0]
        self.act_limit = self.action_space.high[0]

    @nn.compact
    def __call__(self, obs):

        if len(self.obs_shape) == 3:
            x = PlainCNN(
                obs_shape=self.obs_shape,
                hidden_size=self.hidden_sizes[0],
                activation_func=self.activation_func,
            )(obs)
            feat_dim = self.hidden_sizes[0]

        else:
            x = obs
            feat_dim = self.obs_shape[0]

        x = PlainMLP(
            sizes=[feat_dim] + [*self.hidden_sizes],
            activation_func=self.activation_func,
            final_activation_func=self.final_activation_func,
        )(x)

        mu = nn.Dense(self.action_dim)(x)  # [N, A]
        log_std = nn.Dense(self.action_dim)(x)  # [N, A]
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    @staticmethod
    @jax.jit
    def get_dist(params, obs, actor_state):
        mu, log_std = actor_state.apply_fn({"params": params}, obs)
        return mu, std

    @staticmethod
    @partial(jax.jit, static_argnames=["act_limit", "stochastic"])
    def sample_action_withlogp(params, obs, key, act_limit, actor_state, stochastic: bool = True):
        mu, log_std = actor_state.apply_fn({"params": params}, obs)
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
    def sample_action_withologp(params, obs, key, act_limit, actor_state, stochastic: bool = True):
        mu, log_std = actor_state.apply_fn({"params": params}, obs)
        std = jnp.exp(log_std)
        if stochastic:
            eps = jax.random.normal(key, shape=mu.shape)
            pre_tanh = mu + std * eps
        else:
            pre_tanh = mu

        pi_action = jnp.tanh(pre_tanh) * act_limit  # [N, A]
        return pi_action, 0


    @staticmethod
    @partial(jax.jit, static_argnames=["act_limit", "stochastic"])
    def sample_action(params, obs, key, act_limit, actor_state, stochastic: bool = True, with_logprob: bool = True):
        mu, log_std = actor_state.apply_fn({"params": params}, obs)
        std = jnp.exp(log_std)
        if stochastic:
            eps = jax.random.normal(key, shape=mu.shape)
            pre_tanh = mu + std * eps
        else:
            pre_tanh = mu

        if with_logprob:
            log2pi = jnp.log(2.0 * jnp.pi)
            logp = -0.5 * (((pre_tanh - mu) / std) ** 2 + 2.0 * log_std + log2pi)
            logp_pi = jnp.sum(logp, axis=-1, keepdims=True)  # [N,1]
            correction = jnp.sum(
                2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)),
                axis=-1, keepdims=True
            )
            logp_pi = logp_pi - correction
        else:
            logp_pi = None

        pi_action = jnp.tanh(pre_tanh) * act_limit  # [N, A]
        return pi_action, logp_pi
