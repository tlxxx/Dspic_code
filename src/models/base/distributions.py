"""Modify standard PyTorch distributions so they to make compatible with this codebase."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import constant, orthogonal
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Optional, Dict, Any
from src.utils.models_tools import get_init_method

class FixedCategorical:
    """
    A wrapper for tfd.Categorical to match the PyTorch FixedCategorical interface.

    Ensures that sample() and log_probs() outputs have a trailing dimension of size 1.
    """
    def __init__(self, logits: jnp.ndarray):
        self.dist = tfd.Categorical(logits=logits)

    def sample(self, seed: jax.random.PRNGKey) -> jnp.ndarray:
        """Samples from the distribution, adding a trailing dimension."""
        return jnp.expand_dims(self.dist.sample(seed=seed), -1)

    def log_probs(self, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Computes log probabilities, expecting and returning tensors with a 
        trailing dimension.
        """
        # Squeeze the trailing dimension from actions for TFP
        squeezed_actions = jnp.squeeze(actions, axis=-1)
        log_p = self.dist.log_prob(squeezed_actions)
        # Add the trailing dimension back to the output
        return jnp.expand_dims(log_p, -1)

    def mode(self) -> jnp.ndarray:
        """Returns the mode of the distribution, adding a trailing dimension."""
        return jnp.expand_dims(self.dist.mode(), -1)
    
    @property
    def logits(self):
        return self.dist.logits

    @property
    def probs(self):
        return self.dist.probs



class FixedNormal:
    """
    A wrapper for tfd.Normal to match the PyTorch FixedNormal interface.
    """
    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray):
        self.dist = tfd.Normal(loc=loc, scale=scale)

    def log_probs(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Alias for log_prob for interface consistency."""
        return self.dist.log_prob(actions)

    def entropy(self) -> jnp.ndarray:
        """
        Computes the entropy and sums over the last dimension to match the
        PyTorch implementation's behavior for diagonal Gaussians.
        """
        return self.dist.entropy().sum(axis=-1)

    def mode(self) -> jnp.ndarray:
        """Returns the mode of the distribution, which is its mean (loc)."""
        return self.dist.mode()
    
    @property
    def mean(self):
        return self.dist.mean


class Categorical(nn.Module):
    num_inputs: int 
    num_outputs: int
    initialization_method: str = "orthogonal_"
    gain: float = 0.01

    @nn.compact
    def __call__(self, x: jnp.ndarray, available_actions: jnp.ndarray = None) -> "FixedCategorical":
        weight_init_fn = get_init_method(self.initialization_method)
        weight_initializer = weight_init_fn(scale=self.gain)
        bias_initializer = jax.nn.initializers.constant(0.0)
        logits = nn.Dense(
            features=self.num_outputs,
            kernel_init=weight_initializer, 
            bias_init=bias_initializer      
        )(x)

        if available_actions is not None:
            logits = jnp.where(available_actions == 0, -1e10, logits)
    
        return logits

class DiagGaussian(nn.Module):
    """
    A Flax nn.Module equivalent to the PyTorch DiagGaussian layer.

    It produces a diagonal Gaussian distribution. The mean is calculated via a
    Dense layer whose initialization is dynamically configured by a string.
    The standard deviation is derived from a separate, learnable parameter (log_std).
    """
    num_inputs: int
    num_outputs: int
    initialization_method: str = "orthogonal_"
    gain: float = 0.01
    args: Optional[Dict[str, Any]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, available_actions: Optional[jnp.ndarray] = None) -> FixedNormal:
        """
        Produces a FixedNormal (diagonal Gaussian) distribution.
        The `available_actions` argument is ignored but kept for interface compatibility.
        """
        weight_init_fn = get_init_method(self.initialization_method)
        weight_initializer = weight_init_fn(scale=self.gain)
        bias_initializer = jax.nn.initializers.constant(0.0)
        action_mean = nn.Dense(
            features=self.num_outputs,
            kernel_init=weight_initializer,
            bias_init=bias_initializer,
            name="fc_mean"
        )(x)
        if self.args is not None:
            std_x_coef = self.args.get("std_x_coef", 1.0)
            std_y_coef = self.args.get("std_y_coef", 0.5)
        else:
            std_x_coef = 1.0
            std_y_coef = 0.5
        log_std_initializer = lambda key, shape, dtype: jnp.ones(shape, dtype) * std_x_coef
        
        log_std = self.param(
            'log_std',
            init_fn=log_std_initializer,
            shape=(self.num_outputs,),
            dtype=jnp.float32
        )
        action_std = nn.sigmoid(log_std / std_x_coef) * std_y_coef

        return FixedNormal(loc=action_mean, scale=action_std)
