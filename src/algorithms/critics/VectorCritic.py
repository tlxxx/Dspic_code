from typing import Sequence, Optional, Type, Callable, Any, Union, List, Dict, Tuple

import gymnasium as gym
import os
import jax
import flax
from functools import partial
from math import log, exp
import numpy as np

import flax.linen as nn
import jax.numpy as jnp
import optax
from flax.linen import initializers
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from gymnasium import spaces
from src.utils.envs_tools import get_shape_from_obs_space
from flax.training.train_state import TrainState
from src.utils.envs_tools import check
from src.models.base.plain_cnn import PlainCNN
from src.utils.models_tools import get_active_func

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]

class RLCriticTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


class BatchRenorm(Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
  and adapted from Flax's BatchNorm implementation:
  https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228


  Attributes:
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    bn_warmup: int = 100_000
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """
    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r ** 2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100_000 steps to build up proper running statistics
                # warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                warmed_up = jnp.greater_equal(steps.value, self.bn_warmup).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                        self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    bn_warmup: int = 100_000
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    bn_mode: str = "bn"
    n_atoms: int = 101

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:
        if 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        elif 'brn' in self.bn_mode:
            BN = BatchRenorm
        else:
            raise NotImplementedError

        x = jnp.concatenate([x, action], -1)

        if self.use_batch_norm:
            x = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Hack to make flax return state_updates. Is only necessary such that the downstream
            # functions have the same function signature.
            x_dummy = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            # x = self.activation_fn()(x)
            x = nn.relu(x)

            if self.use_batch_norm:
                x = BN(bn_warmup=self.bn_warmup,use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        x = nn.Dense(self.n_atoms)(x)
        # x = nn.Dense(1, kernel_init=nn.initializers.constant(1e-6),
        #                 bias_init=nn.initializers.constant(0.0))(x)
        if self.n_atoms > 1:
            x = jax.nn.softmax(x, axis=-1)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float
    bn_warmup: int = 100_000
    use_batch_norm: bool = False
    batch_norm_mode: str = "bn"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    n_atoms: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_warmup=self.bn_warmup,
            bn_mode=self.batch_norm_mode,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            n_atoms=self.n_atoms
        )(obs, action, train)
        return q_values

class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)

class SoftVectorCritic:
    def __init__(self, args, share_obs_space, act_space, num_agents, state_type, batch_size, key, dtype=jnp.float32):
        self.args = args
        self.share_obs_space = share_obs_space
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.batch_size = batch_size
        self.key = key
        self.dtype = dtype
        self.use_proper_time_limits = self.args["use_proper_time_limits"]
        self.lr = self.args["critic_lr"]
        self.build_()

    def build_(self):
        self.action_type = self.act_space[0].__class__.__name__
        activation_func = self.args["activation_func"]
        hidden_sizes = self.args["hidden_sizes"]
        self.critic = VectorCritic(
            dropout_rate=self.args["dropout_rate"],
            use_layer_norm=self.args["use_layer_norm"],
            use_batch_norm=self.args["use_batch_norm"],
            bn_warmup=self.args["bn_warmup"],
            batch_norm_momentum=self.args["bn_momentum"],
            batch_norm_mode=self.args["bn_mode"],
            net_arch=self.args["critic_hs"], 
            activation_fn=get_active_func(self.args["critic_activation"]),
            n_critics=self.args["n_critics"],
            n_atoms=self.args["n_atoms"],
        )
        self.key, self.noise_key, critic_key, dropout_key, bn_key = jax.random.split(self.key, 5)
        cent_obs_shape = get_shape_from_obs_space(self.share_obs_space)
        if len(cent_obs_shape) == 3:
            assert 0
            self.feature_extractor = PlainCNN(
                cent_obs_shape, hidden_sizes[0], activation_func
            )
            obs_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            obs_dim = cent_obs_shape[0]
        actions_dim = 0
        for space in self.act_space:
            if space.__class__.__name__ == "Box":
                actions_dim += space.shape[0]
            elif space.__class__.__name__ == "Discrete":
                actions_dim += space.n
            else:
                action_dims = space.nvec
                for action_dim in action_dims:
                    actions_dim += action_dim

        critic_variables = self.critic.init(
            {"params": critic_key, "dropout": dropout_key, "batch_stats": bn_key},
            jnp.ones((self.batch_size, obs_dim)),
            jnp.ones((self.batch_size, actions_dim)),
            train=False,
        )

        target_critic_variables = self.critic.init(
            {"params": critic_key, "dropout": dropout_key, "batch_stats": bn_key},
            jnp.ones((self.batch_size, obs_dim)),
            jnp.ones((self.batch_size, actions_dim)),
            train=False,
        )

        self.critic_state = RLCriticTrainState.create(
            apply_fn=self.critic.apply,
            params=critic_variables["params"],
            batch_stats=critic_variables["batch_stats"],
            target_params=target_critic_variables["params"],
            target_batch_stats=target_critic_variables["batch_stats"],
            tx=optax.adam(
                learning_rate=self.args["critic_lr"],
                **dict({
                    'b1': self.args["critic_b1"],
                    'b2': 0.999  # default
                }),
            ),
        )

        self.critic.apply = jax.jit(  # type: ignore[method-assign]
            self.critic.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        self.gamma = self.args["gamma"]
        self.polyak = self.args["polyak"]
        self.auto_alpha = self.args["auto_alpha"]
        if self.auto_alpha:
            self.key, alpha_key = jax.random.split(self.key, 2)
            self.log_alpha = EntropyCoef(self.args["alpha_init"])
            alpha_params = self.log_alpha.init(alpha_key, 0.0)['params']
            alpha_optx = optax.adam(self.args["alpha_lr"])
            self.alpha_state = TrainState.create(apply_fn=self.log_alpha.apply, params=alpha_params, tx=alpha_optx)
        else:
            self.alpha = self.args["alpha_init"]

        self.use_policy_active_masks = self.args["use_policy_active_masks"]
        self.use_huber_loss = self.args["use_huber_loss"]
        self.huber_delta = self.args["huber_delta"]
        self.use_sde = False


    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        new_lr = float(self.lr) - float(self.lr) * ((int(step) - 1) / float(steps))
        self.lr = new_lr
        new_tx = optax.adam(learning_rate=new_lr, **dict({'b1': self.args["critic_b1"], 'b2': 0.999}))
        self.critic_state = self.critic_state.replace(tx=new_tx)

    @staticmethod
    @jax.jit
    def soft_update(tau, qf_state):
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        qf_state = qf_state.replace(
            target_batch_stats=optax.incremental_update(qf_state.batch_stats, qf_state.target_batch_stats, tau))

        return qf_state

    @staticmethod
    @partial(jax.jit, static_argnames=[])
    def get_auto_alpha(alpha_state):
        log_alpha = alpha_state.apply_fn({"params": alpha_state.params}, 0)
        log_alpha = jax.lax.stop_gradient(log_alpha)
        return log_alpha

    def get_alpha(self):
        if self.auto_alpha:
            return SoftVectorCritic.get_auto_alpha(self.alpha_state)
        else:
            return self.alpha

    def predict_critic(self, share_obs, actions) -> np.ndarray:
        share_obs = check(share_obs).astype(self.dtype)
        actions = check(actions).astype(self.dtype)
        if not self.use_sde:
            self.key, self.noise_key = jax.random.split(self.key, 2)

        def Q(params, batch_stats, o, a, dropout_key):
            if self.feature_extractor is not None:
                f = self.feature_extractor(o)
            else:
                f = o
            return self.critic_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                f, a,
                rngs={"dropout": dropout_key},
                train=False
            )

        return jax.jit(Q)(
            self.critic_state.params,
            self.critic_state.batch_stats,
            share_obs,
            actions,
            self.noise_key,
        )

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        critic_state_bytes = flax.serialization.to_bytes(self.critic_state)
        critic_path = os.path.join(save_dir, 'critic_state.msgpack')
        with open(critic_path, 'wb') as f:
            f.write(critic_state_bytes)

        print(f"Critic state saved to {save_dir}")

    def restore(self, model_dir):
        critic_path = os.path.join(model_dir, 'critic_state.msgpack')
        with open(critic_path, 'rb') as f:
            critic_state_bytes = f.read()
        restored_critic_state = flax.serialization.from_bytes(self.critic_state, critic_state_bytes)
        self.critic_state = restored_critic_state

        print(f"Critic state restored from {model_dir}")


