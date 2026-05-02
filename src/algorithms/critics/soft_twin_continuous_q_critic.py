"""Soft Twin Continuous Q Critic."""
import os
import jax
import optax
import jax.numpy as jnp
import flax
from flax import linen as nn
import gymnasium as gym
from flax.training.train_state import TrainState
from src.models.value_function_models.continuous_q_net import ContinuousQNet
from src.utils.envs_tools import check, get_shape_from_obs_space
from typing import Sequence, Dict, Any

class RLCriticTrainState(TrainState): 
    target_params: flax.core.FrozenDict

class VectorCritic(nn.Module):
    args: Any
    share_obs_space: Any
    act_space: Any
    n_critics: int = 2
    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            ContinuousQNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )

        q_values = vmap_critic(
            args=self.args,
            cent_obs_space=self.share_obs_space,
            act_spaces=self.act_space,
        )(obs, action)
        return q_values


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)

class SoftTwinContinuousQCritic:
    """Soft Twin Continuous Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    Note that the name SoftTwinContinuousQCritic emphasizes its structure that takes observations and actions as input
    and outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be
    used in discrete action space.
    """

    def __init__(self, args, share_obs_space, act_space, num_agents, state_type, batch_size, key):
        self.args = args
        self.share_obs_space = share_obs_space
        self.act_space = act_space
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.state_type = state_type
        self.key = key
        self.dtype = jnp.float32
        self.action_type = self.act_space[0].__class__.__name__
        self.gamma = self.args["gamma"]
        self.critic_lr = self.args["critic_lr"]
        self.polyak = self.args["polyak"]
        self.auto_alpha = self.args["auto_alpha"]
        self.critic = VectorCritic(self.args, self.share_obs_space, self.act_space)
        cent_obs_shape = get_shape_from_obs_space(self.share_obs_space)
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
        
        self.key, critic_key= jax.random.split(self.key, 2)
        dummy_obs = jnp.zeros((self.batch_size, obs_dim), jnp.float32)
        dummy_action = jnp.zeros((self.batch_size, actions_dim), jnp.int32)
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)['params']
        target_critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)['params']
        critic_optx = optax.adam(self.critic_lr)
        self.lr = self.critic_lr
        self.critic_state = RLCriticTrainState.create(apply_fn=self.critic.apply, params=critic_params, target_params=target_critic_params, tx=critic_optx)
        if self.auto_alpha:
            self.key, alpha_key = jax.random.split(self.key, 2)
            self.log_alpha = EntropyCoef(self.args["alpha_init"])
            alpha_params = self.log_alpha.init(alpha_key, 0.0)['params']
            alpha_optx = optax.chain(
                optax.clip_by_global_norm(10.0),                
                optax.adam(self.args["alpha_lr"]) 
            )
            self.alpha_state = TrainState.create(apply_fn=self.log_alpha.apply, params=alpha_params, tx=alpha_optx)
           
        else:
            self.alpha = self.args["alpha_init"]

        self.use_policy_active_masks = self.args["use_policy_active_masks"]
        self.use_huber_loss = self.args["use_huber_loss"]
        self.huber_delta = self.args["huber_delta"]
        self.use_proper_time_limits = self.args["use_proper_time_limits"]
        self.use_sde = False

    @staticmethod
    @jax.jit
    def get_q_values(critic_params, critic_params2, critic_state, critic_state2, obs, actions):
        q = critic_state.apply_fn({"params": critic_params}, obs, actions)
        q1, q2 = q[0], q[1]
        return jnp.minimum(q1, q2)

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        new_lr = float(self.lr) - float(self.lr) * ((int(step) - 1) / float(steps))
        new_tx = optax.adam(learning_rate=new_lr)
        self.critic_state = self.critic_state.replace(tx=new_tx)

    @staticmethod
    @jax.jit
    def soft_update(tau, qf_state):
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state
    
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