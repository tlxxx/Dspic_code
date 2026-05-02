"""DiffusionHASAC algorithm."""
import os
# from cv2 import log
import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp
import gymnasium as gym
from functools import partial
from flax.training.train_state import TrainState
from src.models.policy_models.stochastic_mlp_policy import StochasticMlpPolicy
from src.utils.discrete_util import gumbel_softmax
from src.utils.envs_tools import check
from src.models.base.distributions import FixedCategorical
from src.algorithms.actors.off_policy_base import OffPolicyBase
from src.algorithms.actors.diffusion.common.utils import get_sampler_init
from src.algorithms.actors.diffusion.od.od_integrators import get_integrator as get_integrator_od
from src.algorithms.actors.diffusion.od.od_sampling import sample as sample_od
from src.algorithms.actors.diffusion.od.od_sampling_unbound import sample as sample_od_unbound
from typing import Any

class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict

class Dspic(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, batch_size, use_target_network, n_agents, key, role_term=None, latent_dim=10, policy_decoder=False):
        self.args = args
        self.obs_space = obs_space
        self.act_space = act_space
        self.batch_size = batch_size
        self.use_target_network = use_target_network
        self.n_agents = n_agents
        self.key = key
        self.role_term = role_term
        self.latent_dim = latent_dim
        self.policy_decoder = policy_decoder
        self.setup()

    def setup(self):
        self.dtype = jnp.float32
        self.polyak = self.args["polyak"]
        self.lr = self.args["lr"]
        self.action_type = self.act_space.__class__.__name__
        self.key, actor_key = jax.random.split(self.key)
        

        if self.act_space.__class__.__name__ == "Box":
            if self.role_term == "id":
                a_dim, obs_dim = self.act_space.shape[0], self.obs_space.shape[0]
                self.actor_model, self.actor_state, self.encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=self.n_agents)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state, self.target_encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=self.n_agents)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
            elif self.role_term == "vae":
                latent_dim = self.latent_dim
                a_dim, obs_dim = self.act_space.shape[0], self.obs_space.shape[0]
                self.actor_model, self.actor_state, self.encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=latent_dim)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state, self.target_encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=latent_dim)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
            else:
                a_dim, obs_dim = self.act_space.shape[0], self.obs_space.shape[0]
                self.actor_model, self.actor_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)

            self.act_high, self.act_low = self.act_space.high, self.act_space.low

        elif self.act_space.__class__.__name__ == "Discrete":
            if self.role_term == "id":
                a_dim, obs_dim = self.act_space.n, self.obs_space[0] if type(self.obs_space) == list else self.obs_space.shape[0]
                self.actor_model, self.actor_state, self.encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=self.n_agents)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                if self.args["env"] == "lbf":
                    self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                else:
                    self.sampler = partial(sample_od_unbound, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state, self.target_encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=self.n_agents)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    if self.args["env"] == "lbf":
                        self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
                    else:
                        self.target_sampler = partial(sample_od_unbound, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
            elif self.role_term == "vae":
                latent_dim = self.latent_dim
                # a_dim, obs_dim = self.act_space.n, self.obs_space[0] # self.obs_space.shape[0]
                a_dim, obs_dim = self.act_space.n, self.obs_space[0] if type(self.obs_space) == list else self.obs_space.shape[0]
                self.actor_model, self.actor_state, self.encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=latent_dim)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                if self.args["env"] == "lbf":
                    self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                else:
                    self.sampler = partial(sample_od_unbound, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state, self.target_encoder_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim, use_ort=True, latent_dim=latent_dim)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    if self.args["env"] == "lbf":
                        self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
                    else:
                        self.target_sampler = partial(sample_od_unbound, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
            else:
                a_dim, obs_dim = self.act_space.n, self.obs_space[0] if type(self.obs_space) == list else self.obs_space.shape[0]
                self.actor_model, self.actor_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim)
                self.integrator = get_integrator_od(self.args, self.actor_model)
                if self.args["env"] == "lbf":
                    self.sampler = partial(sample_od, integrator=self.integrator, diffusion_model=self.actor_model)
                else:
                    self.sampler = partial(sample_od_unbound, integrator=self.integrator, diffusion_model=self.actor_model)
                if self.use_target_network:
                    self.actor_target_model, self.target_actor_state = get_sampler_init(self.args["sampler_name"])(actor_key, self.args, a_dim, obs_dim)
                    self.target_integrator = get_integrator_od(self.args, self.actor_target_model)
                    if self.args["env"] == "lbf":
                        self.target_sampler = partial(sample_od, integrator=self.target_integrator, diffusion_model=self.actor_target_model)
                    else:
                        self.target_sampler = partial(sample_od_unbound, integrator=self.target_integrator, diffusion_model=self.actor_target_model)

        else:
            assert 0, "bad condition!"
            self.actor = StochasticMlpPolicy(self.args, self.obs_space, self.act_space, device)

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "action_type", "dtype", "stochastic"])
    def get_actions(actor_state, actor_params, observations, P, key, sampler, act_high, act_low, available_actions=None, action_type="Box", dtype=jnp.float32, stochastic=True):
        observations = check(observations).astype(dtype)
        if action_type == "Box":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            actions, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            actions = act_low + (0.5 * (actions + 1.0) * (act_high - act_low))
            return actions, 0
            # actions, _ = self.actor(obs, stochastic=stochastic, with_logprob=False)
        elif action_type == "Discrete":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            logits, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            if available_actions is not None:
                logits = jnp.where(available_actions == 0, -1e16, logits)

            dist = FixedCategorical(logits=logits)
            if stochastic:
                key, sample_key = jax.random.split(key, 2)
                actions = dist.sample(seed=sample_key)
            else:
                actions = dist.mode()
            logp = dist.log_probs(actions)
            return actions, logp
        else:
            # Unimplemented!
            assert 0, "bad condition!"
            actions = self.actor(obs, available_actions, stochastic)
        return actions

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "action_type", "dtype"])
    def get_actions_with_logprobs(actor_state, actor_params, observations, P, key, sampler, act_high, act_low, available_actions=None, action_type="Box", dtype=jnp.float32):
        observations = check(observations).astype(dtype)
        if action_type == "Box":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            actions, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            actions = act_low + (0.5 * (actions + 1.0) * (act_high - act_low))
            return actions, running_costs, stochastic_costs, terminal_costs
        elif action_type == "Discrete":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            logits, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            if available_actions is not None:
                logits = jnp.where(available_actions == 0, -1e16, logits)
            key, sample_key = jax.random.split(key, 2)
            actions = gumbel_softmax(sample_key, logits, hard=True)
            logp_actions = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1, keepdims=True)

            # return actions, running_costs + logp_actions, stochastic_costs, terminal_costs
            return actions, running_costs, logp_actions, terminal_costs
            # return actions, logp_actions
        else:
            # Unimplemented!
            assert 0, "bad condition!"
            actions = self.actor(obs, available_actions, stochastic)
        return actions, running_costs + stochastic_costs + terminal_costs

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "action_type", "dtype", "stochastic"])
    def get_actions_unscale(actor_state, actor_params, observations, P, key, sampler, available_actions=None, action_type="Box", dtype=jnp.float32, stochastic=True):
        observations = check(observations).astype(dtype)
        if action_type == "Box":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            actions, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
        elif action_type == "Discrete":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            logits, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            if available_actions is not None:
                logits = jnp.where(available_actions == 0, -1e16, logits)
            dist = FixedCategorical(logits=logits)
            if stochastic:
                key, sample_key = jax.random.split(key, 2)
                actions = dist.sample(seed=sample_key)
            else:
                actions = dist.mode()
        else:
            # Unimplemented!
            assert 0, "bad condition!"
            actions = self.actor(obs, available_actions, stochastic)
        return actions

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "action_type", "dtype"])
    def get_actions_with_logprobs_unscale(actor_state, actor_params, observations, P, key, sampler, available_actions=None, action_type="Box", dtype=jnp.float32):
        observations = check(observations).astype(dtype)
        if action_type == "Box":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            actions, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            return actions, running_costs, stochastic_costs, terminal_costs
        elif action_type == "Discrete":
            out = sampler(key, actor_state, actor_params, observations, stop_grad=False, P=P)
            logits, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
            if available_actions is not None:
                logits = jnp.where(available_actions == 0, -1e16, logits)
            key, sample_key = jax.random.split(key, 2)
            actions = gumbel_softmax(sample_key, logits, hard=True)
            logp_actions = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1, keepdims=True)
            # return actions, running_costs + logp_actions, stochastic_costs, terminal_costs
            return actions, running_costs, logp_actions, terminal_costs
        else:
            # Unimplemented!
            assert 0, "bad condition!"
            actions = self.actor(obs, available_actions, stochastic)
        return actions, running_costs + stochastic_costs + terminal_costs

    @staticmethod
    @jax.jit
    def get_P(encoder_state, params, latent):
        return encoder_state.apply_fn({"params": params}, latent)
    
    def get_Ps(self, latent):
        return Dspic.get_P(self.encoder_state, self.encoder_state.params, latent)

    @staticmethod
    @jax.jit
    def soft_update_target_actor(tau, actor_state, target_actor_state):
        target_actor_state = target_actor_state.replace(params=optax.incremental_update(actor_state.params, target_actor_state.params, tau))
        return target_actor_state
    
    @staticmethod
    @jax.jit
    def soft_update_target_encoder(tau, encoder_state, target_encoder_state):
        target_encoder_state = target_encoder_state.replace(params=optax.incremental_update(encoder_state.params, target_encoder_state.params, tau))
        return target_encoder_state

    def save(self, save_dir, id):
        """Save the actor."""
        serialized_state = flax.serialization.to_bytes(self.actor_state)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'actor_agent{id}.msgpack')

        try:
            with open(file_path, 'wb') as f:
                f.write(serialized_state)
        except IOError as e:
            print(f"  - Error: Save fail: {file_path}, info: {e}")

        if self.use_target_network:
            serialized_state = flax.serialization.to_bytes(self.target_actor_state)
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f'target_actor_agent{id}.msgpack')

            try:
                with open(file_path, 'wb') as f:
                    f.write(serialized_state)
            except IOError as e:
                print(f"  - Error: Save fail: {file_path}, info: {e}")

        serialized_state = flax.serialization.to_bytes(self.encoder_state)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'encoder{id}.msgpack')

        try:
            with open(file_path, 'wb') as f:
                f.write(serialized_state)
        except IOError as e:
            print(f"  - Error: Save fail: {file_path}, info: {e}")

        if self.use_target_network:
            serialized_state = flax.serialization.to_bytes(self.target_encoder_state)
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f'target_encoder{id}.msgpack')

            try:
                with open(file_path, 'wb') as f:
                    f.write(serialized_state)
            except IOError as e:
                print(f"  - Error: Save fail: {file_path}, info: {e}")


    def restore(self, model_dir, id):
        """Restore the actor."""
        file_path = os.path.join(model_dir, f'actor_agent{id}.msgpack')
        try:
            with open(file_path, 'rb') as f:
                serialized_state = f.read()
            self.actor_state = flax.serialization.from_bytes(self.actor_state, serialized_state)
        except FileNotFoundError:
            print(f"  - Error: File can't found: {file_path}")
        except Exception as e:
            print(f"  - Error: Loading error: {e}")

        if self.use_target_network:
            file_path = os.path.join(model_dir, f'target_actor_agent{id}.msgpack')
            try:
                with open(file_path, 'rb') as f:
                    serialized_state = f.read()
                self.target_actor_state = flax.serialization.from_bytes(self.target_actor_state, serialized_state)
            except FileNotFoundError:
                print(f"  - Error: File can't found: {file_path}")
            except Exception as e:
                print(f"  - Error: Loading error: {e}")

        file_path = os.path.join(model_dir, f'encoder{id}.msgpack')
        try:
            with open(file_path, 'rb') as f:
                serialized_state = f.read()
            self.encoder_state = flax.serialization.from_bytes(self.encoder_state, serialized_state)
        except FileNotFoundError:
            print(f"  - Error: File can't found: {file_path}")
        except Exception as e:
            print(f"  - Error: Loading error: {e}")

        if self.use_target_network:
            file_path = os.path.join(model_dir, f'target_encoder{id}.msgpack')
            try:
                with open(file_path, 'rb') as f:
                    serialized_state = f.read()
                self.target_encoder_state = flax.serialization.from_bytes(self.target_encoder_state, serialized_state)
            except FileNotFoundError:
                print(f"  - Error: File can't found: {file_path}")
            except Exception as e:
                print(f"  - Error: Loading error: {e}")