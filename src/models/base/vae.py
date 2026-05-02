import os
import jax
import flax
import math
import optax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from datetime import datetime
from functools import partial
from matplotlib import pyplot as plt
from flax.training.train_state import TrainState
from src.utils.envs_tools import get_shape_from_obs_space


class VAE_encoder(nn.Module):
    hidden_dim: int = 64
    latent_dim: int = 10
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        encode = nn.relu(x)
        latent = nn.Dense(self.latent_dim * 4)(encode)
        return latent # mu and log_var

class VAE_decoder(nn.Module):
    output_dim: int
    hidden_dim: int
    @nn.compact
    def __call__(self, latent, decode_in):
        x = jnp.concatenate([latent, decode_in], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class VAE:
    def __init__(self, n_agents, hidden_dim, latent_dim, obs_space, action_dim, encoder_lr, decoder_lr, alpha_1, batch_size, key):
        self.n_agents = n_agents
        # TODO: try different lr
        self.n_agents = n_agents
        self.encoder_lr, self.decoder_lr = encoder_lr, decoder_lr
        self.alpha_1 = alpha_1
        self.now_coeff1 = 1.0
        self.batch_size = batch_size
        self.key = key
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.reward_dim = 1
        self.decoder_in_dim = self.obs_dim + self.action_dim
        self.output_dim = self.obs_dim + self.reward_dim
        self.encoder = VAE_encoder(self.hidden_dim, self.latent_dim)
        self.decoder = VAE_decoder(self.output_dim, self.hidden_dim)
        self.key, encoder_key, decoder_key = jax.random.split(self.key, 3)
        encoder_params = self.encoder.init(encoder_key, jnp.ones((self.batch_size * self.n_agents, self.n_agents)))['params']
        decoder_params = self.decoder.init(decoder_key, jnp.ones((self.batch_size * self.n_agents, self.latent_dim)), jnp.ones((self.batch_size * self.n_agents, self.decoder_in_dim)))['params']
        encoder_optx = optax.adam(self.encoder_lr)
        decoder_optx = optax.adam(self.decoder_lr)
        self.encoder_state = TrainState.create(apply_fn=self.encoder.apply, params=encoder_params, tx=encoder_optx)
        self.decoder_state = TrainState.create(apply_fn=self.decoder.apply, params=decoder_params, tx=decoder_optx)
        

    @staticmethod
    @partial(jax.jit, static_argnames=["latent_dim"])
    def forward(x, encoder_state, now_key, latent_dim):
        latent = encoder_state.apply_fn({"params": encoder_state.params}, x)
        mu, log_var = jnp.clip(latent[:, : latent_dim * 2], -10, 10), latent[:, latent_dim * 2:]
        now_key, eps_key = jax.random.split(now_key, 2)
        eps = jax.random.normal(eps_key, shape=mu.shape)
        latent = mu + eps * jnp.exp(log_var * 0.5)
        return mu, latent, now_key


    @staticmethod
    @partial(jax.jit, static_argnames=["latent_dim"])
    def train_step_wpolicy(agent_ids, obs_actions, next_obs_reward, obs, actions, encoder_state, decoder_state, decoder_state1, key, latent_dim, coeff1):
        def loss_fn(encoder_param, decoder_param, decoder_param1, encoder_state, decoder_state, decoder_state1, now_key):
            latent = encoder_state.apply_fn({"params": encoder_param}, agent_ids)
            mu, log_var = jnp.clip(latent[:, : latent_dim * 2], -10, 10), latent[:, latent_dim * 2:]
            now_key, eps_key = jax.random.split(now_key, 2)
            eps = jax.random.normal(eps_key, shape=mu.shape)
            latent = mu + eps * jnp.exp(log_var * 0.5)
            latent1, latent2 = latent[:, : latent_dim], latent[:, latent_dim:]
            pred_next_obs_reward = decoder_state.apply_fn({"params": decoder_param}, latent1, obs_actions)
            pred_actions = decoder_state1.apply_fn({"params": decoder_param1}, latent2, obs)
            construction_loss = coeff1 * jnp.sum(jnp.sum(jnp.square(pred_next_obs_reward - next_obs_reward), axis=1))
            construction_loss = construction_loss + jnp.sum(jnp.sum(jnp.square(pred_actions - actions), axis=1))
            commitment_loss = -0.5 * jnp.sum(1 + log_var - mu ** 2 - jnp.exp(log_var))
            loss = construction_loss + commitment_loss
            return loss, (construction_loss, commitment_loss, now_key)
        (loss, (construction, commitment_loss, key)), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
            encoder_state.params, decoder_state.params, decoder_state1.params, encoder_state, decoder_state, decoder_state1, key)
        encoder_state = encoder_state.apply_gradients(grads=grads[0])
        decoder_state = decoder_state.apply_gradients(grads=grads[1])
        decoder_state1 = decoder_state1.apply_gradients(grads=grads[2])

        return encoder_state, decoder_state, decoder_state1, construction, commitment_loss, key
    
    @staticmethod
    @partial(jax.jit, static_argnames=["latent_dim"])
    def train_step_wopolicy(agent_ids, obs_actions, next_obs_reward, encoder_state, decoder_state, key, latent_dim):
        def loss_fn(encoder_param, decoder_param, encoder_state, decoder_state, now_key):
            latent = encoder_state.apply_fn({"params": encoder_param}, agent_ids)
            mu, log_var = jnp.clip(latent[:, : latent_dim * 2], -10, 10), latent[:, latent_dim * 2:]
            now_key, eps_key = jax.random.split(now_key, 2)
            eps = jax.random.normal(eps_key, shape=mu.shape)
            latent = mu + eps * jnp.exp(log_var * 0.5)
            latent1, latent2 = latent[:, : latent_dim], latent[:, latent_dim:]
            pred_next_obs_reward = decoder_state.apply_fn({"params": decoder_param}, latent1, obs_actions)
            construction_loss = jnp.sum(jnp.sum(jnp.square(pred_next_obs_reward - next_obs_reward), axis=1))
            commitment_loss = -0.5 * jnp.sum(1 + log_var - mu ** 2 - jnp.exp(log_var))
            loss = construction_loss + commitment_loss
            return loss, (construction_loss, commitment_loss, now_key)
        (loss, (construction, commitment_loss, key)), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))(
            encoder_state.params, decoder_state.params, encoder_state, decoder_state, key)
        encoder_state = encoder_state.apply_gradients(grads=grads[0])
        decoder_state = decoder_state.apply_gradients(grads=grads[1])

        return encoder_state, decoder_state, construction, commitment_loss, key

    def train(self, sp_agent_ids, sp_obs_actions, sp_next_obs_reward, vae_epochs, N, pretrain=True):
        construction_losses, commitment_losses = [], []
        for epoch in range(vae_epochs):
            self.key, permutation_key = jax.random.split(self.key, 2)
            shuffled_indices = jax.random.permutation(permutation_key, N)
            now_agent_ids = sp_agent_ids[shuffled_indices, :]
            now_obs_actions = sp_obs_actions[shuffled_indices, :]
            now_next_obs_reward = sp_next_obs_reward[shuffled_indices, :]
            # now_data = train_x
            tot_construction_loss, tot_commitment_loss = 0, 0
            for i in range(0, N, self.batch_size):
                batch_agent_ids = now_agent_ids[i: i + self.batch_size]
                batch_obs_actions = now_obs_actions[i: i + self.batch_size]
                batch_next_obs_reward = now_next_obs_reward[i: i + self.batch_size]
                if pretrain:
                    self.encoder_state, self.decoder_state, construction_loss, commitment_loss, self.key = VAE.train_step_wopolicy(
                        batch_agent_ids, batch_obs_actions, batch_next_obs_reward, self.encoder_state, self.decoder_state, self.key, self.latent_dim)
                
                tot_construction_loss = tot_construction_loss + construction_loss
                tot_commitment_loss = tot_commitment_loss + commitment_loss
            construction_losses.append(tot_construction_loss / N)
            commitment_losses.append(tot_commitment_loss / N)
            print(f"epoch {epoch + 1}, construction loss: {tot_construction_loss / N}, commitment loss: {tot_commitment_loss / N}.")
        return {"construction_loss": np.array(construction_losses).mean(), 
                "commitment_loss": np.array(commitment_losses).mean()}

    def get_embeddings(self):
        indices = np.arange(self.n_agents)
        agent_onehot = np.eye(self.n_agents)[indices]
        embeddings, latent, self.key = self.forward(agent_onehot, self.encoder_state, self.key, self.latent_dim)
        embeddings = embeddings[:, : self.latent_dim]
        e_min = jnp.min(embeddings, axis=0, keepdims=True)
        e_max = jnp.max(embeddings, axis=0, keepdims=True)
        embeddings = (embeddings - e_min) / (e_max - e_min + 1e-8)
        return jax.lax.stop_gradient(embeddings)
    
    def save(self, save_dir):
        serialized_state = flax.serialization.to_bytes(self.encoder_state)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'encoder.msgpack')

        try:
            with open(file_path, 'wb') as f:
                f.write(serialized_state)
        except IOError as e:
            print(f"  - Error: Save fail: {file_path}, info: {e}")

    def restore(self, model_dir):
        file_path = os.path.join(model_dir, f'encoder.msgpack')
        try:
            with open(file_path, 'rb') as f:
                serialized_state = f.read()
            self.encoder_state = flax.serialization.from_bytes(self.encoder_state, serialized_state)
        except FileNotFoundError:
            print(f"  - Error: File can't found: {file_path}")
        except Exception as e:
            print(f"  - Error: Loading error: {e}")