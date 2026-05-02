# import torch
# import numpy as np
# import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from src.utils.envs_tools import check
from src.runners.off_policy_base_runner import OffPolicyBaseRunner
from src.algorithms.actors.dspic import Dspic
from src.algorithms.actors.diffusion.common.learning_rate_scheduler import get_learning_rate_scheduler


def merge_dict(d1, d2):
    return d1 | d2


class DspicRunner(OffPolicyBaseRunner):

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data

        if self.action_spaces[0].__class__.__name__ == "Discrete":
            lim_a = self.action_spaces[0].n
            squeezed_actions = jnp.squeeze(sp_actions, axis=-1)
            sp_actions = jax.nn.one_hot(squeezed_actions, num_classes=lim_a)

        if self.use_id_concat:
            if self.Orthogonal:
                n_agents, batch_size, dim = sp_obs.shape
                agent_embedding_expanded = np.eye(n_agents)
            else:
                n_agents, batch_size, dim = sp_obs.shape
                agent_ids = np.eye(n_agents)
                agent_ids_expanded = np.tile(agent_ids[:, np.newaxis, :], (1, batch_size, 1))
                sp_obs = np.concatenate([sp_obs, agent_ids_expanded], axis=-1)
                sp_next_obs = np.concatenate([sp_next_obs, agent_ids_expanded], axis=-1)

        elif self.use_vae:
            if self.Orthogonal:
                n_agents, batch_size, dim = sp_obs.shape
                agent_embedding_expanded = self.agent_embeddings
            else:
                n_agents, batch_size, dim = sp_obs.shape
                agent_embedding_expanded = np.tile(self.agent_embeddings.reshape(n_agents, 1, -1), (1, batch_size, 1))
                sp_obs = np.concatenate([sp_obs, agent_embedding_expanded], axis=-1)
                sp_next_obs = np.concatenate([sp_next_obs, agent_embedding_expanded], axis=-1)

        if self.args["algo"] == "dspic":
            next_actions = []
            rcs_, scs_, tcs_ = [], [], []
            if self.Orthogonal:

                P = Dspic.get_P(self.actor[0].target_encoder_state,
                                                   self.actor[0].target_encoder_state.params, agent_embedding_expanded)
                P = jax.lax.stop_gradient(P)
                P = jnp.tile(P.reshape(n_agents, 1, self.algo_args["model"]["latent_dim"],
                                       self.algo_args["model"]["latent_dim"]), (1, batch_size, 1, 1))
            for agent_id in range(self.num_agents):
                if self.Orthogonal:
                    self.key, actor_key = jax.random.split(self.key, 2)
                    next_action, rc, sc, tc = Dspic.get_actions_with_logprobs_unscale(
                        self.actor[agent_id].target_actor_state, self.actor[agent_id].target_actor_state.params,
                        sp_next_obs[agent_id], P[agent_id], actor_key,
                        self.actor[agent_id].target_sampler, available_actions=sp_next_available_actions[
                            agent_id] if sp_next_available_actions is not None else None,
                        action_type=self.actor[agent_id].action_type, dtype=jnp.float32)
                else:
                    self.key, actor_key = jax.random.split(self.key, 2)
                    # next_action, rc, sc, tc = DiffusionHASAC.get_actions_with_logprobs_unscale(
                    #     self.actor[agent_id].target_actor_state, self.actor[agent_id].target_actor_state.params,
                    #     sp_next_obs[agent_id], actor_key,
                    #     self.actor[agent_id].target_sampler, available_actions=sp_next_available_actions[
                    #         agent_id] if sp_next_available_actions is not None else None,
                    #     action_type=self.actor[agent_id].action_type, dtype=jnp.float32)
                    pass
                
                next_action, rc, sc, tc = jax.lax.stop_gradient(next_action), jax.lax.stop_gradient(
                    rc), jax.lax.stop_gradient(sc), jax.lax.stop_gradient(tc)
                if self.args["env"] == "lbf":
                    sc = sc + rc + tc
                next_actions.append(next_action)
                rcs_.append(rc)
                scs_.append(sc)
                tcs_.append(tc)
            entr_coeff = self.algo_args["model"]["entr_coeff"]
            metrics = {}
            if self.critic.action_type == "Box":
                act_lows = jnp.stack([jnp.asarray(self.actor[i].act_low) for i in range(self.num_agents)], axis=0)
                act_highs = jnp.stack([jnp.asarray(self.actor[i].act_high) for i in range(self.num_agents)], axis=0)
                act_highs, act_lows = act_highs.reshape(self.num_agents, 1, -1), act_lows.reshape(self.num_agents, 1,
                                                                                                  -1)
                sp_actions_unscale = 2.0 * ((sp_actions - act_lows) / (act_highs - act_lows)) - 1.0
            else:
                sp_actions_unscale = sp_actions

            self.critic.critic_state, critic_metrics, self.key = DspicRunner.train_critic(
                sp_share_obs,
                sp_actions_unscale,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                rcs_,
                scs_,
                tcs_,
                sp_gamma,
                self.use_bnstats_from_live_net,
                self.crossq_style,
                self.num_atoms,
                self.v_min,
                self.v_max,
                entr_coeff,
                self.critic.dtype,
                self.num_agents,
                self.algo_args["algo"]["auto_alpha"],
                self.critic.use_proper_time_limits,
                self.critic.action_type,
                self.state_type,
                self.critic.critic_state,
                self.critic.alpha_state if self.algo_args["algo"]["auto_alpha"] else self.critic.alpha,
                self.key,
            )
            metrics = merge_dict(metrics, critic_metrics)

        else:
            # Unimplemented!
            assert 0
            next_actions = []


        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "dspic":
                actions = []
                rcs, scs, tcs = [], [], []
                for agent_id in range(self.num_agents):
                    if self.Orthogonal:
                        self.key, actor_key = jax.random.split(self.key, 2)
                        action, rc, sc, tc = Dspic.get_actions_with_logprobs_unscale(
                            self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, sp_obs[agent_id],
                            P[agent_id], actor_key,
                            self.actor[agent_id].sampler, available_actions=sp_available_actions[
                                agent_id] if sp_available_actions is not None else None,
                            action_type=self.actor[agent_id].action_type,
                            dtype=jnp.float32)
                    else:
                        self.key, actor_key = jax.random.split(self.key, 2)
                        # action, rc, sc, tc = DiffusionHASAC.get_actions_with_logprobs_unscale(
                        #     self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, sp_obs[agent_id],
                        #     actor_key,
                        #     self.actor[agent_id].sampler, available_actions=sp_available_actions[
                        #         agent_id] if sp_available_actions is not None else None,
                        #     action_type=self.actor[agent_id].action_type,
                        #     dtype=jnp.float32)
                        pass

                    action, rc, sc, tc = jax.lax.stop_gradient(action), jax.lax.stop_gradient(
                        rc), jax.lax.stop_gradient(sc), jax.lax.stop_gradient(tc)
                    if self.args["env"] == "lbf":
                        sc = sc + rc + tc
                    actions.append(action)
                    rcs.append(rc)
                    scs.append(sc)
                    tcs.append(tc)

                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    self.key, permutation_key = jax.random.split(self.key)
                    agent_order = jax.random.permutation(permutation_key, self.num_agents)

                actor_losses = []
                alpha_losses = []
                dif_alpha_losses = []
                orthogonal_losses = []
                q_takens = []
                min_dists = []
                actor_ent_coefs = []
                rc_means = []
                sc_means = []
                tc_means = []
                ent_coefs = []
                actions_arr = jnp.stack(actions, axis=0)
                rc_arr, sc_arr, tc_arr = jnp.stack(rcs, axis=0), jnp.stack(scs, axis=0), jnp.stack(tcs, axis=0)

                for agent_id in agent_order:
                    if self.Orthogonal:
                        self.actor[agent_id].actor_state, self.actor[
                            agent_id].encoder_state, actor_metrics, self.key, actions_arr, rc_arr, sc_arr, tc_arr, ent_coef_value = DspicRunner.train_actor_encoder(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id] if sp_available_actions is not None else None,
                            sp_share_obs,
                            sp_valid_transition[agent_id],
                            rc_arr,
                            sc_arr,
                            tc_arr,
                            actions_arr,
                            self.alpha_states[agent_id] if self.algo_args["algo"]["auto_alpha"] else self.alpha[
                                agent_id],
                            self.critic.critic_state,
                            self.actor[agent_id].actor_state,
                            self.actor[agent_id].sampler,
                            self.actor[agent_id].encoder_state,
                            self.state_type,
                            self.crossq_style,
                            self.algo_args["algo"]["use_policy_active_masks"],
                            self.algo_args["algo"]["auto_alpha"],
                            jnp.asarray(agent_id, jnp.int32).reshape(()),
                            self.actor[agent_id].action_type,
                            self.total_it,
                            self.num_agents,
                            self.num_atoms,
                            self.v_min,
                            self.v_max,
                            self.key,
                            self.args["env"],
                            latent=agent_embedding_expanded,
                        )
                        orthogonal_losses.append(actor_metrics["orthogonal_loss"])
                        q_takens.append(actor_metrics["q_taken_mean"])
                        actor_ent_coefs.append(actor_metrics["actor_ent_coef"])

                    else:
                        # self.actor[
                        #     agent_id].actor_state, actor_metrics, self.key, actions_arr, rc_arr, sc_arr, tc_arr, ent_coef_value = DspicRunner.train_actor(
                        #     sp_obs[agent_id],
                        #     sp_available_actions[agent_id] if sp_available_actions is not None else None,
                        #     sp_share_obs,
                        #     sp_valid_transition[agent_id],
                        #     rc_arr,
                        #     sc_arr,
                        #     tc_arr,
                        #     actions_arr,
                        #     self.alpha_states[agent_id] if self.algo_args["algo"]["auto_alpha"] else self.alpha[
                        #         agent_id],
                        #     self.critic.critic_state,
                        #     self.actor[agent_id].actor_state,
                        #     self.actor[agent_id].sampler,
                        #     self.state_type,
                        #     self.crossq_style,
                        #     self.algo_args["algo"]["use_policy_active_masks"],
                        #     self.algo_args["algo"]["auto_alpha"],
                        #     jnp.asarray(agent_id, jnp.int32).reshape(()),
                        #     self.actor[agent_id].action_type,
                        #     self.num_agents,
                        #     self.num_atoms,
                        #     self.v_min,
                        #     self.v_max,
                        #     self.key,
                        #     self.args["env"],
                        # )
                        pass

                    actor_losses.append(actor_metrics["actor_loss"])
                    rc_means.append(actor_metrics["run_costs"])
                    sc_means.append(actor_metrics["sto_costs"])
                    tc_means.append(actor_metrics["terminal_costs"])
                    ent_coefs.append(ent_coef_value)
                    min_dists.append(actor_metrics["min_dist"])

                    if self.algo_args["algo"]["auto_alpha"]:
                        logp_arr = rc_arr + tc_arr + sc_arr
                        if self.critic.action_type == "Discrete":
                            self.alpha_states[
                                agent_id], actor_alpha_metrics = DspicRunner.train_actor_alpha(
                                sc_arr,
                                self.alpha_states[agent_id],
                                agent_id,
                                self.target_entropy[agent_id],
                                self.loss_low,
                            )
                        else:
                            self.alpha_states[
                                agent_id], actor_alpha_metrics = DspicRunner.train_actor_alpha(
                                logp_arr,
                                self.alpha_states[agent_id],
                                agent_id,
                                self.target_entropy[agent_id],
                                self.loss_low,
                            )

                        alpha_losses.append(actor_alpha_metrics["actor_alpha_loss"])

                if self.Orthogonal:
                    metrics = merge_dict(metrics, {"orthogonal_loss": np.array(orthogonal_losses).mean()})
                    metrics = merge_dict(metrics, {"q_taken_mean": np.array(q_takens).mean()})
                    metrics = merge_dict(metrics, {"actor_ent_coef": np.array(actor_ent_coefs).mean()})

                metrics = merge_dict(metrics, {"actor_loss": np.array(actor_losses).mean(),
                                               "actor_alpha_loss": np.array(alpha_losses).mean() if len(
                                                   alpha_losses) > 0 else 0,
                                               "dif_actor_alpha_loss": np.array(dif_alpha_losses).mean() if len(
                                                   dif_alpha_losses) > 0 else 0})
                metrics = merge_dict(metrics,
                                     {"run_costs": np.array(rc_means).mean(), "sto_costs": np.array(sc_means).mean(),
                                      "terminal_costs": np.array(tc_means).mean()})
                metrics = merge_dict(metrics, {"min_dist_taken": np.array(min_dists).mean()})
                metrics = merge_dict(metrics, {"actor_ent_coef": np.array(ent_coefs).mean(), "actor_dif_ent_coef": 0})
                actor_lr = get_learning_rate_scheduler(self.algo_args["model"], self.algo_args["model"]["lr"])(
                    self.actor[agent_id].actor_state.step)
                critic_lr = self.critic.lr
                metrics = merge_dict(metrics, {"actor_lr": actor_lr, "critic_lr": critic_lr})
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    logps = rcs_ + scs_ + tcs_
                    if self.critic.action_type == "Discrete":
                        merged_next_logp_actions = jnp.sum(jnp.concatenate(scs_, axis=-1), axis=-1,
                                                           keepdims=True).astype(self.critic.dtype)
                    else:
                        merged_next_logp_actions = jnp.sum(jnp.concatenate(logps, axis=-1), axis=-1,
                                                           keepdims=True).astype(self.critic.dtype)

                    self.critic.alpha_state, critic_alpha_metrics = DspicRunner.train_critic_alpha(
                        merged_next_logp_actions, np.sum(np.array(self.target_entropy)), self.critic.alpha_state,
                        self.loss_low
                    )
                    metrics = merge_dict(metrics, critic_alpha_metrics)

                if self.Orthogonal:
                    self.actor[agent_id].target_actor_state = Dspic.soft_update_target_actor(
                        self.actor[agent_id].polyak, self.actor[agent_id].actor_state,
                        self.actor[agent_id].target_actor_state)
                    self.actor[agent_id].target_encoder_state = Dspic.soft_update_target_encoder(
                        self.actor[agent_id].polyak, self.actor[agent_id].encoder_state,
                        self.actor[agent_id].target_encoder_state)
                else:
                    # self.actor[agent_id].target_actor_state = DiffusionHASAC.soft_update_target_actor(
                    #     self.actor[agent_id].polyak, self.actor[agent_id].actor_state,
                    #     self.actor[agent_id].target_actor_state)
                    pass

            else:
                assert 0
            if self.args["algo"] == "dspic":
                if self.crossq_style:
                    from src.algorithms.critics.VectorCritic import SoftVectorCritic
                    self.critic.critic_state = SoftVectorCritic.soft_update(self.algo_args["algo"]["polyak"],
                                                                            self.critic.critic_state)
                else:
                    from src.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
                    self.critic.critic_state = SoftTwinContinuousQCritic.soft_update(self.algo_args["algo"]["polyak"],
                                                                                     self.critic.critic_state)
            else:
                # Unimplemented!
                assert 0
                self.critic.soft_update()

        return metrics

    @staticmethod
    @partial(jax.jit, static_argnames=["use_bnstats_from_live_net", "crossq_style", "num_atoms", "v_min", "v_max",
                                       "dtype", "num_agents", "auto_alpha", "use_proper_time_limits", "action_type",
                                       "state_type"])
    def train_critic(sp_share_obs,
                     sp_actions,
                     sp_reward,
                     sp_done,
                     sp_valid_transition,
                     sp_term,
                     sp_next_share_obs,
                     next_actions,
                     rcs,
                     scs,
                     tcs,
                     sp_gamma,
                     use_bnstats_from_live_net,
                     crossq_style,
                     num_atoms,
                     v_min,
                     v_max,
                     entr_coeff,
                     dtype,
                     num_agents,
                     auto_alpha,
                     use_proper_time_limits,
                     action_type,
                     state_type,
                     critic_state,
                     alpha,
                     key):
        sp_reward = sp_reward.reshape(-1)
        sp_done = sp_done.reshape(-1)
        sp_term = sp_term.reshape(-1)
        sp_gamma = sp_gamma.reshape(-1)
        z_atoms = jnp.linspace(v_min, v_max, num_atoms)
        share_obs = check(sp_share_obs).astype(dtype)
        if action_type == "Box":
            actions = check(sp_actions).astype(dtype)
            actions = jnp.concatenate([actions[i] for i in range(actions.shape[0])], axis=-1)
        elif action_type == "Discrete":
            actions = check(sp_actions).astype(dtype)
            actions = jnp.concatenate([actions[i] for i in range(actions.shape[0])], axis=-1)
        else:
            assert 0, "bad condition!"

        if state_type == "FP":
            actions = jnp.tile(actions, (num_agents, 1))
        reward = check(sp_reward).astype(dtype)
        done = check(sp_done).astype(dtype)
        valid_transition = check(sp_valid_transition).astype(dtype)

        term = check(sp_term).astype(dtype)
        gamma = check(sp_gamma).astype(dtype)
        next_share_obs = check(sp_next_share_obs).astype(dtype)
        if action_type == "Box":
            next_actions = jnp.concatenate(next_actions, axis=-1).astype(dtype)
        elif action_type == "Discrete":
            next_actions = jnp.concatenate(next_actions, axis=-1).astype(dtype)
        else:
            assert 0, "bad condition!"
        rcs = jnp.sum(jnp.concatenate(rcs, axis=-1), axis=-1).astype(dtype)
        scs = jnp.sum(jnp.concatenate(scs, axis=-1), axis=-1).astype(dtype)
        tcs = jnp.sum(jnp.concatenate(tcs, axis=-1), axis=-1).astype(dtype)
        if state_type == "FP":
            next_actions = jnp.tile(next_actions, (num_agents, 1))
            rcs = jnp.tile(rcs, (num_agents))
            scs = jnp.tile(scs, (num_agents))
            tcs = jnp.tile(tcs, (num_agents))
        key, noise_key, dropout_key_target, dropout_key_current, redq_key = jax.random.split(key, 5)
        if auto_alpha:
            ent_coef_value = alpha.apply_fn({"params": alpha.params}, 0)
            ent_coef_value = jax.lax.stop_gradient(ent_coef_value)
            if action_type == "Discrete":
                dif_ent_coef_value = 0
            else:
                dif_ent_coef_value = ent_coef_value
        else:
            ent_coef_value = alpha
            if action_type == "Discrete":
                dif_ent_coef_value = 0
            else:
                dif_ent_coef_value = ent_coef_value

        def loss_fn(params, batch_stats, target_params, dropout_key):
            if not crossq_style:
                if not crossq_style:
                    next_q_values = critic_state.apply_fn({"params": target_params}, next_share_obs, next_actions)
                    current_q_values = critic_state.apply_fn({"params": params}, share_obs, actions)

            else:
                # ----- CrossQ's One Weird Trick™ -----
                # concatenate current and next observations to double the batch size
                # new shape of input is (n_critics, 2*batch_size, obs_dim + act_dim)
                # apply critic to this bigger batch
                catted_q_values, state_updates = critic_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([share_obs, next_share_obs], axis=0),
                    jnp.concatenate([actions, next_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

            if next_q_values.shape[0] > 2:  # only for REDQ
                # REDQ style subsampling of critics.
                m_critics = 2
                # next_q_values = jax.random.choice(redq_key, next_q_values, (m_critics,), replace=False, axis=0)
                idx = jax.random.choice(redq_key, next_q_values.shape[0], (m_critics,), replace=False)
                next_q_values = next_q_values[idx]

            next_q_values_q1 = next_q_values[0]
            next_q_values_q2 = next_q_values[1]

            current_q1 = current_q_values[0]
            current_q2 = current_q_values[1]

            def projection(next_dist, rewards, dones, gamma, v_min, v_max, num_atoms, support):
                # print(next_dist.shape, rewards.shape, dones.shape, gamma, v_min, v_max, num_atoms, support.shape)
                # exit(0)
                delta_z = (v_max - v_min) / (num_atoms - 1)
                batch_size = rewards.shape[0]

                # print(next_logp_actions.shape)
                # exit(0)
                # entr_bon = - (1 - dones[:, None]) * gamma * ent_coef_value * next_logp_actions
                # target_z = jnp.clip(rewards[:,None] + entr_bon + (1 - dones[:, None]) * gamma * support, min=v_min, max=v_max)

                entr_bon = - (1.0 - dones) * gamma * (ent_coef_value * scs + dif_ent_coef_value * (rcs + tcs))  # (B,)
                # entr_bon = - (1.0 - dones) * gamma * (ent_coef_value * (scs + rcs + tcs))  # (B,)
                target_z = jnp.clip(
                    rewards[:, None] + entr_bon[:, None] + (1.0 - dones[:, None]) * gamma[:, None] * support, min=v_min,
                    max=v_max)

                b = (target_z - v_min) / delta_z
                l = jnp.floor(b).astype(jnp.int32)
                u = jnp.ceil(b).astype(jnp.int32)

                # Adjust l and u to ensure they remain within valid bounds
                l = jnp.where((u > 0) & (l == u), l - 1, l)
                u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)

                # Create the projected distribution
                proj_dist = jnp.zeros_like(next_dist)

                # Offset calculation for batch indexing
                offset = jnp.arange(batch_size)[:, None] * num_atoms
                # offset = jnp.tile(offset, (1, num_atoms))  # Repeat along the second axis

                # Index updates for proj_dist
                l_idx = (l + offset).ravel()
                u_idx = (u + offset).ravel()

                # Flattened updates
                l_update = (next_dist * (u.astype(jnp.float32) - b)).ravel()
                u_update = (next_dist * (b - l.astype(jnp.float32))).ravel()

                # Flatten proj_dist for updates
                proj_dist_flat = proj_dist.ravel()

                # Add values to proj_dist
                proj_dist_flat = proj_dist_flat.at[l_idx].add(l_update)
                proj_dist_flat = proj_dist_flat.at[u_idx].add(u_update)

                # Reshape back to [batch_size, num_atoms]
                proj_dist = proj_dist_flat.reshape(batch_size, num_atoms)

                return proj_dist

            if crossq_style:
                if use_proper_time_limits:
                    target_q1_projected = projection(next_dist=next_q_values_q1, rewards=reward, dones=term,
                                                     gamma=gamma,
                                                     v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)
                    target_q2_projected = projection(next_dist=next_q_values_q2, rewards=reward, dones=term,
                                                     gamma=gamma,
                                                     v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)

                else:
                    target_q1_projected = projection(next_dist=next_q_values_q1, rewards=reward, dones=done,
                                                     gamma=gamma,
                                                     v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)
                    target_q2_projected = projection(next_dist=next_q_values_q2, rewards=reward, dones=done,
                                                     gamma=gamma,
                                                     v_min=v_min, v_max=v_max, num_atoms=num_atoms, support=z_atoms)

                next_q_values = jax.lax.stop_gradient(
                    jnp.min(jnp.stack([target_q1_projected, target_q2_projected], axis=0), axis=0))

                def binary_cross_entropy(pred, target, valid_transition):
                    if state_type == "FP":
                        mask = valid_transition.reshape(-1)  # (batch, 1)
                        return (-jnp.sum(jnp.sum(target * jnp.log(pred + 1e-15), axis=-1) * mask) / (
                                    mask.sum() + 1e-8) + entr_coeff * jnp.sum(
                            jnp.sum(pred * jnp.log(pred + 1e-15), axis=-1) * mask) / (
                                        mask.sum() + 1e-8))
                    else:
                        # return (-jnp.sum(jnp.mean(target * jnp.log(pred + 1e-15), axis=-1)) + entr_coeff * jnp.sum(jnp.mean(pred * jnp.log(pred + 1e-15), axis=-1)))
                        return (-jnp.mean(jnp.sum(target * jnp.log(pred + 1e-15), axis=-1)) + entr_coeff * jnp.mean(
                            jnp.sum(pred * jnp.log(pred + 1e-15), axis=-1)))

                loss = binary_cross_entropy(current_q1, next_q_values, valid_transition) + binary_cross_entropy(
                    current_q2, next_q_values, valid_transition)
                qf_pi1 = jnp.sum(current_q1 * z_atoms, axis=-1)
                qf_pi2 = jnp.sum(current_q2 * z_atoms, axis=-1)
                entr_1 = -jnp.mean(jnp.sum(current_q1 * jnp.log(current_q1 + 1e-15), axis=-1))
                entr_2 = -jnp.mean(jnp.sum(current_q2 * jnp.log(current_q2 + 1e-15), axis=-1))
                
                min_qf_pi = jax.lax.stop_gradient(jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze())
                return loss, (state_updates, min_qf_pi, next_q_values, entr_1, entr_2)
            else:
                next_q_values = jax.lax.stop_gradient(jnp.minimum(next_q_values_q1, next_q_values_q2).squeeze())
                if use_proper_time_limits:
                    q_targets = reward + gamma * (
                                next_q_values - (ent_coef_value * scs + dif_ent_coef_value * (rcs + tcs))) * (1 - term)
                else:
                    q_targets = reward + gamma * (
                                next_q_values - (ent_coef_value * scs + dif_ent_coef_value * (rcs + tcs))) * (1 - done)
                q_targets = jax.lax.stop_gradient(q_targets)
                min_qf_pi = jax.lax.stop_gradient(
                    jnp.min(jnp.stack([current_q1, current_q2], axis=0), axis=0).squeeze())

                if state_type == "FP":
                    mask = valid_transition.reshape(-1)
                    critic_loss1 = jnp.sum(jnp.square(current_q1.squeeze() - q_targets) * mask) / (mask.sum() + 1e-8)
                    critic_loss2 = jnp.sum(jnp.square(current_q2.squeeze() - q_targets) * mask) / (mask.sum() + 1e-8)
                else:
                    critic_loss1 = jnp.mean(jnp.square(current_q1.squeeze() - q_targets))
                    critic_loss2 = jnp.mean(jnp.square(current_q2.squeeze() - q_targets))

                loss = critic_loss1 + critic_loss2
                return (loss, (None, min_qf_pi, next_q_values, critic_loss1, critic_loss2))

        (qf_loss_value, (state_updates, current_q_values, next_q_values, entr_1, entr_2)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params,
                                                      critic_state.batch_stats if crossq_style else None,
                                                      critic_state.target_params, dropout_key_current)

        critic_state = critic_state.apply_gradients(grads=grads)
        if crossq_style:
            critic_state = critic_state.replace(batch_stats=state_updates["batch_stats"])

        metrics = {
            'critic_loss': qf_loss_value,
            'ent_coef': ent_coef_value,
            'dif_ent_coef': 0,  
            'current_q_values': current_q_values.mean(),
            'next_q_values': next_q_values.mean(),
            'entrQ_1': entr_1,
            'entrQ_2': entr_2,
        }
        return critic_state, metrics, key

    @staticmethod
    @partial(jax.jit, static_argnames=["target_entropy", "loss_low"])
    def train_critic_alpha(next_logp_actions, target_entropy, alpha_state, loss_low):
        next_logp_actions = next_logp_actions + target_entropy
        next_logp_actions = jax.lax.stop_gradient(next_logp_actions)

        def loss_fn(params):
            log_alpha = alpha_state.apply_fn({"params": params}, 0)
            raw_loss = -(log_alpha * next_logp_actions.mean())
            alpha_loss = jnp.maximum(loss_low, raw_loss)
            return alpha_loss

        alpha_loss, grads = jax.value_and_grad(loss_fn)(alpha_state.params)
        alpha_state = alpha_state.apply_gradients(grads=grads)

        return alpha_state, {"critic_alpha_loss": alpha_loss}

    # @staticmethod
    # @partial(jax.jit, static_argnames=["sampler", "state_type", "crossq_style", "use_policy_active_masks", "auto_alpha",
    #                                    "action_type",
    #                                    "num_agents", "num_atoms", "v_min", "v_max", "env_name"])
    # def train_actor(
    #         sp_obs,
    #         sp_available_actions,
    #         sp_share_obs,
    #         sp_valid_transition,
    #         rc_arr, sc_arr, tc_arr,
    #         actions,
    #         alpha_state,
    #         critic_state,
    #         actor_state,
    #         sampler,
    #         state_type,
    #         crossq_style,
    #         use_policy_active_masks,
    #         auto_alpha,
    #         agent_id,
    #         action_type,
    #         num_agents,
    #         num_atoms,
    #         v_min,
    #         v_max,
    #         key,
    #         env_name,
    # ):
    #     z_atoms = jnp.linspace(v_min, v_max, num_atoms)

    #     def actor_loss_fn(params, actor_state, agent_id, actor_key, dropout_key):
    #         act_i, rc, sc, tc = DiffusionHASAC.get_actions_with_logprobs_unscale(actor_state, params, sp_obs, actor_key,
    #                                                                              sampler, available_actions=sp_available_actions,
    #                                                                              action_type=action_type)
    #         if env_name == "lbf":
    #             sc = sc + rc + tc
    #         actions_up = actions.at[agent_id].set(act_i)
    #         rc_up = rc_arr.at[agent_id].set(rc)
    #         sc_up = sc_arr.at[agent_id].set(sc)
    #         tc_up = tc_arr.at[agent_id].set(tc)

    #         if state_type == "EP":
    #             rc_ = rc_up[agent_id]
    #             sc_ = sc_up[agent_id]
    #             tc_ = tc_up[agent_id]
    #             actions_t = jnp.concatenate(actions_up, axis=-1)
    #         elif state_type == "FP":
    #             rc_ = jnp.tile(rc_up[agent_id], (num_agents, 1))
    #             sc_ = jnp.tile(sc_up[agent_id], (num_agents, 1))
    #             tc_ = jnp.tile(tc_up[agent_id], (num_agents, 1))
    #             actions_t = jnp.tile(jnp.concatenate(actions_up, axis=-1), (num_agents, 1))
    #         else:
    #             # Unimplemented!
    #             assert 0

    #         if crossq_style:
    #             qf_pi = critic_state.apply_fn(
    #                 {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
    #                 sp_share_obs, actions_t,
    #                 rngs={"dropout": dropout_key},
    #                 train=False,
    #             )

    #             qf_pi1 = jnp.sum(qf_pi[0] * z_atoms, axis=-1)
    #             qf_pi2 = jnp.sum(qf_pi[1] * z_atoms, axis=-1)
    #         else:
    #             qf_pi = critic_state.apply_fn({"params": critic_state.params}, sp_share_obs, actions_t)
    #             qf_pi1, qf_pi2 = qf_pi[0], qf_pi[1]

    #         reduced_qf_pi = jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
    #         if auto_alpha:
    #             ent_coef_value = alpha_state.apply_fn({"params": alpha_state.params}, 0)
    #             ent_coef_value = jax.lax.stop_gradient(ent_coef_value)
    #             if action_type == "Discrete":
    #                 dif_ent_coef_value = 0
    #             else:
    #                 dif_ent_coef_value = ent_coef_value
    #         else:
    #             ent_coef_value = alpha_state
    #             if action_type == "Discrete":
    #                 dif_ent_coef_value = 0
    #             else:
    #                 dif_ent_coef_value = ent_coef_value
    #         if use_policy_active_masks:
    #             if state_type == "EP":
    #                 actor_loss = jnp.sum((-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (
    #                             rc_ + tc_)).squeeze()) * sp_valid_transition.squeeze()) / (
    #                                          sp_valid_transition.sum() + 1e-8)
    #             elif state_type == "FP":
    #                 valid_transition = jnp.tile(sp_valid_transition, (num_agents, 1))
    #                 actor_loss = jnp.sum((-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (
    #                             rc_ + tc_)).squeeze()) * valid_transition.squeeze()) / (valid_transition.sum() + 1e-8)
    #         else:
    #             actor_loss = jnp.mean(
    #                 (-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (rc_ + tc_)).squeeze()))

    #         return actor_loss, (actions_up, rc_up, sc_up, tc_up, rc_.mean(), sc_.mean(), tc_.mean(), ent_coef_value)

    #     key, actor_key, dropout_key = jax.random.split(key, 3)
    #     (actor_loss,
    #      (actions_upd, rc_up, sc_up, tc_up, rc_mean, sc_mean, tc_mean, ent_coef_value)), grads = jax.value_and_grad(
    #         actor_loss_fn, has_aux=True)(actor_state.params, actor_state, agent_id, actor_key, dropout_key)
    #     actor_state = actor_state.apply_gradients(grads=grads)

    #     return actor_state, {"actor_loss": actor_loss,
    #                          "run_costs": rc_mean,
    #                          "sto_costs": sc_mean,
    #                          "terminal_costs": tc_mean,
    #                          "min_dist": 0,
    #                          }, key, actions_upd, rc_up, sc_up, tc_up, ent_coef_value

    @staticmethod
    @partial(jax.jit, static_argnames=["target_entropy", "loss_low"])
    def train_actor_alpha(
            logp_actions,
            alpha_state,
            agent_id,
            target_entropy,
            loss_low,
    ):

        def actor_alpha_loss_fn(params, alpha_state, agent_id):
            log_prob = jax.lax.stop_gradient(logp_actions[agent_id]) + target_entropy
            log_alpha = alpha_state.apply_fn({"params": params}, 0)
            raw_loss = -(log_alpha * log_prob.mean())
            alpha_loss = jnp.maximum(loss_low, raw_loss)
            return alpha_loss

        alpha_loss, grads = jax.value_and_grad(actor_alpha_loss_fn)(alpha_state.params, alpha_state, agent_id)
        alpha_state = alpha_state.apply_gradients(grads=grads)
        return alpha_state, {"actor_alpha_loss": alpha_loss}

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "state_type", "crossq_style", "use_policy_active_masks", "auto_alpha",
                                       "action_type",
                                       "num_agents", "num_atoms", "v_min", "v_max", "env_name"])
    def train_actor_encoder(
            sp_obs,
            sp_available_actions,
            sp_share_obs,
            sp_valid_transition,
            rc_arr, sc_arr, tc_arr,
            actions,
            alpha_state,
            critic_state,
            actor_state,
            sampler,
            encoder_state,
            state_type,
            crossq_style,
            use_policy_active_masks,
            auto_alpha,
            agent_id,
            action_type,
            total_it,
            num_agents,
            num_atoms,
            v_min,
            v_max,
            key,
            env_name,
            latent=None,
    ):
        z_atoms = jnp.linspace(v_min, v_max, num_atoms)
        batch_size = sp_obs.shape[0]

        def actor_loss_fn(params, encoder_params, actor_state, encoder_state, agent_id, actor_key, dropout_key):
            P = Dspic.get_P(encoder_state, encoder_params, latent)
            Pi = jax.lax.dynamic_slice_in_dim(P, start_index=agent_id, axis=0, slice_size=1)
            Pi = jnp.tile(Pi, (batch_size, 1, 1))

            act_i, rc, sc, tc = Dspic.get_actions_with_logprobs_unscale(actor_state, params, sp_obs,
                                                                                           Pi, actor_key, sampler,
                                                                                           available_actions=sp_available_actions,
                                                                                           action_type=action_type)
            if env_name == "lbf":
                sc = sc + rc + tc
            actions_up = actions.at[agent_id].set(act_i)
            rc_up = rc_arr.at[agent_id].set(rc)
            sc_up = sc_arr.at[agent_id].set(sc)
            tc_up = tc_arr.at[agent_id].set(tc)

            if state_type == "EP":
                rc_ = rc_up[agent_id]
                sc_ = sc_up[agent_id]
                tc_ = tc_up[agent_id]
                actions_t = jnp.concatenate(actions_up, axis=-1)
            elif state_type == "FP":
                rc_ = jnp.tile(rc_up[agent_id], (num_agents, 1))
                sc_ = jnp.tile(sc_up[agent_id], (num_agents, 1))
                tc_ = jnp.tile(tc_up[agent_id], (num_agents, 1))
                actions_t = jnp.tile(jnp.concatenate(actions_up, axis=-1), (num_agents, 1))
            else:
                # Unimplemented!
                assert 0

            if crossq_style:
                qf_pi = critic_state.apply_fn(
                    {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                    sp_share_obs, actions_t,
                    rngs={"dropout": dropout_key},
                    train=False,
                )

                qf_pi1 = jnp.sum(qf_pi[0] * z_atoms, axis=-1)
                qf_pi2 = jnp.sum(qf_pi[1] * z_atoms, axis=-1)
            else:
                qf_pi = critic_state.apply_fn({"params": critic_state.params}, sp_share_obs, actions_t)
                qf_pi1, qf_pi2 = qf_pi[0], qf_pi[1]

            reduced_qf_pi = jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
            if auto_alpha:
                ent_coef_value = alpha_state.apply_fn({"params": alpha_state.params}, 0)
                ent_coef_value = jax.lax.stop_gradient(ent_coef_value)
                if action_type == "Discrete":
                    dif_ent_coef_value = 0
                else:
                    dif_ent_coef_value = ent_coef_value
            else:
                ent_coef_value = alpha_state
                if action_type == "Discrete":
                    dif_ent_coef_value = 0
                else:
                    dif_ent_coef_value = ent_coef_value

            if use_policy_active_masks:
                if state_type == "EP":
                    actor_loss = jnp.sum((-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (
                                rc_ + tc_)).squeeze()) * sp_valid_transition.squeeze()) / (
                                             sp_valid_transition.sum() + 1e-8)
                elif state_type == "FP":
                    valid_transition = jnp.tile(sp_valid_transition, (num_agents, 1))
                    actor_loss = jnp.sum((-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (
                                rc_ + tc_)).squeeze()) * valid_transition.squeeze()) / (valid_transition.sum() + 1e-8)
            else:
                actor_loss = jnp.mean(
                    (-reduced_qf_pi + (ent_coef_value * sc_ + dif_ent_coef_value * (rc_ + tc_)).squeeze()))

            products = jnp.einsum('imk,jml->ijkl', P, P)
            norms = jnp.sum(jnp.square(products), axis=(2, 3))
            dist_weighted = True
            if dist_weighted:
                sum_sq = jnp.sum(jnp.square(latent), axis=1)
                dot_product = -2 * jnp.dot(latent, latent.T)
                dist_sq = sum_sq[:, None] + dot_product + sum_sq[None, :]
                dist_sq = jnp.maximum(dist_sq, 0.0)
                off_diagonal_norms = norms * jnp.sqrt(dist_sq)
            else:
                N = P.shape[0]
                off_diagonal_norms = norms * (1 - jnp.eye(N, dtype=norms.dtype))
            final_norm_sum = jnp.sum(off_diagonal_norms)

            actor_loss, min_dist = actor_loss + final_norm_sum * 50, 0

            return actor_loss, (actions_up, rc_up, sc_up, tc_up, rc_.mean(), sc_.mean(), tc_.mean(), final_norm_sum,
                                jnp.mean(reduced_qf_pi), min_dist, ent_coef_value, 0)

        key, actor_key, dropout_key = jax.random.split(key, 3)
        (actor_loss, (
        actions_upd, rc_up, sc_up, tc_up, rc_mean, sc_mean, tc_mean, orthogonal_loss, q_taken_mean, min_dist,
        ent_coef_value, _)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True, argnums=(0, 1))(actor_state.params,
                                                                                                     encoder_state.params,
                                                                                                     actor_state,
                                                                                                     encoder_state,
                                                                                                     agent_id,
                                                                                                     actor_key,
                                                                                                     dropout_key)
        actor_state = actor_state.apply_gradients(grads=grads[0])
        encoder_state = encoder_state.apply_gradients(grads=grads[1])

        return actor_state, encoder_state, {"actor_loss": actor_loss,
                                            "run_costs": rc_mean,
                                            "sto_costs": sc_mean,
                                            "terminal_costs": tc_mean,
                                            "orthogonal_loss": orthogonal_loss,
                                            "q_taken_mean": q_taken_mean,
                                            "min_dist": min_dist,
                                            "actor_ent_coef": ent_coef_value,
                                            }, key, actions_upd, rc_up, sc_up, tc_up, ent_coef_value