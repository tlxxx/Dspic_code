import os
import time
import torch
import jax
import optax
import numpy as np
import jax.numpy as jnp
import setproctitle
from pathlib import Path
from flax import linen as nn
from datetime import datetime
from functools import partial
from flax.training.train_state import TrainState
from torch.distributions import Categorical
from src.utils.trans_tools import _t2n
from src.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from src.utils.configs_tools import init_dir, save_config, get_task_name
from src.algorithms.actors import ALGO_REGISTRY
from src.algorithms.critics import CRITIC_REGISTRY
from src.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from src.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from src.algorithms.actors.dspic import Dspic
from src.models.base.vae import VAE


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self, step) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)

def writerunner(outpath, algo):
    with open("src/runners/off_policy_base_runner.py", 'r', encoding='utf-8') as f:
        baserunner = f.read()
    if algo == "dspic":
        with open("src/runners/dspic_runner.py", 'r', encoding='utf-8') as f:
            algorunner = f.read()
    else:
        assert 0

    output_dir = os.path.dirname(outpath)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_content = "BaseRunner: \n" + baserunner + "\n\n\n AlgoRunner: \n" + algorunner
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(write_content)

class OffPolicyBaseRunner:
    """Base runner for off-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OffPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = self.algo_args["algo"]["policy_freq"]
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        set_seed(algo_args["seed"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["render"]["use_render"]:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
                train_args=self.algo_args["train"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"), "w", encoding="utf-8"
            )
            writerunner(outpath=self.log_dir + "/runner_code.txt", algo=args["algo"])
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                # 1,
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)
        self.agent_deaths = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
        )
        self.batch_size = self.algo_args["algo"]["batch_size"]
        self.v_min, self.v_max, self.num_atoms = self.algo_args["model"]["v_min"], self.algo_args["model"]["v_max"], self.algo_args["model"]["n_atoms"]
        self.z_atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(algo_args["seed"]["seed"] + agent_id + 1)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.key = jax.random.PRNGKey(int(algo_args["seed"]["seed"]))
        self.P_dim = self.algo_args["model"]["latent_dim"]
        self.use_id_concat = (self.algo_args["train"]["role_term"] == "id")
        self.use_vae = (self.algo_args["train"]["role_term"] == "vae")
        self.Orthogonal = self.algo_args["train"]["Orthogonal"]

        # if self.Orthogonal:
        #     actor_name = args["algo"] + "_ort"
        actor_name = args["algo"]

        if self.share_param:
            self.actor = []
            agent_key, self.key = jax.random.split(self.key, 2)
            agent = ALGO_REGISTRY[actor_name](
                {**algo_args["model"], **algo_args["algo"], **args},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                self.batch_size,
                use_target_network=True,
                n_agents=self.num_agents,
                key=agent_key,
                role_term=self.algo_args["train"]["role_term"] if "role_term" in self.algo_args["train"].keys() else None,
                latent_dim=10,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent_key, self.key = jax.random.split(self.key, 2)
                agent = ALGO_REGISTRY[actor_name](
                    {**algo_args["model"], **algo_args["algo"], **args},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    self.batch_size,
                    use_target_network=True,
                    n_agents=self.num_agents,
                    key=agent_key,
                    role_term=self.algo_args["train"]["role_term"] if "role_term" in self.algo_args["train"].keys() else None,
                    latent_dim=10,
                )

                self.actor.append(agent)

        if not self.algo_args["render"]["use_render"]:
            critic_key, self.key = jax.random.split(self.key, 2)
            self.use_bnstats_from_live_net = False
            if self.envs.action_space[0].__class__.__name__ == "Discrete":
                self.crossq_style = False
                self.critic = CRITIC_REGISTRY[args["algo"]](
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.envs.action_space,
                    self.num_agents,
                    self.state_type,
                    self.batch_size,
                    critic_key,
                )
            else:
                self.crossq_style = True
                self.critic = CRITIC_REGISTRY[args["algo"] + "_crossq"](
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.envs.action_space,
                    self.num_agents,
                    self.state_type,
                    self.batch_size,
                    critic_key,
                )

            # For Mamujoco, is EP; For smac, is FP
            if self.state_type == "EP":
                self.buffer = OffPolicyBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            elif self.state_type == "FP":
                self.buffer = OffPolicyBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            else:
                raise NotImplementedError

        self.value_normalizer = None
        act_dim = self.envs.action_space[0].shape[0] if self.envs.action_space[0].__class__.__name__ == "Box" else self.envs.action_space[0].n
        if self.use_vae:
            self.key, vae_key = jax.random.split(self.key, 2)
            self.vae_batch_size = self.algo_args["model"]["vae_batch_size"]
            self.role_embedding = VAE(self.num_agents, 64, 10, self.envs.observation_space[0], act_dim, 0.0003, 0.0003, self.algo_args["model"]["vae_alpha1"], self.vae_batch_size, vae_key)
             

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # total iteration

        if (
            "auto_alpha" in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                if (
                    self.envs.action_space[agent_id].__class__.__name__ == "Box"
                ):  # Differential entropy can be negative
                    if self.args["algo"] == "dspic":
                        self.target_entropy.append(-2.0 * self.envs.action_space[agent_id].shape[0])
                    else:
                        self.target_entropy.append(
                            -np.prod(self.envs.action_space[agent_id].shape)
                        )
                else:  # Discrete entropy is always positive. Thus we set the max possible entropy as the target entropy
                    if self.args["algo"] == "dspic":
                        self.target_entropy.append(
                            -0.08
                            * np.log(1.0 / np.prod(self.envs.action_space[agent_id].n))
                        )
                    else:
                        self.target_entropy.append(
                            -0.98
                            * np.log(1.0 / np.prod(self.envs.action_space[agent_id].shape))
                        )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha_states = []
            for agent_id in range(self.num_agents):
                self.key, alpha_key = jax.random.split(self.key, 2)
                self.log_alpha = EntropyCoef(self.algo_args["algo"]["alpha_init"])
                alpha_params = self.log_alpha.init(alpha_key, 0.0)['params']
                alpha_optx = optax.chain(
                    optax.clip_by_global_norm(10.0),                
                    optax.adam(self.algo_args["algo"]["alpha_lr"]) 
                )
                self.alpha_model_state = TrainState.create(apply_fn=self.log_alpha.apply,
                                                                       params=alpha_params, tx=alpha_optx)

                self.alpha_states.append(self.alpha_model_state)

        elif "alpha" in self.algo_args["algo"].keys():
            self.alpha = [self.algo_args["algo"]["alpha_init"]] * self.num_agents

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"]:  # render, not train
            self.render()
            return
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        # warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start training")

        if self.use_vae:
            print("start training VAE!")
            vae_metrics = self.train_vae(self.algo_args["model"]["vae_epochs"], pretrain=True)
            if self.algo_args["train"]["log_tb"]:
                for (k, v) in vae_metrics.items():
                    self.writter.add_scalar(k, v, 0)
            self.agent_embeddings = self.role_embedding.get_embeddings()
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"finish training VAE!Time: {time_str}.")
            
        # train and eval
        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        update_num = int(  # update number per train
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        Metrics = {"step": 0}
        batch_size = obs.shape[0]

        current_episode_steps = np.zeros(self.algo_args["train"]["n_rollout_threads"], dtype=int)
        for step in range(1, steps + 1):
            if self.use_id_concat:
                if self.Orthogonal:
                    agent_onehot = np.tile(np.eye(self.num_agents)[np.newaxis, :, :], (batch_size, 1, 1))
                    P = self.actor[0].get_Ps(agent_onehot[0])
                    P = np.tile(P.reshape(1, self.num_agents, self.algo_args["model"]["latent_dim"], -1), (self.algo_args["train"]["n_rollout_threads"], 1, 1, 1))
                    actions, _ = self.get_actions(
                        obs, available_actions=available_actions, add_random=True, latent=P, share_obs=share_obs, K=1
                    )
                else:
                    agent_onehot = np.tile(np.eye(self.num_agents)[np.newaxis, :, :], (batch_size, 1, 1))
                    obs_input = np.concatenate([obs, agent_onehot], axis=-1)
                    actions, _ = self.get_actions(
                        obs_input, available_actions=available_actions, add_random=True, share_obs=share_obs, K=1
                    )
            elif self.use_vae:
                if self.Orthogonal:
                    agent_embeddings = np.tile(self.agent_embeddings.reshape(1, self.num_agents, -1), (batch_size, 1, 1))
                    P = self.actor[0].get_Ps(agent_embeddings[0])
                    P = np.tile(P.reshape(1, self.num_agents, self.algo_args["model"]["latent_dim"], -1), (self.algo_args["train"]["n_rollout_threads"], 1, 1, 1))
                    actions, _ = self.get_actions(
                        obs, available_actions=available_actions, add_random=True, latent=P, share_obs=share_obs, K=1
                    )
                else:
                    agent_embeddings = np.tile(self.agent_embeddings.reshape(1, self.num_agents, -1), (batch_size, 1, 1))
                    obs_input = np.concatenate([obs, agent_embeddings], axis=-1)
                    actions, _ = self.get_actions(
                        obs_input, available_actions=available_actions, add_random=True, share_obs=share_obs, K=1
                    )
            else:
                actions, _ = self.get_actions(
                    obs, available_actions=available_actions, add_random=True, share_obs=share_obs, K=1
                )


            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(
                actions
            )  # rewards: (n_threads, n_agents, 1); dones: (n_threads, n_agents)
            # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            current_episode_steps += 1
            env_dones = np.any(dones, axis=1)
            current_episode_steps[env_dones] = 0

            if (step % self.algo_args["train"]["train_interval"] == 0) or (step == -1):
                if self.algo_args["train"]["use_linear_lr_decay"]:
                    if self.share_param:
                        self.actor[0].lr_decay(step, steps)
                    else:
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    self.target_ent_coeff = 0 
                    self.loss_low = -float(self.algo_args["train"]["epsilon0"]) if self.algo_args["train"]["warmup_steps"] + step * self.algo_args["train"]["n_rollout_threads"] <= int(self.algo_args["train"]["t0"]) else 0
                    metrics = self.train()
                    for (k, v) in metrics.items():
                        if k not in Metrics or Metrics[k] == []:
                            Metrics[k] = [v]
                        else:
                            Metrics[k].append(v)
                    Metrics["step"] += 1

                if (step % (self.algo_args["train"]["train_interval"] * 200) == 0) & (self.algo_args["model"]["vae_ft_epochs"] > 0) & self.use_vae:
                    vae_metrics = self.train_vae(self.algo_args["model"]["vae_ft_epochs"], pretrain=False)
                    self.agent_embeddings = self.role_embedding.get_embeddings()
                    for (k, v) in vae_metrics.items():
                        if k not in Metrics.keys():
                            Metrics[k] = [v]
                        else:
                            Metrics[k].append(v)
                if step % (self.algo_args["train"]["train_interval"] * 20) == 0:
                    now = datetime.now()
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    tr = step // (self.algo_args["train"]["train_interval"])
                    print(f"train epoch {tr} finished! Time: {time_str}.")
            
            if step % self.algo_args["train"]["eval_interval"] == 0:
                cur_step = (
                    self.algo_args["train"]["warmup_steps"]
                    + step * self.algo_args["train"]["n_rollout_threads"]
                )
                if self.algo_args["train"]["log_tb"]:
                    for (k, v) in Metrics.items():
                        if k == "step":
                            continue
                        if type(v) != list:
                            continue
                        self.writter.add_scalar(k, np.array(v).mean(), cur_step)
                Metrics = {"step": 0}
                if self.algo_args["eval"]["use_eval"]:
                    print(
                        "\033[31m[INFO]\033[0m: ",
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Evaluation at step {cur_step} / {self.algo_args['train']['num_env_steps']}:"
                    )
                    self.eval(cur_step)
                else:
                    print(
                        "\033[31m[INFO]\033[0m: ",
                        f"Env {self.args['env']} Task {self.task_name} Algo {self.args['algo']} Exp {self.args['exp_name']} Step {cur_step} / {self.algo_args['train']['num_env_steps']}, average step reward in buffer: {self.buffer.get_mean_rewards()}.\n"
                    )
                    if len(self.done_episodes_rewards) > 0:
                        aver_episode_rewards = np.mean(self.done_episodes_rewards)
                        print(
                            "Some episodes done, average episode reward is {}.\n".format(
                                aver_episode_rewards
                            )
                        )
                        self.log_file.write(
                            ",".join(map(str, [cur_step, aver_episode_rewards])) + "\n"
                        )
                        self.log_file.flush()
                        self.done_episodes_rewards = []
                # self.save()

    def warmup(self):
        """Warmup the replay buffer with random actions"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"]["n_rollout_threads"]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, available_actions = self.envs.reset()

        self.obs_dim = obs.shape[-1]
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(available_actions)
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = new_available_actions.copy()
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(1, 0, 2)
                if len(np.array(available_actions).shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
        return obs, share_obs, available_actions

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_agents, n_threads, obs_dim)
            actions,  # (n_agents, n_threads, action_dim)
            available_actions,  # None or (n_agents, n_threads, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition denotes whether each transition is valid or not (invalid if corresponding agent is dead)
        # shape: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(dones, axis=-1)

        # terms use False to denote truncation and True to denote termination
        if self.state_type == "EP":
            terms = np.full((self.algo_args["train"]["n_rollout_threads"], 1), False)
            for i in range(self.algo_args["train"]["n_rollout_threads"]):
                if dones_env[i]:
                    if not (
                        "bad_transition" in infos[i][0].keys()
                        and infos[i][0]["bad_transition"] == True
                    ):
                        terms[i][0] = True
        elif self.state_type == "FP":
            terms = np.full(
                (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
                False,
            )
            for i in range(self.algo_args["train"]["n_rollout_threads"]):
                for agent_id in range(self.num_agents):
                    if dones[i][agent_id]:
                        if not (
                            "bad_transition" in infos[i][agent_id].keys()
                            and infos[i][agent_id]["bad_transition"] == True
                        ):
                            terms[i][agent_id][0] = True
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[i]:
                self.done_episodes_rewards.append(self.train_episode_rewards[i])
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1)
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = infos[i][0]["original_obs"].copy()
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = infos[i][0]["original_state"].copy()

        if self.state_type == "EP":
            data = (
                share_obs[:, 0],  # (n_threads, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards[:, 0],  # (n_threads, 1)
                np.expand_dims(dones_env, axis=-1),  # (n_threads, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, 1)
                next_share_obs[:, 0],  # (n_threads, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )
        elif self.state_type == "FP":
            data = (
                share_obs,  # (n_threads, n_agents, share_obs_dim)
                obs,  # (n_agents, n_threads, obs_dim)
                actions,  # (n_agents, n_threads, action_dim)
                available_actions,  # None or (n_agents, n_threads, action_number)
                rewards,  # (n_threads, n_agents, 1)
                np.expand_dims(dones, axis=-1),  # (n_threads, n_agents, 1)
                valid_transitions.transpose(1, 0, 2),  # (n_agents, n_threads, 1)
                terms,  # (n_threads, n_agents, 1)
                next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
                next_obs.transpose(1, 0, 2),  # (n_agents, n_threads, next_obs_dim)
                next_available_actions,  # None or (n_agents, n_threads, next_action_number)
            )

        self.buffer.insert(data)

    def sample_actions(self, available_actions=None):
        """Sample random actions for warmup.
        Args:
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
        Returns:
            actions: (np.ndarray) sampled actions, shape is (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            for thread in range(self.algo_args["train"]["n_rollout_threads"]):
                if available_actions[thread] is None:
                    action.append(self.action_spaces[agent_id].sample())
                else:
                    action.append(
                        Categorical(
                            torch.tensor(available_actions[thread, agent_id, :])
                        ).sample()
                    )
            actions.append(action)
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.expand_dims(np.array(actions).transpose(1, 0), axis=-1)

        return np.array(actions).transpose(1, 0, 2)

    def get_actions(self, obs, available_actions=None, add_random=True, latent=None, share_obs=None, K=1):
        """Get actions for rollout.
        Args:
            obs: (np.ndarray) input observation, shape is (n_threads, n_agents, dim)
            available_actions: (np.ndarray) denotes which actions are available to agent (if None, all actions available),
                                 shape is (n_threads, n_agents, action_number) or (n_threads, ) of None
            add_random: (bool) whether to add randomness
        Returns:
            actions: (np.ndarray) agent actions, shape is (n_threads, n_agents, dim)
        """
        meanlogp = []
        
        if self.args["algo"] == "dspic":
            actions = []
            # print("begin get actions!")
            q_max = -1000000 * np.ones((obs.shape[0], 1), dtype=np.float32)
            batch_size = obs.shape[0]
            for k in range(K):
                now_actions = []
                for agent_id in range(self.num_agents):
                    self.key, actor_key = jax.random.split(self.key, 2)
                    act_high, act_low = self.actor[agent_id].act_high if self.actor[agent_id].action_type == "Box" else 1, self.actor[agent_id].act_low if self.actor[agent_id].action_type == "Box" else 0
                    if (
                        len(np.array(available_actions).shape) == 3
                    ):  # (n_threads, n_agents, action_number)
                        if latent is None:
                            # action_, logp = DiffusionHASAC.get_actions(self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, obs[:, agent_id], actor_key,
                            #                                 self.actor[agent_id].sampler, act_high, act_low, 
                            #                                 available_actions=available_actions[:, agent_id], action_type=self.actor[agent_id].action_type, dtype=jnp.float32, stochastic=True if k > 1 else add_random)
                            # meanlogp.append(logp.mean())
                            # now_actions.append(
                            #     _t2n(
                            #         action_
                            #     )
                            # )
                            pass
                        else:
                            action_, logp = Dspic.get_actions(self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, obs[:, agent_id], latent[:, agent_id], actor_key,
                                                            self.actor[agent_id].sampler, act_high, act_low, 
                                                            available_actions=available_actions[:, agent_id], action_type=self.actor[agent_id].action_type, dtype=jnp.float32, stochastic=True if k > 1 else add_random)
                            meanlogp.append(logp.mean())
                            now_actions.append(
                                _t2n(
                                    action_
                                )
                            )
                    else:  # (n_threads, ) of None
                        if latent is None:
                            # action_, logp = DiffusionHASAC.get_actions(self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, obs[:, agent_id], actor_key,
                            #                                 self.actor[agent_id].sampler, act_high, act_low, 
                            #                                 available_actions=None, action_type=self.actor[agent_id].action_type, dtype=jnp.float32, stochastic=True if k > 1 else add_random)
                            # meanlogp.append(logp.mean())
                            # now_actions.append(
                            #     _t2n(
                            #         action_
                            #     )
                            # )
                            pass
                        else:
                            action_, logp = Dspic.get_actions(self.actor[agent_id].actor_state, self.actor[agent_id].actor_state.params, obs[:, agent_id], latent[:, agent_id], actor_key,
                                                            self.actor[agent_id].sampler, act_high, act_low, 
                                                            available_actions=None, action_type=self.actor[agent_id].action_type, dtype=jnp.float32, stochastic=True if k > 1 else add_random)
                            meanlogp.append(logp.mean())
                            now_actions.append(
                                _t2n(
                                    action_
                                )
                            )
                now_actions = np.array(now_actions).transpose(1, 0, 2).squeeze()
                if self.action_spaces[0].__class__.__name__ == "Discrete":
                    lim_a = self.action_spaces[0].n
                    if self.use_ae_model:
                        now_actions_ = self.action_codebook[now_actions].reshape(batch_size, -1)
                    else:
                        now_actions = jax.nn.one_hot(now_actions, num_classes=lim_a).reshape(batch_size, -1)
                elif self.action_spaces[0].__class__.__name__ == "Box":
                    now_actions = now_actions.reshape(batch_size, -1)
                if len(actions) == 0:
                    actions = np.zeros_like(now_actions, dtype=np.float32)
                share_obs_ = share_obs[:, 0]
                dropout_key, self.key = jax.random.split(self.key, 2)
                if self.action_spaces[0].__class__.__name__ == "Box":
                    act_low_ = jnp.array(act_low)
                    act_low_ = jnp.tile(act_low_.reshape(1, 1, -1), (now_actions.shape[0], self.num_agents, 1)).reshape(now_actions.shape[0], -1)
                    act_high_ = jnp.array(act_high)
                    act_high_ = jnp.tile(act_high_.reshape(1, 1, -1), (now_actions.shape[0], self.num_agents, 1)).reshape(now_actions.shape[0], -1)
                    now_actions_unscale = 2.0 * (now_actions - act_low_) / (act_high_ - act_low_) - 1.0
                    if K > 1:
                        q_value = OffPolicyBaseRunner.get_q(share_obs_, now_actions_unscale, dropout_key, self.critic.critic_state, self.num_atoms, self.v_min, self.v_max, crossq_style=self.crossq_style)
                    else:
                        q_value = jnp.zeros((share_obs_.shape[0], 1))
                else:
                    q_value = OffPolicyBaseRunner.get_q(share_obs_, now_actions, dropout_key, self.critic.critic_state, self.num_atoms, self.v_min, self.v_max, crossq_style=self.crossq_style)

                mask = (q_value >= q_max)
                actions = np.where(mask, now_actions, actions)

            actions = actions.reshape(batch_size, self.num_agents, -1)
            if self.action_spaces[0].__class__.__name__ == "Discrete":
                actions = np.expand_dims(np.argmax(actions, axis=-1), axis=-1)
                actions = actions.astype(np.int32)
            return actions, np.array(meanlogp)

        else:
            # Unimplemented!
            exit(114514)
            actions = []
            for agent_id in range(self.num_agents):
                actions.append(
                    _t2n(self.actor[agent_id].get_actions(obs[:, agent_id], add_random))
                )
        return np.array(actions).transpose(1, 0, 2), np.array(meanlogp).mean()

    def train(self):
        """Train the model"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, step):
        """Evaluate the model"""
        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        if "smac" in self.args["env"]:
            eval_battles_won = 0
        if "football" in self.args["env"]:
            eval_score_cnt = 0
        episode_lens = []
        one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int32
        )

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        logp_mean = []

        while True:
            if self.use_id_concat:
                if self.Orthogonal:
                    agent_onehot = np.tile(np.eye(self.num_agents)[np.newaxis, :, :], (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1))
                    P = self.actor[0].get_Ps(agent_onehot[0])
                    P = np.tile(P.reshape(1, self.num_agents, self.algo_args["model"]["latent_dim"], -1), (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1, 1))
                    eval_actions, logp = self.get_actions(
                        eval_obs, available_actions=eval_available_actions, add_random=False, latent=P, share_obs=eval_share_obs, K=1
                    )
                else:
                    agent_onehot = np.tile(np.eye(self.num_agents)[np.newaxis, :, :], (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1))
                    obs_input = np.concatenate([eval_obs, agent_onehot], axis=-1)
                    eval_actions, logp = self.get_actions(
                        obs_input, available_actions=eval_available_actions, add_random=False, share_obs=eval_share_obs, K=1
                    )
            elif self.use_vae:
                if self.Orthogonal:
                    agent_embeddings = np.tile(self.agent_embeddings.reshape(1, self.num_agents, -1), (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1))
                    P = self.actor[0].get_Ps(agent_embeddings[0])
                    P = np.tile(P.reshape(1, self.num_agents, self.algo_args["model"]["latent_dim"], -1), (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1, 1))
                    eval_actions, logp = self.get_actions(
                        eval_obs, available_actions=eval_available_actions, add_random=False, latent=P, share_obs=eval_share_obs, K=1
                    )
                else:
                    agent_embeddings = np.tile(self.agent_embeddings.reshape(1, self.num_agents, -1), (self.algo_args["eval"]["n_eval_rollout_threads"], 1, 1))
                    obs_input = np.concatenate([eval_obs, agent_embeddings], axis=-1)
                    eval_actions, logp = self.get_actions(
                        obs_input, available_actions=eval_available_actions, add_random=False, share_obs=eval_share_obs, K=1
                    )
            else:
                eval_actions, logp = self.get_actions(
                    eval_obs, available_actions=eval_available_actions, add_random=True, share_obs=eval_share_obs, K=1
                )
            logp_mean.append(logp)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            one_episode_len += 1

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if "smac" in self.args["env"]:
                        if "v2" in self.args["env"]:
                            if eval_infos[eval_i][0]["battle_won"]:
                                eval_battles_won += 1
                        else:
                            if eval_infos[eval_i][0]["won"]:
                                eval_battles_won += 1
                    if "football" in self.args["env"]:
                        if eval_infos[eval_i][0]["score_reward"] > 0:
                            eval_score_cnt += 1
                    eval_episode_rewards[eval_i].append(
                        np.sum(one_episode_rewards[eval_i], axis=0)
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(one_episode_len[eval_i].copy())
                    one_episode_len[eval_i] = 0

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                # eval_log returns whether the current model should be saved
                eval_episode_rewards = np.concatenate(
                    [rewards for rewards in eval_episode_rewards if rewards]
                )
                eval_avg_rew = np.mean(eval_episode_rewards)
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    print(
                        "Eval win rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_battles_won / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                elif "football" in self.args["env"]:
                    print(
                        "Eval score rate is {}, eval average episode rewards is {}, eval average episode length is {}.".format(
                            eval_score_cnt / eval_episode, eval_avg_rew, eval_avg_len
                        )
                    )
                else:
                    print(
                        f"Eval average episode reward is {eval_avg_rew}, eval average episode length is {eval_avg_len}.\n"
                    )
                if "smac" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_battles_won / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                elif "football" in self.args["env"]:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    eval_score_cnt / eval_episode,
                                ],
                            )
                        )
                        + "\n"
                    )
                else:
                    self.log_file.write(
                        ",".join(map(str, [step, eval_avg_rew, eval_avg_len])) + "\n"
                    )
                self.log_file.flush()
                if self.algo_args["train"]["log_tb"]:
                    self.writter.add_scalar(
                        "eval_average_episode_rewards", eval_avg_rew, step
                    )
                    self.writter.add_scalar(
                        "eval_average_episode_length", eval_avg_len, step
                    )
                    # self.writter.add_scalar(
                    #     "eval_min_episode_length", eval_min_len, step
                    # )
                    self.writter.add_scalar(
                        "eval_logp", np.array(logp_mean).mean(), step
                    )
                    if "smac" in self.args["env"]:
                        self.writter.add_scalar(
                            "win_rate", eval_battles_won / eval_episode, step
                        )
                break

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = np.array([eval_available_actions])
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = np.array([eval_available_actions])
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=eval_available_actions,
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        self.role_embedding.restore(self.algo_args["train"]["model_dir"])
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(self.algo_args["train"]["model_dir"], agent_id)
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(self.algo_args["train"]["model_dir"])
            if self.value_normalizer is not None:
                self.value_normalizer.restore(self.algo_args["train"]["model_dir"])

    def save(self):
        """Save the model"""
        # self.role_embedding.save(self.save_dir)
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(self.save_dir, agent_id)
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            self.value_normalizer.save(self.save_dir)

    def close(self):
        """Close environment, writter, and log file."""
        # post process
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            if self.algo_args["train"]["log_tb"]:
                self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
                self.writter.close()
            self.log_file.close()

    def train_vae(self, vae_epochs, pretrain=True):
        # get transitions
        sp_obs, sp_actions, sp_reward, sp_next_obs = self.buffer.get_all_transition()
        if self.action_spaces[0].__class__.__name__ == "Discrete":
            lim_a = self.role_embedding.action_dim
            squeezed_actions = jnp.squeeze(sp_actions, axis=-1)
            sp_actions = jax.nn.one_hot(squeezed_actions, num_classes=lim_a)
        sp_obs_actions = np.concatenate([sp_obs, sp_actions], axis=-1)
        if self.state_type == "EP":
            sp_reward = np.tile(sp_reward, (self.num_agents, 1, 1))
        else:
            sp_reward = jnp.transpose(sp_reward, (1, 0, 2))
        sp_next_obs_reward = np.concatenate([sp_next_obs, sp_reward], axis=-1)
        n_data = sp_obs.shape[1]
        # get agent number
        indices = np.arange(self.num_agents)
        agent_onehot = np.eye(self.num_agents)
        sp_agent_ids = np.repeat(agent_onehot[indices][:, np.newaxis, :], n_data, axis=1)
        sp_agent_ids = sp_agent_ids.reshape(self.num_agents * n_data, -1)
        sp_obs_actions = sp_obs_actions.reshape(self.num_agents * n_data, -1)
        sp_next_obs_reward = sp_next_obs_reward.reshape(self.num_agents * n_data, -1)
        vae_metrics = self.role_embedding.train(sp_agent_ids, sp_obs_actions, sp_next_obs_reward, vae_epochs, n_data * self.num_agents, pretrain=pretrain)

        return vae_metrics

    @staticmethod
    @partial(jax.jit, static_argnames=["num_atoms", "v_min", "v_max", "crossq_style"])
    def get_q(share_obs, actions, dropout_key, critic_state, num_atoms, v_min, v_max, crossq_style):
        if crossq_style:
            z_atoms = jnp.linspace(v_min, v_max, num_atoms)
            qf_pi = critic_state.apply_fn(
                {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                share_obs, actions,
                rngs={"dropout": dropout_key},
                train=False,
            )
            qf_pi1 = jax.lax.stop_gradient(jnp.sum(qf_pi[0] * z_atoms, axis=-1))
            qf_pi2 = jax.lax.stop_gradient(jnp.sum(qf_pi[1] * z_atoms, axis=-1))
        else:
            qf_pi = critic_state.apply_fn({"params": critic_state.params}, share_obs, actions)
            qf_pi1, qf_pi2 = qf_pi[0], qf_pi[1]
        reduced_qf_pi = jnp.min(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).reshape(-1, 1)
        return reduced_qf_pi
