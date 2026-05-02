import copy
from gymnasium.wrappers import TimeLimit
import gymnasium as gym # 使用最新的 gymnasium 库
import numpy as np

class LBFEnv:
    """
    A wrapper for single-agent Gymnasium environments to make them compatible
    with a multi-agent framework by treating the single agent as a team of one.
    """
    def __init__(self, args: dict):
        """
        Initializes the wrapper.
        
        Args:
            args (dict): A dictionary containing environment arguments.
                         Must include a "scenario" key, e.g., {"scenario": "LunarLander-v2"}.
        """
        self.args = copy.deepcopy(args)
        
        # 使用 gymnasium.make 创建环境
        self.env = TimeLimit(gym.make(self.args["scenario"]), max_episode_steps=50)
        self.observation_space = list(self.env.observation_space)
        self.action_space = list(self.env.action_space)
        self.share_observation_space = list(self.env.observation_space)
        self.n_agents = len(self.action_space)

        self.action_type = 'Discrete'

    def step(self, actions):
        """
        Executes a step in the environment.

        Args:
            actions: A list or array containing the action for the single agent.
                     e.g., [[action_val]] or [action_val]

        Returns:
            A tuple in the format (local_obs, global_state, rewards, dones, infos, available_actions),
            where each element is a list of length 1.
        """
        # 从 actions 列表中提取单个动作
        # actions 的形状可能是 (1, action_dim) or (1,)

        # 调用原始环境的 step 方法
        # gymnasium 的 step 返回 5 个值
        
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        dones = terminated or truncated

        # 在 info 中添加 "bad_transition" 标志，用于处理 TimeLimit 导致的 episode 结束
        # 这在计算 TD target 时非常重要
        if truncated:
            for agent in range(self.n_agents):
                infos[f"agent_{agent}"] = {"bad_transition": True}
        

        return list(obs), [obs[0]] * self.n_agents, [[r] for r in rewards], [dones] * self.n_agents, [infos] * self.n_agents, self.get_avail_actions()

    def reset(self, seed=None, options=None):
        """
        Resets the environment.

        Returns:
            A tuple in the format (initial_obs, initial_state, available_actions),
            where each element is a list of length 1.
        """
        # gymnasium 的 reset 返回 2 个值
        obs, info = self.env.reset(seed=seed, options=options)
        
        # return np.array(obs), np.array(obs), np.array(self.get_avail_actions())
        return obs, obs, self.get_avail_actions()


    def get_avail_actions(self):
        """
        Returns the available actions for the agent.
        For LunarLander-v2, all actions are always available.
        """
        if self.action_type == 'Discrete':
            # 创建一个全为 1 的列表，长度为动作空间的大小
            avail_actions = [[1] * self.action_space[i].n for i in range(self.n_agents)]
            return avail_actions
        else:
            # 连续动作空间通常没有 available_actions 的概念
            return None

    def render(self, mode='human'):
        """Renders the environment."""
        self.env.render()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def seed(self, seed=None):
        """Seeds the environment's random number generator."""
        # gymnasium 的 seed 是在 reset 时传入的
        # 这里保留一个兼容旧 gym 的接口
        self.env.reset(seed=seed)