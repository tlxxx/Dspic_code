import numpy as np
# from gym import utils
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import mujoco_env
# from gym.envs.mujoco import mujoco_env
import os
from jinja2 import Template


class ManyAgentSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        agent_conf = kwargs.get("agent_conf")
        n_agents = int(agent_conf.split("x")[0])
        n_segs_per_agents = int(agent_conf.split("x")[1])
        n_segs = n_agents * n_segs_per_agents

        obs_dim = 2 * n_segs + 2
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "manyagent_swimmer_{}_agents_each_{}_segments.auto.xml".format(
                n_agents, n_segs_per_agents
            ),
        )
        # if not os.path.exists(asset_path):
        print(
            "Auto-Generating Manyagent Swimmer asset with {} segments at {}.".format(
                n_segs, asset_path
            )
        )
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        # asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',git p
        #                          'manyagent_swimmer.xml')

        mujoco_env.MujocoEnv.__init__(self, asset_path, 4, observation_space=observation_space)
        utils.EzPickle.__init__(self)

    # def _generate_asset(self, n_segs, asset_path):
    #     template_path = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "assets",
    #         "manyagent_swimmer.xml.template",
    #     )
    #     with open(template_path, "r") as f:
    #         t = Template(f.read())
    #     body_str_template = """
    #     <body name="mid{:d}" pos="-1 0 0">
    #       <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
    #       <joint axis="0 0 {:d}" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
    #     """

    #     body_end_str_template = """
    #     <body name="back" pos="-1 0 0">
    #         <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
    #         <joint axis="0 0 1" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
    #       </body>
    #     """

    #     body_close_str_template = "</body>\n"
    #     actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

    #     body_str = ""
    #     for i in range(1, n_segs - 1):
    #         body_str += body_str_template.format(i, (-1) ** (i + 1), i)
    #     body_str += body_end_str_template.format(n_segs - 1)
    #     body_str += body_close_str_template * (n_segs - 2)

    #     actuator_str = ""
    #     for i in range(n_segs):
    #         actuator_str += actuator_str_template.format(i)

    #     rt = t.render(body=body_str, actuators=actuator_str)
    #     with open(asset_path, "w") as f:
    #         f.write(rt)
    #     pass

    def _generate_asset(self, n_segs, asset_path):
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "manyagent_swimmer.xml.template",
        )
        with open(template_path, "r") as f:
            t = Template(f.read())

        # (改动) 模板1：用于中间的身体节段
        body_part_template = """
        <body name="mid{i}" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 {axis}" limited="true" name="rot{i}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        # (新增) 模板2：用于最后一个身体节段
        last_body_part_template = """
        <body name="back" pos="-1 0 0">
            <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="rot{i}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        # (新增) 定义3：单独的闭合标签，用于后续循环
        body_close_str = "</body>"
        
        actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

        # [添加] 下面是新的、正确的循环逻辑
        body_str = ""
        # (改动) 循环 n_segs - 1 次来创建正确的嵌套结构
        for i in range(1, n_segs):
            if i == n_segs - 1:
                # 如果是最后一个节段，使用 last_body_part_template
                body_str += last_body_part_template.format(i=i)
            else:
                # 其他中间节段，使用普通模板
                axis = (-1) ** (i + 1)
                body_str += body_part_template.format(i=i, axis=axis)
        
        # (改动) 添加正确数量的闭合标签
        # 创建了 n_segs - 1 个 opening <body>, 就需要 n_segs - 1 个 closing </body>
        body_str += body_close_str * (n_segs - 1)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(i)

        rt = t.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as f:
            f.write(rt)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.data.qpos[0]  # 修改这里
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos[0]  # 修改这里
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl

        terminated = False
        truncated = False
        info = dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
        
        ob = self._get_obs()
        
        return ob, reward, terminated, truncated, info

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()
