import robosuite as suite
import os 
import logging 
import time
import mujoco_py
import numpy as np
from gym import spaces
from collections import OrderedDict
from robosuite import load_controller_config
from robosuite.controllers import controller_factory 

import util.transform_utils as T
from util.logger import logger

def update_controller_config(env, config):
    controller_config = config.copy()
    controller_config["robot_name"] = env.robots[0].name
    controller_config["sim"] = env.robots[0].sim
    controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    controller_config["actuator_range"] = env.robots[0].torque_limits
    controller_config["policy_freq"] = env.robots[0].control_freq
    controller_config["ndim"] = len(env.robots[0].robot_joints)
    return controller_config

def apply_controller(controller, action, robot, policy_step):
    gripper_action = None
    if robot.has_gripper:
        gripper_action = action[
            controller.control_dim :
        ]  # all indexes past controller dimension indexes
        arm_action = action[: controller.control_dim]
    else:
        arm_action = action

    # Update the controller goal if this is a new policy step
    if policy_step:
        controller.set_goal(arm_action)

    # Now run the controller for a step
    torques = controller.run_controller()

    # Clip the torques
    low, high = robot.torque_limits
    torques = np.clip(torques, low, high)

    # Get gripper action, if applicable
    if robot.has_gripper:
        robot.grip_action(gripper=robot.gripper, gripper_action=gripper_action)

    # Apply joint torque control
    robot.sim.data.ctrl[robot._ref_joint_actuator_indexes] = torques

class Lift():

    def __init__(self, **kwargs):
        # default env config
        # self._env_config = {
        #     "frame_skip": kwargs["frame_skip"],
        #     "ctrl_reward": kwargs["ctrl_reward_coef"],
        #     "init_randomness": 1e-5,
        #     "max_episode_steps": kwargs["max_episode_steps"],
        #     "unstable_penalty": 0,
        #     "reward_type": kwargs["reward_type"],
        #     "distance_threshold": kwargs["distance_threshold"],
        # }
        logger.setLevel(logging.INFO)
        self.render_mode = "no"  # ['no', 'human', 'rgb_array']
        # self._screen_width = kwargs["screen_width"]
        # self._screen_height = kwargs["screen_height"]
        # self._seed = kwargs["seed"]
        # self.seed(self._seed)
        self._gym_disable_underscore_compat = True
        # self._kp = kwargs["kp"]
        # self._kd = kwargs["kd"]
        # self._ki = kwargs["ki"]
        # self._frame_dt = kwargs["frame_dt"]
        # self._ctrl_reward_coef = kwargs["ctrl_reward_coef"]
        # self._camera_name = kwargs["camera_name"]
        self._kwargs = kwargs
        self._terminal = False
        # env config: 
        expl_environment_kwargs = {
            "control_freq": 20,
            "controller_configs": {
            "control_delta": True,
            "damping": 1,
            "damping_limits": [
                0,
                10
            ],
            "impedance_mode": "fixed",
            "input_max": 1,
            "input_min": -1,
            "interpolation": None,
            "kp": 150,
            "kp_limits": [
                0,
                300
            ],
            "orientation_limits": None,
            "output_max": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5
            ],
            "output_min": [
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5
            ],
            "position_limits": None,
            "ramp_ratio": 0.2,
            "type": "JOINT_POSITION",
            "uncouple_pos_ori": True
            },
            "env_name": "Lift",
            "horizon": 250,
            "ignore_done": True,
            "reward_shaping": True,
            "robots": "Sawyer",
            "use_object_obs": True,
            "camera_names":"frontview"
        }
        # config = load_controller_config(default_controller="JOINT_POSITION")
        # config['kp'] = 150
        # config['output_max'] = 0.5
        # config['output_min'] = -0.5
        self.horizon = 250
        self.env = suite.make(**expl_environment_kwargs)
        self.sim = self.env.sim 
        
        self.xml_path = "/home/tarunc/mopa-rl/rl/lift_env.xml"
        minimum = -np.ones(self.env.robots[0].dof)
        maximum = np.ones(self.env.robots[0].dof)
        # for osc pose
        # minimum = -np.ones(7)
        # maximum = np.ones(7)

        self.action_space = spaces.Dict(
            [("default", spaces.Box(low=minimum, high=maximum, dtype=np.float32))]
        )

        jnt_range = self.sim.model.jnt_range
        is_jnt_limited = self.sim.model.jnt_limited.astype(np.bool)
        jnt_minimum = np.full(len(is_jnt_limited), fill_value=-np.inf, dtype=np.float)
        jnt_maximum = np.full(len(is_jnt_limited), fill_value=np.inf, dtype=np.float)
        jnt_minimum[is_jnt_limited], jnt_maximum[is_jnt_limited] = jnt_range[
            is_jnt_limited
        ].T
        jnt_minimum[np.invert(is_jnt_limited)] = -3.14
        jnt_maximum[np.invert(is_jnt_limited)] = 3.14
        self._is_jnt_limited = is_jnt_limited
        self.joint_space = spaces.Dict(
            [
                (
                    "default",
                    spaces.Box(low=jnt_minimum, high=jnt_maximum, dtype=np.float32),
                )
            ]
        )
        self._jnt_minimum = jnt_minimum
        self._jnt_maximum = jnt_maximum

        # joint position indices in qpos and qvel
        self.ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self.ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        # indices for grippers in qpos, qvel
        self.ref_gripper_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
        ]
        self.ref_gripper_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
        ]

        self.jnt_indices = []
        for i, jnt_type in enumerate(self.sim.model.jnt_type):
            if jnt_type == 0:
                for _ in range(7):
                    self.jnt_indices.append(i)
            elif jnt_type == 1:
                for _ in range(4):
                    self.jnt_indices.append(i)
            else:
                self.jnt_indices.append(i)

        # dof 
        self.dof = self.env.robots[0].dof
        self.robot_dof = 7
        
        # action scale for planner 
        self._ac_scale = 0.05

        # ignoring actuator indexes for now
        # controller config for joint space actions 
        # jp_ctrl_config = update_controller_config(self.env, config)
        # self.jp_ctrl = controller_factory("JOINT_POSITION", jp_ctrl_config)

    @property 
    def robot_joints(self):
        return [f"robot0_right_j{i}" for i in range(7)]

    @property 
    def gripper_joints(self):
        return ['gripper0_l_finger_joint', 'gripper0_r_finger_joint']
    @property
    def observation_space(self):
        observation = self.env._get_observations()
        observation_space = OrderedDict()
        for k, v in observation.items():
            if k not in ['object-state', 'image-state', 'robot0_proprio-state', 'frontview_image']:
                observation_space[k] = spaces.Box(low=-1.0, high=-1.0, shape = v.shape)
        return spaces.Dict(observation_space)

    @property 
    def manipulation_geom(self):
        return ["cube_g0"]

    @property 
    def manipulation_geom_ids(self):
        return [self.sim.model.geom_name2id(name) for name in self.manipulation_geom]
    # maybe need to fix these guys later
    @property
    def static_bodies(self):
        return ["world", "table"]
    
    @property
    def static_geoms(self):
        return []

    @property
    def static_geom_ids(self):
        body_ids = []
        for body_name in self.static_bodies:
            body_ids.append(self.sim.model.body_name2id(body_name))

        geom_ids = []
        for geom_id, body_id in enumerate(self.sim.model.geom_bodyid):
            if body_id in body_ids:
                geom_ids.append(geom_id)
        return geom_ids

    def _get_obs(self, obs):
        observation = OrderedDict()
        for k, v in obs.items():
            if k not in ['object-state', 'image-state', 
                        'robot0_proprio-state', 'frontview_image']:
                observation[k] = v
        return observation

    def reset(self):
        o = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._success = False
        self.sim = self.env.sim
        self._terminal = False
        return self._get_obs(o)

    def compute_reward(self, action):
        r = self.env.reward(action)
        i = {}
        i["episode_success"] = int(self.env._check_success())
        i["grasping_reward"] = 0.25 if self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube) else 0
        i["success_reward"] = 2.25 if self.env._check_success() else 0.0
        i["reaching_reward"] = r - i["grasping_reward"] - i["success_reward"]
        return r, i
    
    def step(self, action, is_planner=False):
        if type(action) == OrderedDict:
            action = action["default"]
        else:
            raise NotImplementedError
        
        # if not is_planner:
        #     action[:self.robot_dof] *= self._ac_scale

        # see if there is some inherent scaling down in robosuite
        o, r, d, i = self.env.step(action) 
        o = self._get_obs(o)
        #r = self.env.reward()
        self._success = self.env._check_success()
        self._episode_length += 1
        d = self._success or self._episode_length >= self.horizon
        self._episode_reward += r
        i = {}
        i["episode_success"] = int(self._success)
        i["grasping_reward"] = 0.25 if self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube) else 0
        i["success_reward"] = 2.25 if self.env._check_success() else 0.0
        i["reaching_reward"] = r - i["grasping_reward"] - i["success_reward"]
        self.sim = self.env.sim
        self._terminal = d
        return self._get_obs(o), r, d, i

    def _after_step(self, r, d, i):
        self._episode_reward += r
        self._episode_length += 1
        self._terminal = d or self._episode_length >= self.horizon
        return d, i, None

    def render(self, render_mode):
        img = self.env._get_observations()['image-state']/255.
        return img

    def get_contact_force(self):
        mjcontacts = self.env.sim.data.contact
        ncon = self.env.sim.data.ncon
        # total_contact_force = np.zeros(6, dtype=np.float64)
        total_contact_force = 0.0
        for i in range(ncon):
            contact = mjcontacts[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(
                self.sim.model, self.env.sim.data, i, c_array
            )
            total_contact_force += np.sum(np.abs(c_array))

        return total_contact_force

    def form_action(self, next_qpos, curr_qpos=None):
        if curr_qpos is None:
            curr_qpos = self.env.sim.data.qpos.copy()
        joint_ac = (
            next_qpos[self.ref_joint_pos_indexes]
            - curr_qpos[self.ref_joint_pos_indexes]
        )
        if self.dof == 8:
            gripper = (
                next_qpos[self.ref_gripper_joint_pos_indexes]
                - curr_qpos[self.ref_gripper_joint_pos_indexes]
            )
            gripper_ac = gripper[0]
            ac = OrderedDict([("default", np.concatenate([joint_ac, [gripper_ac]]))])
        else:
            ac = OrderedDict([("default", joint_ac)])
        return ac


def make_standard_environment():
    expl_environment_kwargs = {
            "control_freq": 20,
            "controller_configs": {
            "control_delta": True,
            "damping": 1,
            "damping_limits": [
                0,
                10
            ],
            "impedance_mode": "fixed",
            "input_max": 1,
            "input_min": -1,
            "interpolation": None,
            "kp": 150,
            "kp_limits": [
                0,
                300
            ],
            "orientation_limits": None,
            "output_max": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5
            ],
            "output_min": [
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5
            ],
            "position_limits": None,
            "ramp_ratio": 0.2,
            "type": "JOINT_POSITION",
            "uncouple_pos_ori": True
            },
            "env_name": "Lift",
            "horizon": 250,
            "ignore_done": True,
            "reward_shaping": True,
            "robots": "Sawyer",
            "use_object_obs": True,
            "camera_names":"frontview"
        }
    env = suite.make(**expl_environment_kwargs)
    return env
        