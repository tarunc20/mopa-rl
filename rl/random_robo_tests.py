import robosuite as suite
import numpy as np
import time
from robosuite.controllers import controller_factory
from robosuite import load_controller_config 

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

config = load_controller_config(default_controller="JOINT_POSITION")
config['kp'] = 150
config['output_min'] = -0.5
config['output_max'] = 0.5
np.set_printoptions(suppress=True)
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
        0.5,
      ],
      "output_min": [
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
        -0.5,
      ],
      "position_limits": None,
      "ramp_ratio": 0.2,
      "type": "OSC_POSE",
      "uncouple_pos_ori": True
    },
    "env_name": "Lift",
    "horizon": 250,
    "ignore_done": True,
    "reward_shaping": True,
    "robots": "Sawyer",
    "use_object_obs": True,
  }
env = suite.make(**expl_environment_kwargs)
o = env.reset()
print(o.keys())
ac = np.zeros(6)
env.step(ac)