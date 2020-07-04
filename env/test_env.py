import env
import gym
from config import argparser
import numpy as np
import time

# mujocopy_render_hack() # rendering fix for gautam
is_save_video = False
parser = argparser()
args, unparsed = parser.parse_known_args()

if 'reacher' in args.env:
    from config.reacher import add_arguments
elif 'pusher' in args.env:
    from config.pusher import add_arguments
elif 'mover' in args.env:
    from config.mover import add_arguments
elif 'robosuite' in args.env:
    from config.robosuite import add_arguments
elif 'sawyer' in args.env:
    from config.sawyer import add_arguments
elif 'jaco' in args.env:
    from config.jaco import add_arguments

else:
    raise ValueError('args.env (%s) is not supported' % args.env)

add_arguments(parser)
args, unparsed = parser.parse_known_args()
env = gym.make(args.env, **args.__dict__)
if 'robosuite' in args.env:
    env.use_camera_obs = False
obs = env.reset()
obs = env.reset()
# env.render('human') # uncomment if you don't use mujocopy hack

#for i in range(10000):
timestep = 0
env.reset_visualized_indicator()
while True:
    # env.render(mode='rgb_array')
    action = env.action_space.sample()
    # action = np.ones(env.dof)
    # action = np.zeros(env.dof)
    # action[0] = -0.5
    # action[1] = -0.5
    # qpos = env.sim.data.qpos.ravel().copy()[env.ref_joint_pos_indexes].copy() + action['default'][:env.mujoco_robot.dof]
    # env.set_robot_indicator_joint_positions(qpos)
    obs, reward, done, _ = env.step(action)
    env.render(mode='human')
    if env.sim.data.ncon > 0:
        for i in range(env.sim.data.ncon):
            print(env.sim.data.contact[i].dist)
    if done:
        print('done')
        break
