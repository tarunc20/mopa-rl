import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        "HRL-Planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # environment
    parser.add_argument("--env", type=str, default="reacher-obstacle-v0",
                        help="environment name")
    parser.add_argument("--env_args", type=str, default=None)

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "ppo", "ddpg", "td3"])
    parser.add_argument("--policy", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    parser.add_argument("--meta_update_target", type=str, default="HL",
                        choices=['HL', 'LL', 'both'])

    # hrl
    parser.add_argument("--hrl", type=str2bool, default=False,
                        help="whether to use HRL or not")
    parser.add_argument("--hrl_network_to_update", type=str, default="HL",
                        choices=["HL", "LL", "both"])
    parser.add_argument("--primitive_dir", type=str, default=None,
                        help="path to primitive directory")
    parser.add_argument("--max_meta_len", type=int, default=25)

    parser.add_argument("--ll_type", type=str, default="rl",
                        help="low level controller choice", choices=["rl", "mp"])
    parser.add_argument("--primitive_skills", nargs='+', default=['skill'])
    parser.add_argument("--hl_type", type=str, default='discrete',
                        choices=['discrete', 'subgoal'])
    parser.add_argument("--meta_subgoal_rew", type=float, default=-100)

    parser.add_argument("--kinematics", type=str2bool, default=False)

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=64)
    parser.add_argument("--rl_activation", type=str, default="relu",
                        choices=["relu", "elu", "tanh"])
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--meta_tanh_policy", type=str2bool, default=False)
    parser.add_argument("--subgoal_type", type=str, default='joint', choices=['joint', 'cart'])

    parser.add_argument("--kernel_size", nargs='+', default=[3, 3, 3])
    parser.add_argument("--conv_dim", nargs='+', default=[32, 64, 32])
    parser.add_argument("--stride", nargs='+', default=[2, 1, 1])

    # observation normalization
    parser.add_argument("--ob_norm", type=str2bool, default=True)
    parser.add_argument("--max_ob_norm_step", type=int, default=int(1e7))
    parser.add_argument("--clip_obs", type=float, default=200, help="the clip range of observation")
    parser.add_argument("--clip_range", type=float, default=5, help="the clip range after normalization of observation")

    # motion planning
    parser.add_argument("--use_ik", type=str2bool, default=False)

    # off-policy rl
    parser.add_argument("--buffer_size", type=int, default=int(1e3), help="the size of the buffer")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="the discount factor")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="the learning rate of the actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="the learning rate of the critic")
    parser.add_argument("--polyak", type=float, default=0.995, help="the average coefficient")
    parser.add_argument("--use_ae", type=str2bool, default=False, help="use AutoEncoder")
    parser.add_argument("--encoder_kernel_size", nargs='+', default=[3, 3, 3])
    parser.add_argument("--encoder_conv_dim", nargs='+', default=[32, 64, 32])
    parser.add_argument("--encoder_stride", nargs='+', default=[2, 2, 1])
    parser.add_argument("--decoder_kernel_size", nargs='+', default=[3, 3, 3])
    parser.add_argument("--decoder_conv_dim", nargs='+', default=[64, 32, 3])
    parser.add_argument("--decoder_stride", nargs='+', default=[1, 2, 2])
    parser.add_argument("--decoder_out_padding", nargs='+', default=[0, 0, 1])
    parser.add_argument("--decoder_padding", nargs='+', default=[0, 0,  0])
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument("--ae_feat_dim", type=int, default=50)
    parser.add_argument("--actor_update_freq", type=int, default=1)
    parser.add_argument("--critic_target_update_freq", type=int, default=1)

    parser.add_argument("--lr_encoder", type=float, default=1e-3, help="the learning rate of the encoder")
    parser.add_argument("--lr_decoder", type=float, default=1e-3, help="the learning rate of the decoder")

    parser.add_argument("--mp_ratio", type=float, default=1., help='How often a motion planner is used')


    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--num_batches", type=int, default=50,
                        help="the times to update the network per epoch")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="the sample batch size")
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--max_global_step", type=int, default=int(2e6))
    parser.add_argument("--gpu", type=int, default=None)

    # sac
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")

    # ppo
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)
    parser.add_argument("--rollout_length", type=int, default=1000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # log
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--evaluate_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument("--wandb", type=str2bool, default=False,
                        help="set it True if you want to use wandb")
    parser.add_argument("--group", type=str, default=None)

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--save_rollout", type=str2bool, default=False,
                        help="save rollout information during evaluation")
    parser.add_argument("--record", type=str2bool, default=True,
                        help="enable video recording")
    parser.add_argument("--record_caption", type=str2bool, default=True)
    parser.add_argument("--num_record_samples", type=int, default=1,
                        help="number of trajectories to collect during eval")
    parser.add_argument("--save_qpos", type=str2bool, default=False,
                        help="save entire qpos history of success rollouts to file (for idle primitive training)")
    parser.add_argument("--save_success_qpos", type=str2bool, default=True,
                        help="save later segment of success rollouts to file (for moving and placing primitie trainings)")

    # misc
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--debug_render", type=str2bool, default=False)

    return parser
