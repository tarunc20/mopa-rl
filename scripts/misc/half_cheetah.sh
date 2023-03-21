#!/bin/bash -x
gpu=$1
seed=$2
prefix="BASELINE"
env="HalfCheetah-v2"
algo='sac'
max_episode_step="250"
debug="False"
log_root_dir="./logs"
vis_replay="True"
plot_type='3d'
reward_scale="10."
use_ik_target="False"
ik_target="grip_site"
action_range="0.001"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --reward_scale $reward_scale \
    --use_ik_target $use_ik_target \
    --ik_target $ik_target \
    --action_range $action_range
