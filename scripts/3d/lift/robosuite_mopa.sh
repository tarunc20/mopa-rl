#!/bin/bash -x
gpu=$1
seed=$2
prefix="BASELINE"
env="Lift"
algo='sac'
debug="False"
log_root_dir="./logs"
mopa="True"
vis_replay="True"
plot_type='3d'
reward_scale="10."
use_ik_target="False"
ik_target="grip_site"
action_range="0.001"
lr_actor="0.001"
lr_critic="0.0005"
num_record_samples="1"
omega="0.5"
action_range="1.0"
invalid_target_handling="True"

python -m rl.main \
    --log_root_dir $log_root_dir \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --mopa $mopa \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --reward_scale $reward_scale \
    --use_ik_target $use_ik_target \
    --ik_target $ik_target \
    --action_range $action_range \
    --lr_actor $lr_actor \
    --lr_critic $lr_critic \
    --num_record_samples $num_record_samples \
    --omega $omega \
    --action_range $action_range \
    --invalid_target_handling $invalid_target_handling
    
